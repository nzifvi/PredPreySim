import torch
import pybullet


class Agent:
    def __init__(self, position, isPredator, agentConfig, device="cpu"):
        self.isPredator = isPredator
        self.agentConfig = agentConfig
        self.device = device

        self.radius = agentConfig.radius
        self.colour = agentConfig.colour

        self.agent = self._initBody(position)

        self.position = torch.zeros(2, device=device)
        self.velocity = torch.zeros(2, device=device)

        self.isAlive = True
        self.timeAlive = 0.0

    def _initBody(self, position):
        visual = pybullet.createVisualShape(
            pybullet.GEOM_BOX,
            halfExtents=[self.agentConfig.radius, self.agentConfig.radius, self.agentConfig.radius],
            rgbaColor=self.colour
        )
        collision = pybullet.createCollisionShape(
            pybullet.GEOM_BOX,
            halfExtents=[self.agentConfig.radius, self.agentConfig.radius, self.agentConfig.radius]
        )
        agent = pybullet.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[position[0], position[1], self.agentConfig.radius]
        )
        pybullet.changeDynamics(
            agent,
            -1,
            lateralFriction=0.1,
            spinningFriction=0.001,
            rollingFriction=0.001,
            linearDamping=0.5,
            angularDamping=0.9
        )
        return agent

    def updateState(self) -> None:
        currentPosition, _ = pybullet.getBasePositionAndOrientation(self.agent)
        currentVelocity, _ = pybullet.getBaseVelocity(self.agent)

        self.position = torch.tensor(
            currentPosition[:2],
            dtype=torch.float32,
            device=self.device
        )
        self.velocity = torch.tensor(
            currentVelocity[:2],
            dtype=torch.float32,
            device=self.device
        )

    def applyAction(self, action):
        if not self.isAlive:
            return

        action = torch.clamp(action, -1.0, 1.0).cpu().numpy()
        force = action * self.agentConfig.maxSpeed * 10.0

        pybullet.applyExternalForce(
            self.agent,
            -1,
            forceObj=[force[0], force[1], 0.0],
            posObj=[0, 0, 0],
            flags=pybullet.LINK_FRAME
        )
        self._applyBoundaryForce()

    def _applyBoundaryForce(self):
        currentPosition = self.position.cpu().numpy()

        size = self.agentConfig.arenaSize
        margin = self.agentConfig.boundaryMargin
        k = self.agentConfig.boundaryForce

        force = [0.0, 0.0]

        if currentPosition[0] > size / 2 - margin:
            force[0] = -k * (currentPosition[0] - (size / 2 - margin))
        elif currentPosition[0] < -size / 2 + margin:
            force[0] = -k * (currentPosition[0] - (-size / 2 + margin))

        if currentPosition[1] > size / 2 - margin:
            force[1] = -k * (currentPosition[1] - (size / 2 - margin))
        elif currentPosition[1] < -size / 2 + margin:
            force[1] = -k * (currentPosition[1] - (-size / 2 + margin))

        if force[0] != 0.0 or force[1] != 0.0:
            pybullet.applyExternalForce(
                self.agent,
                -1,
                forceObj=[float(force[0]), float(force[1]), 0.0],
                posObj=[0, 0, 0],
                flags=pybullet.WORLD_FRAME
            )

    def getObservation(self, predators, prey, visionRadius):
        size = self.agentConfig.arenaSize

        obs = [self.velocity]
        obs.append(torch.tensor(
            [
                (size / 2 - self.position[0]) / size,
                (size / 2 + self.position[0]) / size,
                (size / 2 - self.position[1]) / size,
                (size / 2 + self.position[1]) / size,
            ],
            dtype=torch.float32,
            device=self.device
        ))

        obs.append(self._findNearestEntity(predators, visionRadius))

        nearestPrey, averageDirection = self._findNearestPreyAndAvgDirection(prey, visionRadius)
        obs.append(nearestPrey)
        obs.append(averageDirection)

        return torch.cat(obs).view(1, -1)

    def _findNearestEntity(self, agents, visionRadius):
        minDistance = torch.tensor(visionRadius, device=self.device, dtype=torch.float32)
        direction = torch.zeros(2, device=self.device)
        foundEntity = False

        for a in agents:
            if a.agent == self.agent or not a.isAlive:
                continue

            distanceDifference = a.position - self.position
            distance = torch.norm(distanceDifference)

            if distance > visionRadius:
                continue

            if distance < minDistance:
                minDistance = distance
                direction = distanceDifference / (distance + 1e-6)
                foundEntity = True

        if not foundEntity:
            normalizedDistance = torch.tensor([1.0], device=self.device)
            return torch.cat([torch.zeros(2, device=self.device), normalizedDistance])

        normalizedDistance = (minDistance / visionRadius).unsqueeze(0)
        return torch.cat([direction, normalizedDistance])

    def _findNearestPreyAndAvgDirection(self, agents, visionRadius):
        minDistance = torch.tensor(visionRadius, device=self.device, dtype=torch.float32)
        nearestDirection = torch.zeros(2, device=self.device)
        averageDirection = torch.zeros(2, device=self.device)

        count = 0
        foundEntity = False

        for a in agents:
            if a.agent == self.agent or not a.isAlive:
                continue

            distanceDifference = a.position - self.position
            distance = torch.norm(distanceDifference)

            if distance > visionRadius:
                continue

            if distance < minDistance:
                minDistance = distance
                nearestDirection = distanceDifference / (distance + 1e-6)
                foundEntity = True

            averageDirection += distanceDifference
            count += 1

        if count > 0:
            averageDirection /= count
            averageDirection /= (torch.norm(averageDirection) + 1e-6)
        else:
            averageDirection = torch.zeros(2, device=self.device)

        if not foundEntity:
            nearest = torch.cat([
                torch.zeros(2, device=self.device),
                torch.tensor([1.0], device=self.device)
            ])
        else:
            nearest = torch.cat([
                nearestDirection,
                (minDistance / visionRadius).unsqueeze(0)
            ])

        return nearest, averageDirection

    def kill(self):
        self.isAlive = False

        pybullet.resetBasePositionAndOrientation(
            self.agent,
            [self.position[0].item(), self.position[1].item(), -10],
            [0, 0, 0, 1]
        )

    def stepTime(self, dt):
        if self.isAlive:
            self.timeAlive += dt

    def getMessageContext(self, predators, prey, visionRadius):
        if self.isPredator:
            allies = predators
            enemies = prey
        else:
            allies = prey
            enemies = predators

        nearestEnemyDist = visionRadius
        nearestAllyDist  = visionRadius

        visibleEnemyCount = 0
        visibleAllyCount  = 0

        for a in enemies:
            if a.agent == self.agent or not a.isAlive:
                continue

            dist = torch.norm(a.position - self.position).item()
            if dist <= visionRadius:
                visibleEnemyCount += 1
                nearestEnemyDist = min(nearestEnemyDist, dist)

        for a in allies:
            if a.agent == self.agent or not a.isAlive:
                continue

            dist = torch.norm(a.position - self.position).item()
            if dist <= visionRadius:
                visibleAllyCount += 1
                nearestAllyDist = min(nearestAllyDist, dist)

        return {
            "nearestEnemyDist"  : nearestEnemyDist,
            "nearestAllyDist"   : nearestAllyDist,
            "visibleEnemyCount" : visibleEnemyCount,
            "visibleAllyCount"  : visibleAllyCount
        }