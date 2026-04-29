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

        self.maxEnergy        = agentConfig.maxEnergy
        self.energy           = self.maxEnergy
        self.energyDrainRate  = agentConfig.energyDrainRate

        self.baseSpeed        = agentConfig.baseSpeed
        self.sprintDrainRate  = agentConfig.sprintDrainRate
        self.sprintMultiplier = agentConfig.sprintMultiplier
        self.isSprinting      = False

        self.stomachCapacity    = 0.0
        self.maxStomachCapacity = agentConfig.maxStomachCapacity

        self.position = torch.zeros(2, device=device)
        self.velocity = torch.zeros(2, device=device)

        self.isAlive   = True
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

    def updateState(self, dt) -> None:
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

        if self.isAlive:
            self._drainEnergy(dt)

    def applyAction(self, action):
        if not self.isAlive:
            return

        xVelocity = action[0].item()
        yVelocity = action[1].item()
        if len(action >= 3):
            sprintSignal = action[2].item()
            self.isSprinting = (sprintSignal > 0.5 and self.energy > self.sprintDrainRate)
        else:
            self.isSprinting = False

        xVelocity = torch.clamp(
            torch.tensor(xVelocity), -5.0, 5.0
        ).item()
        yVelocity = torch.clamp(
            torch.tensor(yVelocity), -5.0, 5.0
        ).item()

        speedMultiplier = self.sprintMultiplier if self.isSprinting else 1.0

        force = [
            xVelocity * self.baseSpeed * speedMultiplier,
            yVelocity * self.baseSpeed * speedMultiplier,
            0.0 # no z-axis force allowed.
        ]

        pybullet.applyExternalForce(
            self.agent,
            -1,
            forceObj=force,
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

    def getObservation(self, predators, prey, visionRadius, foodSources = None):
        size = self.agentConfig.arenaSize

        if self.isPredator:
            allies  = predators
            enemies = prey
        else:
            allies  = prey
            enemies = predators

        normalisedVelocity = torch.clamp(
            self.velocity, # NORMALISE LATER
            min = - 1.0,
            max = 1.0
        )

        wallDistances = torch.tensor(
            [
                (size / 2 - self.position[0]) / size,
                (size / 2 + self.position[0]) / size,
                (size / 2 - self.position[1]) / size,
                (size / 2 + self.position[1]) / size
            ],
            dtype = torch.float32,
            device = self.device
        )

        nearestAlly  = self._findNearestEntity(allies, visionRadius)
        nearestEnemy = self._findNearestEntity(enemies, visionRadius)

        avgAllyDirection, visibleAllyCount = self._findAvgDirectionAndCount(allies, visionRadius)
        avgEnemyDirection, visibleEnemyCount = self._findAvgDirectionAndCount(enemies, visionRadius)

        localNumericalAdvantage = torch.tensor(
            [
                (visibleAllyCount - visibleEnemyCount) / max(1, len(allies))
            ],
            dtype = torch.float32,
            device = self.device
        )

        energyRatio = torch.tensor(
            [self.energy / self.maxEnergy],
            dtype  = torch.float32,
            device = self.device
        )

        if foodSources is not None:
            nearestFood = self._findNearestFood(foodSources, visionRadius)
        else:
            nearestFood = torch.zeros(
                3,
                dtype  = torch.float32,
                device = self.device
            )

        observations = [
            normalisedVelocity,
            wallDistances,
            nearestAlly,
            nearestEnemy,
            avgAllyDirection,
            avgEnemyDirection,
            torch.tensor([visibleAllyCount], dtype = torch.float32, device = self.device),
            torch.tensor([visibleEnemyCount], dtype = torch.float32, device = self.device),
            localNumericalAdvantage,
            energyRatio,
            nearestFood
        ]

        return torch.cat(observations).view(1, -1)

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

    def _findAvgDirectionAndCount(self, agents:list, visionRadius:float) -> tuple:
        avgDirection = torch.zeros(2, device=self.device)
        count = 0

        for a in agents:
            if a.agent == self.agent or not a.isAlive:
                continue

            distanceDiff = a.position - self.position
            distance = torch.norm(distanceDiff)

            if distance > visionRadius:
                continue
            avgDirection += distanceDiff
            count += 1

        if count > 0:
            avgDirection /= count
            avgDirection /= (torch.norm(avgDirection) + 1e-6)
        else:
            avgDirection = torch.zeros(2, device=self.device)

        return avgDirection, count

    def _findNearestFood(self, foodSources, visionRadius) -> torch.tensor:
        nearestFoodPosition = None
        minDistance         = visionRadius

        for food in foodSources:
            if not food["available"]:
                continue

            foodPos = torch.tensor(
                food["position"],
                dtype = torch.float32,
                device = self.device
            )
            diff = foodPos - self.position
            dist = torch.norm(diff).item()
            if dist < visionRadius:
                minDistance    = dist
                nearestFoodPosition = diff

        if nearestFoodPosition is None:
            return torch.zeros(
                3,
                dtype = torch.float32,
                device = self.device
            )
        else:
            direction = nearestFoodPosition / (torch.norm(nearestFoodPosition) + 1e-6)
            normalisedDistance = torch.tensor(
                [minDistance / visionRadius],
                dtype = torch.float32,
                device = self.device
            )
            return torch.cat([direction, normalisedDistance])

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

    def _drainEnergy(self, dt):
        drainAmount = self.energyDrainRate * dt
        if self.isSprinting:
            drainAmount += self.sprintDrainRate * dt

        self.energy -= drainAmount

        if self.energy <= 0:
            self.energy = 0
            if self.isAlive:
                self.kill()

    def eat(self, energyGained):
        self.energy = min(
            self.energy + energyGained, self.maxEnergy
        )