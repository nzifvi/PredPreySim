import pybullet
import pybullet_data
import numpy
import time
import torch

import Agent
import SimulationConfig
import sys
import os


class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

class Simulator:
    def __init__(self, simConfig=SimulationConfig.SimulationConfig, numPredators=4, numPrey=8, simDuration=30.0, gui=True):
        self.numPredators = numPredators
        self.numPrey = numPrey
        self.simDuration = simDuration
        self.haveGUI = gui

        self.arenaSize           = simConfig.arenaSize
        self.visionRadius        = simConfig.visionRadius
        self.communicationRadius = simConfig.communicationRadius

        self.timeStep = 1.0 / 120.0

        with SuppressOutput():
            if self.haveGUI:
                self.simClient = pybullet.connect(pybullet.GUI)
                self.birdsEyeView()
            else:
                self.simClient = pybullet.connect(pybullet.DIRECT)

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.setGravity(0, 0, -9.81)
        pybullet.setTimeStep(self.timeStep)

        self._initSimulator()

        self.predators = []
        self.prey      = []

        self.messageLog = []

        self.predToPredDistances = None
        self.predToPreyDistances = None
        self.preyToPreyDistances = None

        self.foodSources          = []
        self.minClusterFood       = simConfig.minClusterFood
        self.maxClusterFood       = simConfig.maxClusterFood
        self.preyEnergyValue      = simConfig.preyEnergyValue
        self.arenaFoodEnergyValue = simConfig.arenaFoodEnergyValue

        self.catchDistance = simConfig.catchDistance
        self.eatDistance   = simConfig.eatDistance

    def _initSimulator(self):
        self.plane = pybullet.loadURDF("plane.urdf")

        arenaDelimiter = pybullet.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=[0.2, 0.2, 1.0],
            rgbaColor=[0.5, 0.5, 0.5, 0.5]
        )

        arenaCorners = [
            [self.arenaSize / 2, self.arenaSize / 2, 1],
            [self.arenaSize / 2, -self.arenaSize / 2, 1],
            [-self.arenaSize / 2, self.arenaSize / 2, 1],
            [-self.arenaSize / 2, -self.arenaSize / 2, 1],
        ]

        for arenaCorner in arenaCorners:
            pybullet.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=arenaDelimiter,
                basePosition=arenaCorner,
            )

    def reset(self):
        for agent in self.predators + self.prey:
            pybullet.removeBody(agent.agent)

        self.predators   = []
        self.prey        = []
        self.messageLog  = []
        self.foodSources = []

        for _ in range(self.numPredators):
            startingX = numpy.random.uniform(-self.arenaSize / 3, self.arenaSize / 3)
            startingY = numpy.random.uniform(-self.arenaSize / 3, self.arenaSize / 3)

            predator = Agent.Agent(
                position=[startingX, startingY],
                isPredator=True,
                agentConfig=SimulationConfig.PredatorConfig()
            )
            self.predators.append(predator)

        for _ in range(self.numPrey):
            startingX = numpy.random.uniform(-self.arenaSize / 3, self.arenaSize / 3)
            startingY = numpy.random.uniform(-self.arenaSize / 3, self.arenaSize / 3)

            prey = Agent.Agent(
                position=[startingX, startingY],
                isPredator=False,
                agentConfig=SimulationConfig.PreyConfig()
            )
            self.prey.append(prey)

        clusterAmount = numpy.random.randint(3, 6)
        for i in range(0, clusterAmount):
            clusterXPos = numpy.random.uniform(-self.arenaSize / 3, self.arenaSize / 3)
            clusterYPos = numpy.random.uniform(-self.arenaSize / 3, self.arenaSize / 3)

            clusterRadius = numpy.random.uniform(1.5, 3.0)  # Cluster spread
            foodInCluster = numpy.random.randint(5, 10)

            for _ in range(foodInCluster):
                angle = numpy.random.uniform(0, 2 * numpy.pi)
                distance = numpy.abs(numpy.random.normal(0, clusterRadius / 2))

                xPos = clusterXPos + distance * numpy.cos(angle)
                yPos = clusterYPos + distance * numpy.sin(angle)

                xPos = numpy.clip(xPos, -self.arenaSize/3, self.arenaSize/3)
                yPos = numpy.clip(yPos, -self.arenaSize/3, self.arenaSize/3)

                foodVisual = pybullet.createVisualShape(
                    shapeType = pybullet.GEOM_SPHERE,
                    radius = 0.1,
                    rgbaColor = [0.0, 0.0, 0.0, 1.0]
                )
                foodBody = pybullet.createMultiBody(
                    baseMass = 0,
                    baseVisualShapeIndex = foodVisual,
                    basePosition = [xPos, yPos, 0.0],
                )
                self.foodSources.append({
                    "body" : foodBody,
                    "position" :[xPos, yPos],
                    "available" : True,
                    "respawnTime" : 0.0,
                    "clusterId" : i
                })



        for _ in range(10):
            pybullet.stepSimulation()


    def runSimulation(self, predatorNNs, preyNNs, predatorGenotypeIDs = None, preyGenotypeIDs = None):
        self.reset()

        self._initialiseInternalAgentState(predatorNNs, preyNNs)

        preyCaught                       = [0] * self.numPredators
        predatorTeamHuntScore            = [0.0] * self.numPredators

        predatorCommMagnitudes = [[] for _ in range(self.numPredators)]
        preyCommMagnitudes     = [[] for _ in range(self.numPrey)]

        preyGrouping                     = [0.0] * self.numPrey

        currentTime                      = 0.0
        step                             = 0

        while currentTime < self.simDuration:
            self._updateAllStates(self.timeStep)
            self._computeDistanceMatrices()


            self._updateReceivedMessages()

            predatorActions = self._getPredatorActions(
                predatorNNs,
                predatorGenotypeIDs,
                step,
                predatorCommMagnitudes
            )
            preyActions = self._getPreyActions(
                preyNNs,
                preyGenotypeIDs,
                step,
                preyCommMagnitudes
            )

            self._applyPredatorActions(predatorActions)
            self._applyPreyActions(preyActions)

            self._updatePreyEating(currentTime)
            self._respawnFood(currentTime)

            pybullet.stepSimulation()

            self._updateAllStates(self.timeStep)
            self._computeDistanceMatrices()

            self._processCatches(preyCaught, predatorTeamHuntScore)
            self._updatePredatorTeamHuntScore(predatorTeamHuntScore)
            self._updatePreyGrouping(preyGrouping)

            if self.haveGUI and step % 5 == 0:
                self.visualiseCommunication()
                time.sleep(self.timeStep * 2)

            currentTime += self.timeStep
            step += 1

        return {
            "predators": [
                {
                    "catches"                 : preyCaught[i],
                    "teamHuntScore"           : predatorTeamHuntScore[i] / max(step, 1),
                    "meanCommMagnitude"       : (
                        sum(predatorCommMagnitudes[i]) / len(predatorCommMagnitudes[i]) if len(predatorCommMagnitudes[i]) > 0 else 0.0
                    ),
                    "timeAlive"               : self.predators[i].timeAlive,
                    "finalEnergy"             : self.predators[i].energy,
                }
                for i in range(len(self.predators))
            ],
            "prey": [
                {
                    "timeAlive"                   : self.prey[i].timeAlive,
                    "groupingScore"               : preyGrouping[i] / max(step, 1),
                    "meanCommMagnitude"           : (
                        sum(preyCommMagnitudes[i]) / len(preyCommMagnitudes[i]) if len(preyCommMagnitudes[i]) > 0 else 0.0
                    ),
                    "finalEnergy"                 : self.prey[i].energy,
                }
                for i in range(len(self.prey))
            ],
            "messageLog" : self.messageLog
        }

    def _initialiseInternalAgentState(self, predatorNNs, preyNNs):
        for predator, nn in zip(self.predators, predatorNNs):
            predator.hiddenState = nn.initHidden()
            predator.message = torch.zeros(1, 2, dtype=torch.float32)
            predator.receivedMessage = torch.zeros(1, 2, dtype=torch.float32)

        for prey, nn in zip(self.prey, preyNNs):
            prey.hiddenState = nn.initHidden()
            prey.message = torch.zeros(1, 2, dtype=torch.float32)
            prey.receivedMessage = torch.zeros(1, 2, dtype=torch.float32)

    def _updateAllStates(self, dt:float):
        for agent in self.predators + self.prey:
            agent.updateState(dt)

    def _averageAllyMessages(self, agents, currentAgent):
        messages = []

        for agent in agents:
            if agent is currentAgent:
                continue
            if not agent.isAlive:
                continue
            if agent.message is None:
                continue

            dist = torch.norm(agent.position - currentAgent.position)
            if dist > self.communicationRadius:
                continue

            messages.append(agent.message)

        if len(messages) == 0:
            return torch.zeros(1, 2, dtype=torch.float32)

        return torch.mean(
            torch.stack(messages, dim = 0),
            dim = 0
        )

    def _updateReceivedMessages(self):
        for predator in self.predators:
            predator.receivedMessage = self._averageAllyMessages(self.predators, predator)

        for prey in self.prey:
            if not prey.isAlive:
                prey.receivedMessage = torch.zeros(1, 2, dtype=torch.float32)
                continue

            prey.receivedMessage = self._averageAllyMessages(self.prey, prey)

    def _getPredatorActions(self, predatorNNs, predatorGenotypeIDs, step, predatorCommMagnitudes) -> list:
        predatorActions = []

        for i, predator in enumerate(self.predators):
            if not predator.isAlive:
                predator.message = None
                predator.receivedMessage = torch.zeros(
                    1,
                    2,
                    dtype = torch.float32
                )
                continue

            predatorObservation = predator.getObservation(
                predators    = self.predators,
                prey         = self.prey,
                visionRadius = self.visionRadius,
                foodSources  = self.foodSources
            ).view(1, 23)

            with torch.no_grad():
                movement, communication, newHidden = predatorNNs[i].forward(
                    predatorObservation,
                    predator.hiddenState,
                    predator.receivedMessage
                )

            predator.hiddenState = newHidden
            predator.message = communication
            predatorCommMagnitudes[i].append(
                communication.abs().mean().item()
            )

            messageContext = predator.getMessageContext(
                self.predators,
                self.prey,
                self.visionRadius
            )
            genotypeID = predatorGenotypeIDs[i] if predatorGenotypeIDs is not None else i
            self._logMessageEvent(
                step = step,
                species = "predator",
                genotypeID = genotypeID,
                agentIndex = i,
                agent = predator,
                outgoingMessage = communication,
                receivedMessage = predator.receivedMessage,
                movement = movement,
                nearestEnemyDist = messageContext["nearestEnemyDist"],
                nearestAllyDist = messageContext["nearestAllyDist"],
                visibleEnemyCount = messageContext["visibleEnemyCount"],
                visibleAllyCount = messageContext["visibleAllyCount"],
            )

            predatorActions.append(movement.flatten())

        return predatorActions

    def _getPreyActions(self, preyNNs, preyGenotypeIDs, step, preyCommMagnitudes) -> list:
        preyActions = []

        for i, prey in enumerate(self.prey):
            if not prey.isAlive:
                prey.message = None
                prey.receivedMessage = torch.zeros(1, 2, dtype=torch.float32)
                preyActions.append(None)
                preyCommMagnitudes[i].append(0.0)
                continue

            preyObservation = prey.getObservation(
                predators    = self.predators,
                prey         = self.prey,
                visionRadius = self.visionRadius,
                foodSources  = self.foodSources
            ).view(1, 23)

            with torch.no_grad():
                movement, communication, newHidden = preyNNs[i].forward(
                    preyObservation,
                    prey.hiddenState,
                    prey.receivedMessage
                )

            prey.hiddenState = newHidden
            prey.message = communication
            preyCommMagnitudes[i].append(
                communication.abs().mean().item()
            )

            messageContext = prey.getMessageContext(
                self.predators,
                self.prey,
                self.visionRadius
            )

            genotypeID = preyGenotypeIDs[i] if preyGenotypeIDs is not None else i

            self._logMessageEvent(
                step = step,
                species = "prey",
                genotypeID = genotypeID,
                agentIndex = i,
                agent = prey,
                outgoingMessage = communication,
                receivedMessage = prey.receivedMessage,
                movement = movement,
                nearestEnemyDist = messageContext["nearestEnemyDist"],
                nearestAllyDist = messageContext["nearestAllyDist"],
                visibleEnemyCount = messageContext["visibleEnemyCount"],
                visibleAllyCount = messageContext["visibleAllyCount"]
            )

            preyActions.append(movement.flatten())

        return preyActions

    def _applyPredatorActions(self, predatorActions) -> None:
        for predator, action in zip(self.predators, predatorActions):
            if not predator.isAlive or action is None:
                continue

            predator.applyAction(action)
            predator.stepTime(self.timeStep)

    def _applyPreyActions(self, preyActions) -> None:
        for prey, action in zip(self.prey, preyActions):
            if not prey.isAlive or action is None:
                continue

            prey.applyAction(action)
            prey.stepTime(self.timeStep)

    def _processCatches(self, preyCaught, predatorTeamHuntScore = None) -> None:
        catchSupportRadius = 3.0

        for i, predator in enumerate(self.predators):
            if not predator.isAlive:
                continue

            for j, prey in enumerate(self.prey):
                if not prey.isAlive:
                    continue

                dist = self.predToPreyDistances[i, j]
                if dist < self.catchDistance:
                    nearbyPredatorIndices = []

                    for k, otherPredator in enumerate(self.predators):
                        if not otherPredator.isAlive:
                            continue

                        supportDist = self.predToPreyDistances[k, j].item()
                        if supportDist <= catchSupportRadius:
                            nearbyPredatorIndices.append(k)

                    prey.kill()
                    prey.message = None
                    prey.receivedMessage = torch.zeros(
                        1,
                        2,
                        dtype=torch.float32
                    )
                    preyCaught[i] += 1

                    predator.eat(
                        self.preyEnergyValue
                    )

                    if predatorTeamHuntScore is not None and len(nearbyPredatorIndices) >= 2:
                        for predatorIndex in nearbyPredatorIndices:
                            predatorTeamHuntScore[predatorIndex] += 5.0

    def disconnect(self):
        pybullet.disconnect()

    def _logMessageEvent(self, *, step, species, genotypeID, agentIndex, agent, outgoingMessage, receivedMessage, movement, nearestEnemyDist, nearestAllyDist, visibleEnemyCount, visibleAllyCount):
        self.messageLog.append(
            {
                "step" : step,
                "species" : species,
                "genotypeID" : genotypeID,
                "agentIndex" : agentIndex,
                "x" : float(agent.position[0].item()),
                "y" : float(agent.position[1].item()),
                "vx" : float(agent.velocity[0].item()),
                "vy" : float(agent.velocity[1].item()),
                "alive" : bool(agent.isAlive),
                "msg0" : float(outgoingMessage[0, 0].item()),
                "msg1" : float(outgoingMessage[0, 1].item()),

                "receivedMessage0" : float(receivedMessage[0, 0].item()),
                "receivedMessage1" : float(receivedMessage[0, 1].item()),

                "move0" : float(movement[0, 0].item()),
                "move1" : float(movement[0, 1].item()),

                "nearestEnemyDist" : float(nearestEnemyDist),
                "nearestAllyDist" : float(nearestAllyDist),
                "visibleEnemyCount" : int(visibleEnemyCount),
                "visibleAllyCount" : int(visibleAllyCount),
            }
        )

    def _updatePreyGrouping(self, preyGrouping):
        for i, prey in enumerate(self.prey):
            if not prey.isAlive:
                continue

            groupingCount = 0
            for j, other in enumerate(self.prey):
                if i == j or not other.isAlive:
                    continue

                dist = self.preyToPreyDistances[i, j]
                if dist < 2.5:
                    groupingCount += 1

            preyGrouping[i] += groupingCount

    def _updatePredatorTeamHuntScore(self, predatorTeamHuntScore) -> None:
        teamHuntRadius = 3.0
        minRequiredPredators = 2

        for j, prey in enumerate(self.prey):
            if not prey.isAlive:
                continue

            nearbyPredatorIndices = []

            for i, predator in enumerate(self.predators):
                if not predator.isAlive:
                    continue

                dist = self.predToPreyDistances[i, j].item()
                if dist < teamHuntRadius:
                    nearbyPredatorIndices.append(i)

            if len(nearbyPredatorIndices) >= minRequiredPredators:
                for pIndex in nearbyPredatorIndices:
                    predatorTeamHuntScore[pIndex] += 1.0

    def _debugObservations(self, label, obs, isPredator) -> None:
        obs = obs.view(-1).cpu().numpy()

        print(f"{label}:")
        print(f"    - velocity : (x = {obs[0]:.3f}, y = {obs[1]:.3f})")
        print(f"    - walls    :    ({obs[2]:.3f}, {obs[3]:.3f}, {obs[4]:.3f}, {obs[5]:.3f})")
        if isPredator:
            print(f"    - nearestAllyPredator : direction = ({obs[6]:.3f}, {obs[7]:.3f}, {obs[8]:.3f})")
            print(f"    - nearestPrey         : direction = ({obs[9]:.3f}, {obs[10]:.3f}, {obs[11]:.3f})")
            print(f"    - avgPreyDirection    : ({obs[12]:.3f}, {obs[13]:.3f}")
        else:
            print(f"    - nearestPredator : direction = ({obs[6]:.3f}, {obs[7]:.3f}, {obs[8]:.3f})")
            print(f"    - nearestAllyPrey         : direction = ({obs[9]:.3f}, {obs[10]:.3f}, {obs[11]:.3f})")
            print(f"    - avgAllyPreyDirection    : ({obs[12]:.3f}, {obs[13]:.3f}")

    def _computeDistanceMatrices(self) -> None:
        predPositions = torch.stack(
            [p.position for p in self.predators]
        )
        preyPositions = torch.stack(
            [p.position for p in self.prey]
        )

        # Predator-to-Predator distances
        predDiff = predPositions.unsqueeze(1) - predPositions.unsqueeze(0)  # ← Fixed: pred - pred
        self.predToPredDistances = torch.norm(
            predDiff,
            dim = 2
        )

        # Predator-to-Prey distances
        predPreyDiff = predPositions.unsqueeze(1) - preyPositions.unsqueeze(0)
        self.predToPreyDistances = torch.norm(
            predPreyDiff,
            dim = 2
        )

        # Prey-to-Prey distances
        preyDiff = preyPositions.unsqueeze(1) - preyPositions.unsqueeze(0)
        self.preyToPreyDistances = torch.norm(  # ← Fixed: preyToPrey not preyToPred
            preyDiff,
            dim = 2
        )

    def birdsEyeView(self) -> None:
        if not self.haveGUI:
            return
        pybullet.resetDebugVisualizerCamera(
            cameraDistance = 25,
            cameraYaw      = 0,
            cameraPitch    = -89,
            cameraTargetPosition = [0, 0, 0]
        )

    def visualiseCommunication(self) -> None:
        if not self.haveGUI:
            return

        if not hasattr(self, '_messageLineIDs'):
            self._messageLineIDs = []

        for lineID in self._messageLineIDs:
            pybullet.removeUserDebugItem(lineID)
        self._messageLineIDs = []

        for p in self.predators:
            if hasattr(p, 'lastMessage') and p.lastMessage is not None:
                msg = p.lastMessage
                pos = p.position

                msgEnd = [pos[0] + msg[0] * 2.0, pos[1] + msg[1] * 2.0, 0.5]
                lineID = pybullet.addUserDebugLine(
                    lineFromXYZ = [pos[0], pos[1], 0.5],
                    lineToXYZ   = msgEnd,
                    lineColorRGB = [1, 0, 0],
                    lineWidth    = 3,
                    lifeTime     = 0.1
                )
                self._messageLineIDs.append(lineID)

        for p in self.prey:
            if hasattr(p, 'lastMessage') and p.lastMessage is not None:
                msg = p.lastMessage
                pos = p.position

                msgEnd = [pos[0] + msg[0] * 2.0, pos[1] + msg[1] * 2.0, 0.5]
                lineID = pybullet.addUserDebugLine(
                    lineFromXYZ=[pos[0], pos[1], 0.5],
                    lineToXYZ=msgEnd,
                    lineColorRGB=[0, 1, 0],
                    lineWidth=3,
                    lifeTime=0.1
                )
                self._messageLineIDs.append(lineID)

    def _updatePreyEating(self, currentTime) -> None:
        for prey in self.prey:
            if not prey.isAlive:
                continue

            preyPos = [prey.position[0].item(), prey.position[1].item()]

            for food in self.foodSources:
                if not food["available"]:
                    continue

                foodPosition = food["position"]
                dist = numpy.linalg.norm([
                    preyPos[0] - foodPosition[0],
                    preyPos[1] - foodPosition[1]
                ])

                if dist < self.eatDistance:
                    prey.eat(self.arenaFoodEnergyValue)
                    food["available"] = False
                    food["respawnTime"] = currentTime + 10

                    pybullet.changeVisualShape(
                        food["body"],
                        -1,
                        rgbaColor = [0.0, 0.0, 0.0, 0.2]
                    )
                break

    def _respawnFood(self, currentTime) -> None:
        for food in self.foodSources:
            if not food["available"] and currentTime >= food["respawnTime"]:
                food["available"] = True
                pybullet.changeVisualShape(
                    food["body"],
                    -1,
                    rgbaColor = [1.0, 1.0, 1.0, 1.0]
                )