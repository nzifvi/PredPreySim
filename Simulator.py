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
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


class Simulator:
    def __init__(self, simConfig=SimulationConfig.SimulationConfig, numPredators=4, numPrey=8, simDuration=30.0, gui=True):
        self.numPredators = numPredators
        self.numPrey = numPrey
        self.simDuration = simDuration
        self.haveGUI = gui

        self.arenaSize           = simConfig.arenaSize
        self.catchDistance       = simConfig.catchDistance
        self.visionRadius        = simConfig.visionRadius
        self.communicationRadius = simConfig.communicationRadius

        self.timeStep = 1.0 / 240.0

        if self.haveGUI:
            self.simClient = pybullet.connect(pybullet.GUI)
        else:
            self.simClient = pybullet.connect(pybullet.DIRECT)

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.setGravity(0, 0, -9.81)
        pybullet.setTimeStep(self.timeStep)

        self._initSimulator()

        self.predators = []
        self.prey = []

        self.messageLog = []

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

        self.predators = []
        self.prey = []
        self.messageLog = []

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

        for _ in range(10):
            pybullet.stepSimulation()

    def runSimulation(self, predatorNNs, preyNNs, predatorGenotypeIDs = None, preyGenotypeIDs = None):
        self.reset()

        self._initialiseInternalAgentState(predatorNNs, preyNNs)

        preyCaught = [0] * self.numPredators
        currentTime = 0.0
        step = 0

        while currentTime < self.simDuration:
            self._updateAllStates()
            self._updateReceivedMessages()

            predatorActions = self._getPredatorActions(predatorNNs, predatorGenotypeIDs, step)
            preyActions = self._getPreyActions(preyNNs, preyGenotypeIDs, step)

            self._applyPredatorActions(predatorActions)
            self._applyPreyActions(preyActions)

            pybullet.stepSimulation()

            self._updateAllStates()
            self._processCatches(preyCaught)

            if self.haveGUI and step % 5 == 0:
                time.sleep(self.timeStep * 2)

            currentTime += self.timeStep
            step += 1

        return {
            "predators": [
                {
                    "catches": preyCaught[i],
                    "timeActive": self.predators[i].timeAlive
                }
                for i in range(len(self.predators))
            ],
            "prey": [
                {
                    "timeAlive": self.prey[i].timeAlive,
                    "alive": self.prey[i].isAlive
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

    def _updateAllStates(self):
        for agent in self.predators + self.prey:
            agent.updateState()

    def _averageAllyMessages(self, agents, currentAgent):
        messages = []

        for agent in agents:
            if agent is currentAgent:
                continue
            if not agent.isAlive:
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

    def _getPredatorActions(self, predatorNNs, predatorGenotypeIDs, step) -> list:
        predatorActions = []

        for i, predator in enumerate(self.predators):
            predatorObservation = predator.getObservation(self.predators, self.prey, self.visionRadius).view(1, 14)

            with torch.no_grad():
                movement, communication, newHidden = predatorNNs[i].forward(
                    predatorObservation,
                    predator.hiddenState,
                    predator.receivedMessage
                )

            predator.hiddenState = newHidden
            predator.message = communication

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

    def _getPreyActions(self, preyNNs, preyGenotypeIDs, step) -> list:
        preyActions = []

        for i, prey in enumerate(self.prey):
            if not prey.isAlive:
                prey.message = torch.zeros(1, 2, dtype=torch.float32)
                prey.receivedMessage = torch.zeros(1, 2, dtype=torch.float32)
                preyActions.append(None)
                continue

            preyObservation = prey.getObservation(self.predators, self.prey, self.visionRadius).view(1, 14)

            with torch.no_grad():
                movement, communication, newHidden = preyNNs[i].forward(
                    preyObservation,
                    prey.hiddenState,
                    prey.receivedMessage
                )

            prey.hiddenState = newHidden
            prey.message = communication

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
            predator.applyAction(action)
            predator.stepTime(self.timeStep)

    def _applyPreyActions(self, preyActions) -> None:
        for prey, action in zip(self.prey, preyActions):
            if not prey.isAlive or action is None:
                continue

            prey.applyAction(action)
            prey.stepTime(self.timeStep)

    def _processCatches(self, preyCaught) -> None:
        for i, predator in enumerate(self.predators):
            for prey in self.prey:
                if not prey.isAlive:
                    continue

                dist = torch.norm(predator.position - prey.position)

                if dist < self.catchDistance:
                    prey.kill()
                    prey.message = torch.zeros(1, 2, dtype=torch.float32)
                    prey.receivedMessage = torch.zeros(1, 2, dtype=torch.float32)
                    preyCaught[i] += 1

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