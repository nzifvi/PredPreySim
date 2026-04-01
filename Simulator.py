import pybullet
import pybullet_data
import numpy
import time
import torch
import Agent
import SimulationConfig
import sys
import os

class SupressOutput:
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
    def __init__(self, simConfig = SimulationConfig.SimulationConfig, numPredators = 4, numPrey = 8, simDuration = 30.0, gui = True):
        self.numPredators = numPredators
        self.numPrey      = numPrey
        self.simDuration  = simDuration
        self.haveGUI = gui
        self.device = "cpu"

        self.arenaSize     = simConfig.arenaSize
        self.catchDistance = simConfig.catchDistance

        self.timeStep      = 1.0 / 240.0

        if self.haveGUI:
            self.simClient = pybullet.connect(pybullet.GUI)
        else:
            self.simClient = pybullet.connect(pybullet.DIRECT)

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.setGravity(0, 0, -9.81)
        pybullet.setTimeStep(self.timeStep)

        self._initSimulator()

        self.predators = []
        self.prey      = []

    def _initSimulator(self):
        self.plane = pybullet.loadURDF("plane.urdf")

        arenaDelimiter = pybullet.createVisualShape(
            shapeType   = pybullet.GEOM_BOX,
            halfExtents = [0.2, 0.2, 1.0],
            rgbaColor   = [0.5, 0.5, 0.5, 0.5]
        )
        arenaCorners = [
            [self.arenaSize / 2, self.arenaSize / 2, 1],
            [self.arenaSize / 2, -self.arenaSize / 2, 1],
            [-self.arenaSize / 2, self.arenaSize / 2, 1],
            [-self.arenaSize / 2, -self.arenaSize / 2, 1],
        ]
        for arenaCorner in arenaCorners:
            pybullet.createMultiBody(
                baseMass = 0,
                baseVisualShapeIndex = arenaDelimiter,
                basePosition = arenaCorner,
            )

    def reset(self):
        for agent in self.predators + self.prey:
            pybullet.removeBody(agent.agent)

        self.predators = []
        self.prey      = []

        for _ in range(self.numPredators):
            startingX = numpy.random.uniform(-self.arenaSize/3, self.arenaSize/3)
            startingY = numpy.random.uniform(-self.arenaSize/3, self.arenaSize/3)

            predator = Agent.Agent(
                position = [startingX, startingY],
                isPredator = True,
                agentConfig = SimulationConfig.PredatorConfig
            )
            self.predators.append(predator)

        for _ in range(self.numPrey):
            startingX = numpy.random.uniform(-self.arenaSize/3, self.arenaSize/3)
            startingY = numpy.random.uniform(-self.arenaSize/3, self.arenaSize/3)

            prey = Agent.Agent(
                position = [startingX, startingY],
                isPredator = False,
                agentConfig = SimulationConfig.PreyConfig
            )

            self.prey.append(prey)

        for _ in range(10):
            pybullet.stepSimulation()

    def runSimulation(self, predatorNNs, preyNNs):
        self.reset()

        preyCaught = [0] * self.numPredators

        currentTime = 0.0
        step = 0

        while currentTime < self.simDuration:

            for agent in self.predators + self.prey:
                agent.updateState()

            for i, predator in enumerate(self.predators):
                predatorObservation = predator.getObservation(self.predators, self.prey).view(1, 14)

                with torch.no_grad():
                    action = predatorNNs[i].inference(predatorObservation).flatten()

                predator.applyAction(action)
                predator.stepTime(self.timeStep)

            for i, prey in enumerate(self.prey):
                if not prey.isAlive:
                    continue
                else:
                    preyObservation = prey.getObservation(self.predators, self.prey).view(1, 14)

                    with torch.no_grad():
                        action = preyNNs[i].inference(preyObservation).flatten()

                    prey.applyAction(action)
                    prey.stepTime(self.timeStep)

            pybullet.stepSimulation()

            for i, predator in enumerate(self.predators):
                for prey in self.prey:
                    if not prey.isAlive:
                        continue

                    dist = torch.norm(predator.position - prey.position)

                    if dist < self.catchDistance:
                        prey.kill()
                        preyCaught[i] += 1

            if self.haveGUI and step % 10 == 0:
                time.sleep(self.timeStep * 10)

            currentTime += self.timeStep
            step += 1

        return {
            "predatorCatches" : preyCaught,
            "predatorTimeActive" : [predator.timeAlive for predator in self.predators],
            "preyTimeAlive" : [prey.timeAlive for prey in self.prey],
            "preyAlive" : [prey.isAlive for prey in self.prey],
        }

    def disconnect(self):
        pybullet.disconnect()