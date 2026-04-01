import multiprocessing

import Evolver
import Simulator

import os
import torch
torch.set_num_threads(1)
import NeuralNetwork
import Simulator
import numpy

from concurrent.futures import ProcessPoolExecutor


RETRIAL_AMOUNT = 2
WORKER_COUNT   = 4

def evaluate(args):
    predators, prey, duration = args

    sumPredKills = {genotypeID:0 for genotypeID, _ in predators}
    sumPreyTime  = {genotypeID:0.0 for genotypeID, _ in prey}

    predNNs = [item[1] for item in predators]
    preyNNs = [item[1] for item in prey]

    with Simulator.SupressOutput():
        sim = Simulator.Simulator(
            numPredators = len(predNNs),
            numPrey      = len(preyNNs),
            simDuration  = duration,
            gui          = False
        )

        for _ in range(RETRIAL_AMOUNT):
            telemetry = sim.runSimulation(predNNs, preyNNs)

            for i, (genotypeID, nn) in enumerate(predators):
                sumPredKills[genotypeID] += telemetry["predatorCatches"][i]

            for i, (genotypeID, nn) in enumerate(prey):
                sumPreyTime[genotypeID] += telemetry["preyTimeAlive"][i]

        sim.disconnect()

    finalPredResults = {genotypeID: k / RETRIAL_AMOUNT for genotypeID, k in sumPredKills.items()}
    finalPreyResults = {genotypeID: k / RETRIAL_AMOUNT for genotypeID, k in sumPreyTime.items()}
    return finalPredResults, finalPreyResults

class GenerationController:
    def __init__(self, predPopSize:int, preyPopSize:int, checkpointControl:int):
        self.generationNo = self._readRecentCheckpoint()

        self.predPopSize = predPopSize
        self.preyPopSize = preyPopSize

        self.currentPredatorGeneration = []
        self.nextPredatorGeneration    = []

        self.currentPreyGeneration = []
        self.nextPreyGeneration    = []

        # each member of a generation represented as...
        # {
        #    "generationNo": parent1.get("generationNo", 0) + 1,
        #    "genotypeID": None,
        #    "genotypeNN": None,
        #    "genotype": child,
        #    "fitness": None
        # }

        self.checkpointControl = checkpointControl

        self.layer1 = {
            "inputs"       : 14,
            "outputs"      : 16,
            "totalWeights" : 224,
            "biases"       : 16
        }
        self.layer2 = {
            "inputs"       : 16,
            "outputs"      : 2,
            "totalWeights" : 32,
            "biases"       : 2
         }

        self.predEvolver = Evolver.Evolver(
            tournamentSize = 2,
            mutationRate = 0.35,
            sigma = 0.3
        )
        self.preyEvolver = Evolver.Evolver(
            tournamentSize = 2,
            mutationRate = 0.35,
            sigma = 0.3
        )

        if self.generationNo == 0: # create progenitor generation
            self._initProgenitorGeneration()
        else: # load descendant generation
            self._initDescendantGeneration()

    def _initProgenitorGeneration(self):
        self._createGenerationDirectory()

        for i in range(0, self.predPopSize):
            weights, biases = self._createProgenitorNeuralNetwork()
            nn = NeuralNetwork.NeuralNetwork(weights = weights, biases = biases, isPredator = True)
            self.currentPredatorGeneration.append(
                {
                    "generationNo" : 0,
                    "genotypeID"   : i,
                    "genotypeNN"   : nn,
                    "genotype"     : self._flattenNeuralNetwork(nn),
                    "fitness"      : None
                }
            )

        for i in range(0, self.preyPopSize):
            weights, biases = self._createProgenitorNeuralNetwork()
            nn = NeuralNetwork.NeuralNetwork(weights = weights, biases = biases, isPredator = False)
            self.currentPreyGeneration.append(
                {
                    "generationNo" : 0,
                    "genotypeID"   : i,
                    "genotypeNN"   : nn,
                    "genotype"     : self._flattenNeuralNetwork(nn),
                    "fitness"      : None
                }
            )

    def _initDescendantGeneration(self):
        self.predPopSize, self.preyPopSize = self._readPopulationSize()

        self.currentPredatorGeneration, self.currentPreyGeneration = self._loadGeneration()

    def _readRecentCheckpoint(self) -> int:
        try:
            with open("Generations/GenerationCount.txt", "r") as f:
                return int(f.readline())
        except Exception as e:
            raise ValueError("Cannot load GenerationCount.txt") from e

    def _writeRecentCheckpoint(self, newCheckpoint) -> None:
        try:
            with open(r"Generations\GenerationCount.txt", "w") as f:
                f.write(str(newCheckpoint))
        except Exception as e:
            raise ValueError("Cannot write GenerationCount.txt") from e

    def _readPopulationSize(self) -> tuple:
        currentGenDirectoryPath = f"Generations/Generation{self.generationNo}"
        try:
            predPopulationSize = 0
            preyPopulationSize = 0
            file = open(currentGenDirectoryPath + "/PredatorCount.txt", "r")
            predPopulationSize = int(file.read())
            file.close()
            file = open(currentGenDirectoryPath + "/PreyCount.txt", "r")
            preyPopulationSize = int(file.read())
            file.close()

            return predPopulationSize, preyPopulationSize
        except Exception as e:
            raise ValueError(f"Cannot read population counts from {currentGenDirectoryPath}") from e

    def _createProgenitorNeuralNetwork(self) -> tuple:
        totalParams = self.layer1["totalWeights"] + self.layer2["totalWeights"] + self.layer1["biases"] + self.layer2["biases"]
        genotype = torch.randn(totalParams)
        weights, biases = self._unflattenNeuralNetwork(genotype)
        return weights, biases

    def _saveGeneration(self) -> None:
        self._createGenerationDirectory()
        predWeights = []
        predBiases  = []
        preyWeights = []
        preyBiases  = []

        for indiv in self.currentPredatorGeneration:
            weights, biases = self._unflattenNeuralNetwork(indiv["genotype"])
            predWeights.append(weights)
            predBiases.append(biases)
        for indiv in self.currentPreyGeneration:
            weights, biases = self._unflattenNeuralNetwork(indiv["genotype"])
            preyWeights.append(weights)
            preyBiases.append(biases)

        self._saveNeuralNetworks(predWeights, predBiases, preyWeights, preyBiases)

    def _saveNeuralNetworks(self, predWeights, predBiases, preyWeights, preyBiases) -> None:
        generationPredDirectoryPath = "Generations/Generation" + str(self.generationNo) + "/Predators"
        generationPreyDirectoryPath = "Generations/Generation" + str(self.generationNo) + "/Prey"
        try:
            for i in range(self.predPopSize):
                weightsPath = os.path.join(generationPredDirectoryPath, f"NeuralNetworks/Genotype{i}/Weights/weights.pt")
                biasesPath = os.path.join(generationPredDirectoryPath, f"NeuralNetworks/Genotype{i}/Biases/biases.pt")

                torch.save(predWeights[i], weightsPath)
                torch.save(predBiases[i], biasesPath)

            for i in range(self.preyPopSize):
                weightsPath = os.path.join(
                    generationPreyDirectoryPath,
                    f"NeuralNetworks/Genotype{i}/Weights/weights.pt"
                )
                biasesPath = os.path.join(
                    generationPreyDirectoryPath,
                    f"NeuralNetworks/Genotype{i}/Biases/biases.pt"
                )
                torch.save(preyWeights[i], weightsPath)
                torch.save(preyBiases[i], biasesPath)
        except Exception as e:
            raise ValueError(f"! Error saving binary data at generation {self.generationNo}: {e}")

    def _loadGeneration(self) -> list:
        predPopulation = []
        preyPopulation = []

        for genotypeID in range(0, self.predPopSize):
            nn = NeuralNetwork.NeuralNetwork(self.generationNo, genotypeID, isPredator = True)
            predPopulation.append(
                {
                    "generationNo" : self.generationNo,
                    "genotypeID"   : genotypeID,
                    "genotypeNN"   : nn,
                    "genotype"     : self._flattenNeuralNetwork(nn),
                    "fitness"      : None
                }
            )
        for genotypeID in range(0, self.preyPopSize):
            nn = NeuralNetwork.NeuralNetwork(self.generationNo, genotypeID, isPredator = False)
            preyPopulation.append(
                {
                    "generationNo" : self.generationNo,
                    "genotypeID" : genotypeID,
                    "genotypeNN" : nn,
                    "genotype" : self._flattenNeuralNetwork(nn),
                    "fitness" : None
                }
            )
        return predPopulation, preyPopulation

    def _flattenNeuralNetwork(self, nn:NeuralNetwork.NeuralNetwork) -> torch.Tensor:
        parameters = []
        for i in range(len(nn.weights)):
            parameters.append(nn.weights[i].flatten())
            parameters.append(nn.biases[i].flatten())
        return torch.cat(parameters)

    def _unflattenNeuralNetwork(self, genotype:torch.Tensor) -> tuple:
        unflattenedWeights = []
        unflattenedBiases = []
        j = 0

        for layer in [self.layer1, self.layer2]:
            inDim, outDim = layer["inputs"], layer["outputs"]
            weightsArea = layer["totalWeights"]
            biasArea = layer["biases"]

            unflattenedWeights.append(
                genotype[j : j + weightsArea].reshape(inDim, outDim)
            )
            j = j + weightsArea
            unflattenedBiases.append(
                genotype[j : j + biasArea].reshape(1, outDim)
            )
            j = j + biasArea

        return unflattenedWeights, unflattenedBiases

    def _createGenerationDirectory(self) -> None:
        try:
            newGenerationDirectoryPath = "Generations/Generation" + str(self.generationNo)
            os.makedirs(newGenerationDirectoryPath, exist_ok = True)
            os.makedirs(newGenerationDirectoryPath + "/Predators", exist_ok = True)
            os.makedirs(newGenerationDirectoryPath + "/Prey", exist_ok = True)
            os.makedirs(newGenerationDirectoryPath + "/Predators/NeuralNetworks", exist_ok = True)
            os.makedirs(newGenerationDirectoryPath + "/Prey/NeuralNetworks", exist_ok = True)

            for i in range(self.predPopSize):
                os.makedirs(newGenerationDirectoryPath + "/Predators/NeuralNetworks/" + "Genotype" + str(i), exist_ok = True)
                os.makedirs(newGenerationDirectoryPath + "/Predators/NeuralNetworks/" + "Genotype" + str(i) + "/Weights", exist_ok = True)
                os.makedirs(newGenerationDirectoryPath + "/Predators/NeuralNetworks/" + "Genotype" + str(i) + "/Biases", exist_ok = True)

            for i in range(self.preyPopSize):
                os.makedirs(newGenerationDirectoryPath + "/Prey/NeuralNetworks/" + "Genotype" + str(i), exist_ok = True)
                os.makedirs(newGenerationDirectoryPath + "/Prey/NeuralNetworks/" + "Genotype" + str(i) + "/Weights", exist_ok = True)
                os.makedirs(newGenerationDirectoryPath + "/Prey/NeuralNetworks/" + "Genotype" + str(i) + "/Biases", exist_ok = True)

            file = open(newGenerationDirectoryPath + "/PredatorCount.txt", "w")
            file.write(str(self.predPopSize))
            file.close()

            file = open(newGenerationDirectoryPath + "/PreyCount.txt", "w")
            file.write(str(self.preyPopSize))
            file.close()
        except Exception as e:
            print(e)

    def _runSimulator(self):
        predators = [
            (p["genotypeID"], p["genotypeNN"]) for p in self.currentPredatorGeneration
        ]
        prey = [
            (p["genotypeID"], p["genotypeNN"]) for p in self.currentPreyGeneration
        ]

        tasks = [
            (predators, prey, 30.0) for _ in range(WORKER_COUNT)
        ]

        with ProcessPoolExecutor(max_workers=WORKER_COUNT) as executor:
            results = list(
                executor.map(evaluate, tasks)
            )

        for p in self.currentPredatorGeneration:
            p["fitness"] = 0.0
        for p in self.currentPreyGeneration:
            p["fitness"] = 0.0

        for predMap, preyMap in results:
            for genotypeID, fitness in predMap.items():
                self.currentPredatorGeneration[genotypeID]["fitness"] += fitness / WORKER_COUNT
            for genotypeID, fitness in preyMap.items():
                self.currentPreyGeneration[genotypeID]["fitness"] += fitness / WORKER_COUNT

    def run(self):
        self._runSimulator()

        predData = calculateDescriptiveStatisticsFromGeneration(self.currentPredatorGeneration)
        preyData = calculateDescriptiveStatisticsFromGeneration(self.currentPreyGeneration)

        if self.generationNo % self.checkpointControl == 0:
            self._saveGeneration()
            self._writeRecentCheckpoint(self.generationNo)

        self.currentPredatorGeneration = self.predEvolver.produceNextGeneration(self.currentPredatorGeneration)
        self.currentPreyGeneration = self.predEvolver.produceNextGeneration(self.currentPreyGeneration)

        for i, pop in enumerate(self.currentPredatorGeneration):
            self.currentPredatorGeneration[i]["genotypeID"] = i
            self.currentPredatorGeneration[i]["fitness"] = 0.0
            weights, biases = self._unflattenNeuralNetwork(
                self.currentPredatorGeneration[i]["genotype"]
            )
            self.currentPredatorGeneration[i]["genotypeNN"] = NeuralNetwork.NeuralNetwork(weights = weights, biases = biases, isPredator = True)

        for i, pop in enumerate(self.currentPreyGeneration):
            self.currentPreyGeneration[i]["genotypeID"] = i
            self.currentPreyGeneration[i]["fitness"] = 0.0
            weights, biases = self._unflattenNeuralNetwork(
                self.currentPreyGeneration[i]["genotype"]
            )
            self.currentPreyGeneration[i]["genotypeNN"] = NeuralNetwork.NeuralNetwork(weights = weights, biases = biases, isPredator = False)
        self.generationNo += 1
        return (predData, preyData)

def calculateDescriptiveStatisticsFromGeneration(generation:list) -> tuple:
    best = 0.0
    worst = 0.0
    sum = 0.0
    for indiv in generation:
        sum += indiv["fitness"]
        if indiv["fitness"] > best:
            best = indiv["fitness"]
        if indiv["fitness"] < worst:
            worst = indiv["fitness"]

    return (sum/len(generation), best, worst)