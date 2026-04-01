import os
from concurrent.futures import ProcessPoolExecutor
import torch
torch.set_num_threads(1)
import csv

import Evolver
import FitnessFunctions
import NeuralNetwork
import Simulator


RETRIAL_AMOUNT = 1
WORKER_COUNT = 8
ENABLE_MESSAGE_LOGGING = True


def evaluate(args):
    predators, prey, duration, enableMessageLogging = args

    sumPredatorFitness = {genotypeID: 0.0 for genotypeID, _ in predators}
    sumPreyFitness = {genotypeID: 0.0 for genotypeID, _ in prey}

    predNNs     = [item[1] for item in predators]
    preyNNs     = [item[1] for item in prey]
    predatorIDs = [item[0] for item in predators]
    preyIDs     = [item[0] for item in prey]

    capturedMessageLog = None

    with Simulator.SuppressOutput():
        sim = Simulator.Simulator(
            numPredators=len(predNNs),
            numPrey=len(preyNNs),
            simDuration=duration,
            gui=False
        )

        for _ in range(RETRIAL_AMOUNT):
            telemetry = sim.runSimulation(
                predatorNNs = predNNs,
                preyNNs = preyNNs,
                predatorGenotypeIDs = predatorIDs,
                preyGenotypeIDs = preyIDs,
            )

            for i, (genotypeID, _) in enumerate(predators):
                predatorTelemetry = telemetry["predators"][i]
                fitness = FitnessFunctions.calculatePredatorFitness(predatorTelemetry)
                sumPredatorFitness[genotypeID] += fitness

            for i, (genotypeID, _) in enumerate(prey):
                preyTelemetry = telemetry["prey"][i]
                fitness = FitnessFunctions.calculatePreyFitness(preyTelemetry)
                sumPreyFitness[genotypeID] += fitness

            if enableMessageLogging and capturedMessageLog is None:
                capturedMessageLog = telemetry.get("messageLog", None)

        sim.disconnect()

    finalPredatorResults = {
        genotypeID: fitness / RETRIAL_AMOUNT
        for genotypeID, fitness in sumPredatorFitness.items()
    }
    finalPreyResults = {
        genotypeID: fitness / RETRIAL_AMOUNT
        for genotypeID, fitness in sumPreyFitness.items()
    }

    return finalPredatorResults, finalPreyResults, capturedMessageLog

def saveMessageLog(messageLog, generationNo):
    if not messageLog:
        return

    logPath = os.path.join(
        "Generations", f"Generation{generationNo}", "messageLog.csv"
    )
    with open(logPath , "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=messageLog[0].keys())
        writer.writeheader()
        writer.writerows(messageLog)

class GenerationController:
    def __init__(self, predPopSize: int, preyPopSize: int, checkpointControl: int):
        self.generationNo = self._readRecentCheckpoint()

        self.predPopSize = predPopSize
        self.preyPopSize = preyPopSize
        self.checkpointControl = checkpointControl

        self.currentPredatorGeneration = []
        self.currentPreyGeneration = []

        self.nextPredatorGeneration = []
        self.nextPreyGeneration = []

        # Hardcoded RNN architecture layout
        self.layerSpecs = [
            {
                "name": "encoder",
                "inputs": 16,
                "outputs": 32,
                "totalWeights": 16 * 32,
                "biases": 32
            },
            {
                "name": "rnn_ih",
                "inputs": 32,
                "outputs": 32,
                "totalWeights": 32 * 32,
                "biases": 32
            },
            {
                "name": "rnn_hh",
                "inputs": 32,
                "outputs": 32,
                "totalWeights": 32 * 32,
                "biases": 32
            },
            {
                "name": "head",
                "inputs": 32,
                "outputs": 4,
                "totalWeights": 32 * 4,
                "biases": 4
            }
        ]

        self.totalParams = sum(layer["totalWeights"] + layer["biases"] for layer in self.layerSpecs)

        self.predEvolver = Evolver.Evolver(
            tournamentSize=2,
            mutationRate=0.35,
            sigma=0.3
        )
        self.preyEvolver = Evolver.Evolver(
            tournamentSize=2,
            mutationRate=0.35,
            sigma=0.3
        )

        if self.generationNo == 0:
            self._initProgenitorGeneration()
        else:
            self._initDescendantGeneration()

    def _initProgenitorGeneration(self):
        self._createGenerationDirectory()

        for i in range(self.predPopSize):
            weights, biases = self._createProgenitorNeuralNetwork()
            nn = NeuralNetwork.NeuralNetwork(weights=weights, biases=biases, isPredator=True)
            self.currentPredatorGeneration.append(
                {
                    "generationNo": 0,
                    "genotypeID": i,
                    "genotypeNN": nn,
                    "genotype": self._flattenNeuralNetwork(nn),
                    "fitness": None
                }
            )

        for i in range(self.preyPopSize):
            weights, biases = self._createProgenitorNeuralNetwork()
            nn = NeuralNetwork.NeuralNetwork(weights=weights, biases=biases, isPredator=False)
            self.currentPreyGeneration.append(
                {
                    "generationNo": 0,
                    "genotypeID": i,
                    "genotypeNN": nn,
                    "genotype": self._flattenNeuralNetwork(nn),
                    "fitness": None
                }
            )

    def _initDescendantGeneration(self):
        self.predPopSize, self.preyPopSize = self._readPopulationSize()
        self.currentPredatorGeneration, self.currentPreyGeneration = self._loadGeneration()

    def _readRecentCheckpoint(self) -> int:
        path = os.path.join("Generations", "GenerationCount.txt")
        if not os.path.exists(path):
            return 0

        try:
            with open(path, "r") as f:
                return int(f.readline().strip())
        except Exception as e:
            raise ValueError("Cannot load GenerationCount.txt") from e

    def _writeRecentCheckpoint(self, newCheckpoint) -> None:
        path = os.path.join("Generations", "GenerationCount.txt")
        try:
            with open(path, "w") as f:
                f.write(str(newCheckpoint))
        except Exception as e:
            raise ValueError("Cannot write GenerationCount.txt") from e

    def _readPopulationSize(self) -> tuple:
        currentGenDirectoryPath = os.path.join("Generations", f"Generation{self.generationNo}")
        try:
            with open(os.path.join(currentGenDirectoryPath, "PredatorCount.txt"), "r") as file:
                predPopulationSize = int(file.read())

            with open(os.path.join(currentGenDirectoryPath, "PreyCount.txt"), "r") as file:
                preyPopulationSize = int(file.read())

            return predPopulationSize, preyPopulationSize
        except Exception as e:
            raise ValueError(f"Cannot read population counts from {currentGenDirectoryPath}") from e

    def _createProgenitorNeuralNetwork(self) -> tuple:
        genotype = torch.randn(self.totalParams)
        weights, biases = self._unflattenNeuralNetwork(genotype)
        return weights, biases

    def _saveGeneration(self) -> None:
        self._createGenerationDirectory()

        predWeights = []
        predBiases = []
        preyWeights = []
        preyBiases = []

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
        generationPredDirectoryPath = os.path.join("Generations", f"Generation{self.generationNo}", "Predators")
        generationPreyDirectoryPath = os.path.join("Generations", f"Generation{self.generationNo}", "Prey")

        try:
            for i in range(self.predPopSize):
                weightsPath = os.path.join(
                    generationPredDirectoryPath,
                    f"NeuralNetworks/Genotype{i}/Weights/weights.pt"
                )
                biasesPath = os.path.join(
                    generationPredDirectoryPath,
                    f"NeuralNetworks/Genotype{i}/Biases/biases.pt"
                )

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
            raise ValueError(f"Error saving binary data at generation {self.generationNo}: {e}") from e

    def _loadGeneration(self) -> list:
        predPopulation = []
        preyPopulation = []

        for genotypeID in range(self.predPopSize):
            nn = NeuralNetwork.NeuralNetwork(self.generationNo, genotypeID, isPredator=True)
            predPopulation.append(
                {
                    "generationNo": self.generationNo,
                    "genotypeID": genotypeID,
                    "genotypeNN": nn,
                    "genotype": self._flattenNeuralNetwork(nn),
                    "fitness": None
                }
            )

        for genotypeID in range(self.preyPopSize):
            nn = NeuralNetwork.NeuralNetwork(self.generationNo, genotypeID, isPredator=False)
            preyPopulation.append(
                {
                    "generationNo": self.generationNo,
                    "genotypeID": genotypeID,
                    "genotypeNN": nn,
                    "genotype": self._flattenNeuralNetwork(nn),
                    "fitness": None
                }
            )

        return predPopulation, preyPopulation

    def _flattenNeuralNetwork(self, nn: NeuralNetwork.NeuralNetwork) -> torch.Tensor:
        parameters = []

        for weightBlock, biasBlock in zip(nn.weights, nn.biases):
            parameters.append(weightBlock.flatten())
            parameters.append(biasBlock.flatten())

        return torch.cat(parameters)

    def _unflattenNeuralNetwork(self, genotype: torch.Tensor) -> tuple:
        unflattenedWeights = []
        unflattenedBiases = []

        j = 0
        for layer in self.layerSpecs:
            inDim = layer["inputs"]
            outDim = layer["outputs"]
            weightsArea = layer["totalWeights"]
            biasArea = layer["biases"]

            weightTensor = genotype[j: j + weightsArea].reshape(inDim, outDim)
            j += weightsArea

            biasTensor = genotype[j: j + biasArea].reshape(1, outDim)
            j += biasArea

            unflattenedWeights.append(weightTensor)
            unflattenedBiases.append(biasTensor)

        return unflattenedWeights, unflattenedBiases

    def _createGenerationDirectory(self) -> None:
        try:
            newGenerationDirectoryPath = os.path.join("Generations", f"Generation{self.generationNo}")
            os.makedirs(newGenerationDirectoryPath, exist_ok=True)
            os.makedirs(os.path.join(newGenerationDirectoryPath, "Predators"), exist_ok=True)
            os.makedirs(os.path.join(newGenerationDirectoryPath, "Prey"), exist_ok=True)
            os.makedirs(os.path.join(newGenerationDirectoryPath, "Predators", "NeuralNetworks"), exist_ok=True)
            os.makedirs(os.path.join(newGenerationDirectoryPath, "Prey", "NeuralNetworks"), exist_ok=True)

            for i in range(self.predPopSize):
                os.makedirs(
                    os.path.join(newGenerationDirectoryPath, "Predators", "NeuralNetworks", f"Genotype{i}", "Weights"),
                    exist_ok=True
                )
                os.makedirs(
                    os.path.join(newGenerationDirectoryPath, "Predators", "NeuralNetworks", f"Genotype{i}", "Biases"),
                    exist_ok=True
                )

            for i in range(self.preyPopSize):
                os.makedirs(
                    os.path.join(newGenerationDirectoryPath, "Prey", "NeuralNetworks", f"Genotype{i}", "Weights"),
                    exist_ok=True
                )
                os.makedirs(
                    os.path.join(newGenerationDirectoryPath, "Prey", "NeuralNetworks", f"Genotype{i}", "Biases"),
                    exist_ok=True
                )

            with open(os.path.join(newGenerationDirectoryPath, "PredatorCount.txt"), "w") as file:
                file.write(str(self.predPopSize))

            with open(os.path.join(newGenerationDirectoryPath, "PreyCount.txt"), "w") as file:
                file.write(str(self.preyPopSize))

        except Exception as e:
            print(e)

    def _runSimulator(self):
        predators = [
            (p["genotypeID"], p["genotypeNN"]) for p in self.currentPredatorGeneration
        ]
        prey = [
            (p["genotypeID"], p["genotypeNN"]) for p in self.currentPreyGeneration
        ]

        tasks = []
        for workerIndex in range(WORKER_COUNT):
            enableLoggingForThisWorker = ENABLE_MESSAGE_LOGGING and workerIndex == 0
            tasks.append(
                (predators, prey, 30.0, enableLoggingForThisWorker)
            )

        with ProcessPoolExecutor(max_workers=WORKER_COUNT) as executor:
            results = list(executor.map(evaluate, tasks))

        for p in self.currentPredatorGeneration:
            p["fitness"] = 0.0
        for p in self.currentPreyGeneration:
            p["fitness"] = 0.0

        savedMessageLog = None

        for predMap, preyMap, messageLog in results:
            for genotypeID, fitness in predMap.items():
                self.currentPredatorGeneration[genotypeID]["fitness"] += fitness / WORKER_COUNT

            for genotypeID, fitness in preyMap.items():
                self.currentPreyGeneration[genotypeID]["fitness"] += fitness / WORKER_COUNT

            if savedMessageLog is None and messageLog:
                savedMessageLog = messageLog

        if ENABLE_MESSAGE_LOGGING and savedMessageLog:
            return savedMessageLog
        else:
            return None

    def run(self):
        messageLog = self._runSimulator()

        predData = calculateDescriptiveStatisticsFromGeneration(self.currentPredatorGeneration)
        preyData = calculateDescriptiveStatisticsFromGeneration(self.currentPreyGeneration)

        if self.generationNo % self.checkpointControl == 0:
            self._saveGeneration()
            self._writeRecentCheckpoint(self.generationNo)
            if ENABLE_MESSAGE_LOGGING and messageLog:
                saveMessageLog(messageLog, self.generationNo)

        self.currentPredatorGeneration = self.predEvolver.produceNextGeneration(self.currentPredatorGeneration)
        self.currentPreyGeneration = self.preyEvolver.produceNextGeneration(self.currentPreyGeneration)

        for i, _ in enumerate(self.currentPredatorGeneration):
            self.currentPredatorGeneration[i]["genotypeID"] = i
            self.currentPredatorGeneration[i]["fitness"] = 0.0

            weights, biases = self._unflattenNeuralNetwork(
                self.currentPredatorGeneration[i]["genotype"]
            )
            self.currentPredatorGeneration[i]["genotypeNN"] = NeuralNetwork.NeuralNetwork(
                weights=weights,
                biases=biases,
                isPredator=True
            )

        for i, _ in enumerate(self.currentPreyGeneration):
            self.currentPreyGeneration[i]["genotypeID"] = i
            self.currentPreyGeneration[i]["fitness"] = 0.0

            weights, biases = self._unflattenNeuralNetwork(
                self.currentPreyGeneration[i]["genotype"]
            )
            self.currentPreyGeneration[i]["genotypeNN"] = NeuralNetwork.NeuralNetwork(
                weights=weights,
                biases=biases,
                isPredator=False
            )

        self.generationNo += 1
        return predData, preyData


def calculateDescriptiveStatisticsFromGeneration(generation: list) -> tuple:
    if not generation:
        raise ValueError("Generation is empty")

    best = generation[0]["fitness"]
    worst = generation[0]["fitness"]
    fitnessSum = 0.0

    for indiv in generation:
        fitness = indiv["fitness"]
        fitnessSum += fitness

        if fitness > best:
            best = fitness
        if fitness < worst:
            worst = fitness

    return (fitnessSum / len(generation), best, worst)