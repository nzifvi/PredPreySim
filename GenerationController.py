import os
from concurrent.futures import ProcessPoolExecutor
import torch
torch.set_num_threads(1)
import csv

import Evolver
import FitnessFunctions
import NeuralNetwork
import Simulator


RETRIAL_AMOUNT      = 1
WORKER_COUNT        = 8

SIMULATION_BATCH_SIZE   = 4
SIMULATION_REPEAT_COUNT = 2

ENABLE_MESSAGE_LOGGING = True


def evaluate(args):
    predators, prey, duration, enableMessageLogging = args

    predatorFitnessSums = {genotypeID: 0.0 for genotypeID, _ in predators}
    preyFitnessSums     = {genotypeID : 0.0 for genotypeID, _ in prey}

    predatorNNs = [item[1] for item in predators]
    preyNNs     = [item[1] for item in prey]

    predatorIDs = [item[0] for item in predators]
    preyIDs =     [item[0] for item in prey]

    capturedMessageLog = None

    predatorDiagnostics = {
        "catches": {genotypeID: 0.0 for genotypeID, _ in predators},
        "catchReward": {genotypeID: 0.0 for genotypeID, _ in predators},
        "teamHuntBonus": {genotypeID: 0.0 for genotypeID, _ in predators},
        "preyPressureBonus": {genotypeID: 0.0 for genotypeID, _ in predators},
        "totalFitness": {genotypeID: 0.0 for genotypeID, _ in predators}
    }

    with Simulator.SuppressOutput():
        sim = Simulator.Simulator(
            numPredators = len(predatorIDs),
            numPrey      = len(preyNNs),
            simDuration  = duration,
            gui = False
        )
        try:
            for _ in range(RETRIAL_AMOUNT):
                telemetry = sim.runSimulation(
                    predatorNNs         = predatorNNs,
                    preyNNs             = preyNNs,
                    predatorGenotypeIDs = predatorIDs,
                    preyGenotypeIDs     = preyIDs
                )
                for i, (genotypeID, _) in enumerate(predators):
                    predatorTelemetry = telemetry["predators"][i]
                    fitnessData = FitnessFunctions.calculatePredatorFitnessBreakdown(predatorTelemetry)

                    predatorFitnessSums[genotypeID] += fitnessData["totalFitness"]

                    predatorDiagnostics["catches"][genotypeID]     += fitnessData["catches"]
                    predatorDiagnostics["catchReward"][genotypeID] += fitnessData["catchReward"]
                    predatorDiagnostics["teamHuntBonus"][genotypeID] += fitnessData["teamHuntBonus"]
                    predatorDiagnostics["preyPressureBonus"][genotypeID] += fitnessData["preyPressureBonus"]
                    predatorDiagnostics["totalFitness"][genotypeID] += fitnessData["totalFitness"]

                for i, (genotypeID, _) in enumerate(prey):
                    preyTelemetry = telemetry["prey"][i]
                    fitness = FitnessFunctions.calculatePreyFitness(preyTelemetry)
                    preyFitnessSums[genotypeID] += fitness

                if enableMessageLogging and capturedMessageLog is None:
                    capturedMessageLog = telemetry.get("messageLog", None)
        finally:
            sim.disconnect()

    finalPredatorResults = {
        genotypeID : fitness / RETRIAL_AMOUNT for genotypeID, fitness in predatorFitnessSums.items()
    }
    finalPreyResults = {
        genotypeID : fitness / RETRIAL_AMOUNT for genotypeID, fitness in preyFitnessSums.items()
    }
    finalPredatorDiagnostics = {
        metric : {genotypeID : value / RETRIAL_AMOUNT for genotypeID, value in metricMap.items()} for metric, metricMap in predatorDiagnostics.items()
    }

    return finalPredatorResults, finalPreyResults, capturedMessageLog, finalPredatorDiagnostics

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
            tournamentSize=4,
            mutationRate=0.10,
            sigma=0.08
        )
        self.preyEvolver = Evolver.Evolver(
            tournamentSize=3,
            mutationRate=0.15,
            sigma=0.1
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

    def _runSimulator(self, duration):
        predators = [
            (p["genotypeID"], p["genotypeNN"]) for p in self.currentPredatorGeneration
        ]
        prey = [
            (p["genotypeID"], p["genotypeNN"]) for p in self.currentPreyGeneration
        ]

        simulationBatches = self._prepareSimulationBatches(predators, prey)

        tasks = []
        for i in range(0, SIMULATION_REPEAT_COUNT):
            for j, simulationBatch in enumerate(simulationBatches):
                enableSimulationLog = ENABLE_MESSAGE_LOGGING and i == 0 and j == 0
                tasks.append(
                    (
                        simulationBatch["predators"],
                        simulationBatch["prey"],
                        duration,
                        enableSimulationLog,
                    )
                )

        with ProcessPoolExecutor(max_workers = min(WORKER_COUNT, len(tasks))) as executor:
            results = list(
                executor.map(evaluate, tasks)
            )

        for predator in self.currentPredatorGeneration:
            predator["fitness"] = 0.0
        for prey in self.currentPreyGeneration:
            prey["fitness"] = 0.0

        predatorEvaluationCounts = {
            predator["genotypeID"] : 0 for predator in self.currentPredatorGeneration
        }
        preyEvaluationCounts = {
            prey["genotypeID"] : 0 for prey in self.currentPreyGeneration
        }

        savedMessageLog = None

        aggregatedPredatorDiagnostics = {
            "catches"           : 0.0,
            "catchReward"       : 0.0,
            "teamHuntBonus"     : 0.0,
            "preyPressureBonus" : 0.0,
            "totalFitness"      : 0.0
        }

        diagnosticBatchCount = 0

        for predMap, preyMap, messageLog, predatorDiagnostics in results:
            for genotypeID, fitness in predMap.items():
                self.currentPredatorGeneration[genotypeID]["fitness"] += fitness
                predatorEvaluationCounts[genotypeID] += 1

            for genotypeID, fitness in preyMap.items():
                self.currentPreyGeneration[genotypeID]["fitness"] += fitness
                preyEvaluationCounts[genotypeID] += 1

            for metric in aggregatedPredatorDiagnostics:
                metricValues = list(predatorDiagnostics[metric].values())
                if metricValues:
                    aggregatedPredatorDiagnostics[metric] += sum(metricValues) / len(metricValues)

            diagnosticBatchCount += 1

            if savedMessageLog is None and messageLog:
                savedMessageLog = messageLog

        for predator in self.currentPredatorGeneration:
            genotypeID = predator["genotypeID"]
            evalCount = predatorEvaluationCounts[genotypeID]
            if evalCount == 0:
                raise ValueError(f"Error: predator genotype {genotypeID} has no evaluation count. Suggests genotype was not evaluated")
            predator["fitness"] /= evalCount

        for prey in self.currentPreyGeneration:
            genotypeID = prey["genotypeID"]
            evalCount = preyEvaluationCounts[genotypeID]
            if evalCount == 0:
                raise ValueError(f"Error: prey genotype {genotypeID} has no evaluation count. Suggests genotype was not evaluated")
            prey["fitness"] /= evalCount

        if diagnosticBatchCount > 0:
            for metric in aggregatedPredatorDiagnostics:
                aggregatedPredatorDiagnostics[metric] /= diagnosticBatchCount

        return savedMessageLog, aggregatedPredatorDiagnostics



    def run(self, duration):
        messageLog, predatorDiagnostics = self._runSimulator(duration)

        predData = calculateDescriptiveStatisticsFromGeneration(self.currentPredatorGeneration)
        preyData = calculateDescriptiveStatisticsFromGeneration(self.currentPreyGeneration)

        print(
            f"Predator Diagnostics | "
            f"avg catches {predatorDiagnostics['catches']:.2f} |"
            f"catchReward {predatorDiagnostics['catchReward']:.2f} |"
            f"teamHuntBonus {predatorDiagnostics['teamHuntBonus']:.2f} |"
            f"preyPressureBonus {predatorDiagnostics['preyPressureBonus']:.2f} |"
        )

        if self.generationNo % self.checkpointControl == 0:
            self._saveGeneration()
            self._writeRecentCheckpoint(self.generationNo)
            if ENABLE_MESSAGE_LOGGING and messageLog:
                saveMessageLog(messageLog, self.generationNo)

        self.currentPredatorGeneration = self.predEvolver.produceNextGeneration(
            self.currentPredatorGeneration,
            kFittest = 4
        )
        self.currentPreyGeneration = self.preyEvolver.produceNextGeneration(
            self.currentPreyGeneration
        )

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

    def _prepareSimulationBatches(self, predators, prey) -> list:
        i = 0
        simulationBatches = []
        while i < self.predPopSize:
            simulationBatches.append(
                {
                    "predators" : predators[i:i + SIMULATION_BATCH_SIZE],
                    "prey"      : prey
                }
            )
            i = i + SIMULATION_BATCH_SIZE
        return simulationBatches



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