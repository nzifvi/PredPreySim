import os
from concurrent.futures import ProcessPoolExecutor
import torch
torch.set_num_threads(1)
import csv

import Evolver
import FitnessFunctions
import NeuralNetwork
import Simulator

import tqdm

RETRIAL_AMOUNT      = 1
WORKER_COUNT        = 18

SIMULATION_BATCH_SIZE   = 4
SIMULATION_REPEAT_COUNT = 1

ENABLE_MESSAGE_LOGGING = True


def evaluate(args) -> tuple:
    predators, prey, duration, enableMessageLogging = args

    predatorNNs = [item[1] for item in predators]
    preyNNs = [item[1] for item in prey]

    predatorIDs = [item[0] for item in predators]
    preyIDs = [item[0] for item in prey]

    capturedMessageLog = None

    predatorTelemetrySums = {
        genotypeID : {
            "catches"           : 0.0,
            "teamHuntScore"     : 0.0,
            "meanCommMagnitude" : 0.0,
            "timeAlive"         : 0.0,
            "finalEnergy"       : 0.0,
        } for genotypeID, _ in predators
    }
    preyTelemetrySums = {
        genotypeID : {
            "foodEaten"         : 0.0,
            "timeAlive"         : 0.0,
            "groupingScore"     : 0.0,
            "meanCommMagnitude" : 0.0,
            "finalEnergy"       : 0.0,
        } for genotypeID, _ in prey
    }

    with Simulator.SuppressOutput():
        sim = Simulator.Simulator(
            numPredators = len(predatorIDs),
            numPrey      = len(preyIDs),
            simDuration  = duration,
            gui          = False
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
                    predTelemetry                                      = telemetry["predators"][i]
                    predatorTelemetrySums[genotypeID]["catches"]       += predTelemetry["catches"]
                    predatorTelemetrySums[genotypeID]["teamHuntScore"] += predTelemetry["teamHuntScore"]
                    predatorTelemetrySums[genotypeID]["meanCommMagnitude"] += predTelemetry["meanCommMagnitude"]
                    predatorTelemetrySums[genotypeID]["timeAlive"]         += predTelemetry["timeAlive"]
                    predatorTelemetrySums[genotypeID]["finalEnergy"]       += predTelemetry["finalEnergy"]


                for i, (genotypeID, _) in enumerate(prey):
                    preyTelemetry = telemetry["prey"][i]
                    preyTelemetrySums[genotypeID]["foodEaten"] += preyTelemetry["foodEaten"]
                    preyTelemetrySums[genotypeID]["timeAlive"] += preyTelemetry["timeAlive"]
                    preyTelemetrySums[genotypeID]["groupingScore"] += preyTelemetry["groupingScore"]
                    preyTelemetrySums[genotypeID]["meanCommMagnitude"] += preyTelemetry["meanCommMagnitude"]
                    preyTelemetrySums[genotypeID]["finalEnergy"] += preyTelemetry["finalEnergy"]

                if enableMessageLogging and capturedMessageLog is None:
                    capturedMessageLog = telemetry.get("messageLog", None)
        finally:
            sim.disconnect()

    finalPredatorTelemetry = {
        genotypeID : {
            metric : value / RETRIAL_AMOUNT for metric, value in metricMap.items()
        } for genotypeID, metricMap in predatorTelemetrySums.items()
    }
    finalPreyTelemetry = {
        genotypeID : {
            metric : value / RETRIAL_AMOUNT for metric, value in metricMap.items()
        } for genotypeID, metricMap in preyTelemetrySums.items()
    }

    return finalPredatorTelemetry, finalPreyTelemetry, capturedMessageLog

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
        self.nnBlueprint = NeuralNetwork.NeuralNetworkConfig.nnBlueprint

        self.totalParams = sum(layer["totalWeights"] + layer["biases"] for layer in self.nnBlueprint)

        self.predEvolver = Evolver.Evolver(
            tournamentSize = 2,
            mutationRate   = 0.12,
            sigma          = 0.1
        )
        self.preyEvolver = Evolver.Evolver(
            tournamentSize = 2,
            mutationRate   = 0.12,
            sigma          = 0.1
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
        self.currentPredatorGeneration, self.currentPreyGeneration = self._loadGeneration(self.generationNo)

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

    def _loadGeneration(self, genNo:int) -> tuple:
        predPopulation = []
        preyPopulation = []

        for genotypeID in range(self.predPopSize):
            nn = NeuralNetwork.NeuralNetwork(genNo, genotypeID, isPredator=True)
            predPopulation.append(
                {
                    "generationNo": genNo,
                    "genotypeID": genotypeID,
                    "genotypeNN": nn,
                    "genotype": self._flattenNeuralNetwork(nn),
                    "fitness": None
                }
            )

        for genotypeID in range(self.preyPopSize):
            nn = NeuralNetwork.NeuralNetwork(genNo, genotypeID, isPredator=False)
            preyPopulation.append(
                {
                    "generationNo": genNo,
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
        for layer in self.nnBlueprint:
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

    def _runSimulator(self, duration) -> tuple:
        predators = [
            (p["genotypeID"], p["genotypeNN"]) for p in self.currentPredatorGeneration
        ]
        prey = [
            (p["genotypeID"], p["genotypeNN"]) for p in self.currentPreyGeneration
        ]

        simBatches = self._prepareSimulationBatches(predators, prey)

        tasks = []
        for i in range(SIMULATION_REPEAT_COUNT):
            for j, simBatch in enumerate(simBatches):
                enableSimulationLog = ENABLE_MESSAGE_LOGGING and i == 0 and j == 0
                tasks.append(
                    (
                        simBatch["predators"],
                        simBatch["prey"],
                        duration,
                        enableSimulationLog,
                    )
                )
        totalTasks = len(tasks)
        with ProcessPoolExecutor(max_workers = max(WORKER_COUNT, len(tasks))) as executor:
            results = list(
                tqdm.tqdm(
                    executor.map(evaluate, tasks),
                    total = totalTasks,
                    desc = f"    Simulating {SIMULATION_REPEAT_COUNT * len(simBatches)} batches",
                    unit = "trial",
                    ncols = 100
                )
            )

        print("    - Simulations finished\n    - Processing telemetry data")
        idLinkedPredatorTelemetry = {
            predator["genotypeID"] : {
                "catches" : 0.0,
                "teamHuntScore" : 0.0,
                "meanCommMagnitude" : 0.0,
                "timeAlive" : 0.0,
                "finalEnergy" : 0.0,
            }
            for predator in self.currentPredatorGeneration
        }
        idLinkedPreyTelemetry = {
            prey["genotypeID"] : {
                "foodEaten" : 0.0,
                "timeAlive" : 0.0,
                "groupingScore" : 0.0,
                "meanCommMagnitude" : 0.0,
                "finalEnergy" : 0.0,
            }
            for prey in self.currentPreyGeneration
        }
        predEvalCounts = {
            predator["genotypeID"] : 0 for predator in self.currentPredatorGeneration
        }
        preyEvalCounts = {
            prey["genotypeID"] : 0 for prey in self.currentPreyGeneration
        }
        savedMessageLog = None

        for predTelemetries, preyTelemetries, messageLog in results:
            for gID, t in predTelemetries.items():
                idLinkedPredatorTelemetry[gID]["catches"] += t["catches"]
                idLinkedPredatorTelemetry[gID]["teamHuntScore"] += t["teamHuntScore"]
                idLinkedPredatorTelemetry[gID]["meanCommMagnitude"] += t.get("meanCommMagnitude", 0.0)  # NEW
                idLinkedPredatorTelemetry[gID]["timeAlive"] += t.get("timeAlive", 0.0)  # NEW
                idLinkedPredatorTelemetry[gID]["finalEnergy"] += t.get("finalEnergy", 0.0)  # NEW

                predEvalCounts[gID] += 1

            for gID, t in preyTelemetries.items():
                idLinkedPreyTelemetry[gID]["foodEaten"] += t["foodEaten"]
                idLinkedPreyTelemetry[gID]["timeAlive"] += t["timeAlive"]
                idLinkedPreyTelemetry[gID]["groupingScore"] += t["groupingScore"]
                idLinkedPreyTelemetry[gID]["meanCommMagnitude"] += t.get("meanCommMagnitude", 0.0)  # NEW
                idLinkedPreyTelemetry[gID]["finalEnergy"] += t.get("finalEnergy", 0.0)  # NEW

                preyEvalCounts[gID] += 1

            if savedMessageLog is None and messageLog:
                savedMessageLog = messageLog

        for gID, t in idLinkedPredatorTelemetry.items():
            evalCount = predEvalCounts[gID]
            if evalCount == 0:
                raise ValueError(f"! Predator gID {gID} has no eval count. Implies predator {gID} was not simulated")

            for key in t:
                t[key] /= evalCount

        for gID, t in idLinkedPreyTelemetry.items():
            evalCount = preyEvalCounts[gID]
            if evalCount == 0:
                print(f"! Prey gID {gID} has no eval count. Implies prey {gID} was not simulated")

            for key in t:
                t[key] /= evalCount

        for predator in self.currentPredatorGeneration:
            gID = predator["genotypeID"]

            predatorTelemetry = {
                "catches": idLinkedPredatorTelemetry[gID]["catches"],
                "teamHuntScore": idLinkedPredatorTelemetry[gID]["teamHuntScore"],
                "meanCommMagnitude": idLinkedPredatorTelemetry[gID]["meanCommMagnitude"],
                "timeAlive": idLinkedPredatorTelemetry[gID]["timeAlive"],
                "finalEnergy": idLinkedPredatorTelemetry[gID]["finalEnergy"],
            }

            fitnessBreakdown = FitnessFunctions.calculatePredatorFitnessBreakdown(predatorTelemetry)
            predator["fitness"] = fitnessBreakdown["totalFitness"]

        for prey in self.currentPreyGeneration:
            gID = prey["genotypeID"]

            preyTelemetry = {
                "foodEaten" : idLinkedPreyTelemetry[gID]["foodEaten"],
                "timeAlive": idLinkedPreyTelemetry[gID]["timeAlive"],
                "groupingScore": idLinkedPreyTelemetry[gID]["groupingScore"],
                "meanCommMagnitude": idLinkedPreyTelemetry[gID]["meanCommMagnitude"],
                "finalEnergy": idLinkedPreyTelemetry[gID]["finalEnergy"],
            }

            prey["fitness"] = FitnessFunctions.calculatePreyFitness(preyTelemetry)

        aggregatedPredatorDiagnostics = {
            "catches": 0.0,
            "catchReward": 0.0,
            "teamHuntBonus": 0.0,
            "commBonus": 0.0,
            "survivalBonus": 0.0,
            "energyBonus": 0.0,
            "starvationPenalty": 0.0,
            "totalFitness": 0.0,
        }

        if self.currentPredatorGeneration:
            for predator in self.currentPredatorGeneration:
                gID = predator["genotypeID"]

                predatorTelemetry = {
                    "catches": idLinkedPredatorTelemetry[gID]["catches"],
                    "teamHuntScore": idLinkedPredatorTelemetry[gID]["teamHuntScore"],
                    "meanCommMagnitude": idLinkedPredatorTelemetry[gID]["meanCommMagnitude"],
                    "timeAlive": idLinkedPredatorTelemetry[gID]["timeAlive"],
                    "finalEnergy": idLinkedPredatorTelemetry[gID]["finalEnergy"],
                }

                breakdown = FitnessFunctions.calculatePredatorFitnessBreakdown(predatorTelemetry)

                aggregatedPredatorDiagnostics["catches"] += breakdown["catches"]
                aggregatedPredatorDiagnostics["catchReward"] += breakdown["catchReward"]
                aggregatedPredatorDiagnostics["teamHuntBonus"] += breakdown["teamHuntBonus"]
                aggregatedPredatorDiagnostics["commBonus"] += breakdown["commBonus"]  # NEW
                aggregatedPredatorDiagnostics["survivalBonus"] += breakdown["survivalBonus"]  # NEW
                aggregatedPredatorDiagnostics["energyBonus"] += breakdown["energyBonus"]  # NEW
                aggregatedPredatorDiagnostics["starvationPenalty"] += breakdown["starvationPenalty"]  # NEW
                aggregatedPredatorDiagnostics["totalFitness"] += breakdown["totalFitness"]

            predatorCount = len(self.currentPredatorGeneration)
            for metric in aggregatedPredatorDiagnostics:
                aggregatedPredatorDiagnostics[metric] /= predatorCount

        print("    - Telemetry data processed: fitness scores calculated")
        return savedMessageLog, aggregatedPredatorDiagnostics, idLinkedPredatorTelemetry, idLinkedPreyTelemetry

    def run(self, duration) -> tuple:
        print(f"! Simulating generation {self.generationNo}")
        messageLog, predatorDiagnostics, predatorTelemetryByID, preyTelemetryByID = self._runSimulator(duration)

        predData = calculateDescriptiveStatisticsFromGeneration(self.currentPredatorGeneration)
        preyData = calculateDescriptiveStatisticsFromGeneration(self.currentPreyGeneration)

        print("\n" + "─" * 80)
        print("PREDATOR DIAGNOSTICS")
        print("─" * 80)
        print(f"  Avg Catches:          {predatorDiagnostics['catches']:>8.2f}")
        print(f"  Catch Reward:         {predatorDiagnostics['catchReward']:>8.2f}")
        print(f"  Team Hunt Bonus:      {predatorDiagnostics['teamHuntBonus']:>8.2f}")
        print(f"  Communication Bonus:  {predatorDiagnostics['commBonus']:>8.2f}")
        print(f"  Survival Bonus:       {predatorDiagnostics['survivalBonus']:>8.2f}")
        print(f"  Energy Bonus:         {predatorDiagnostics['energyBonus']:>8.2f}")
        print(f"  Starvation Penalty:   {predatorDiagnostics['starvationPenalty']:>8.2f}")
        print(f"  {'─' * 40}")
        print(f"  Total Fitness:        {predatorDiagnostics['totalFitness']:>8.2f}")

        print("\n" + "─" * 80)
        print("PREDATOR POPULATION STATISTICS")
        print("─" * 80)
        print(f"  Average Fitness:      {predData[0]:>8.2f}")
        print(f"  Best Fitness:         {predData[1]:>8.2f}")
        print(f"  Worst Fitness:        {predData[2]:>8.2f}")

        print("\n" + "─" * 80)
        print("PREY POPULATION STATISTICS")
        print("─" * 80)
        print(f"  Average Fitness:      {preyData[0]:>8.2f}")
        print(f"  Best Fitness:         {preyData[1]:>8.2f}")
        print(f"  Worst Fitness:        {preyData[2]:>8.2f}")
        print()

        if self.generationNo % self.checkpointControl == 0:
            self._saveGeneration()
            self._writeRecentCheckpoint(self.generationNo)

            self._savePredatorTelemetry(predatorTelemetryByID, self.generationNo)
            self._savePreyTelemetry(preyTelemetryByID, self.generationNo)

            if ENABLE_MESSAGE_LOGGING and messageLog:
                saveMessageLog(messageLog, self.generationNo)

        self.currentPredatorGeneration = self.predEvolver.produceNextGeneration(
            self.currentPredatorGeneration,
            kFittest=4
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

        print("! Generation fully simulated")
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

    def _savePredatorTelemetry(self, idLinkedTelemetries, generationNo):
        path = os.path.join(
            "Generations",
            f"Generation{generationNo}",
            "predatorTelemetry.csv"
        )

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "generationNo",
                    "genotypeID",
                    "catches",
                    "teamHuntScore",
                    "meanCommMagnitude",
                    "timeAlive",
                    "finalEnergy"
                ]
            )
            writer.writeheader()

            for genotypeID, telemetry in idLinkedTelemetries.items():
                writer.writerow({
                    "generationNo": generationNo,
                    "genotypeID": genotypeID,
                    "catches": telemetry["catches"],
                    "teamHuntScore": telemetry["teamHuntScore"],
                    "meanCommMagnitude" : telemetry["meanCommMagnitude"],
                    "timeAlive" : telemetry["timeAlive"],
                    "finalEnergy" : telemetry["finalEnergy"]
                })

    def _savePreyTelemetry(self, idLinkedTelemetries, generationNo):
        path = os.path.join(
            "Generations",
            f"Generation{generationNo}",
            "preyTelemetry.csv"
        )

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "generationNo",
                    "genotypeID",
                    "foodEaten",
                    "timeAlive",
                    "groupingScore",
                    "meanCommMagnitude",
                    "finalEnergy"
                ]
            )
            writer.writeheader()

            for genotypeID, telemetry in idLinkedTelemetries.items():
                writer.writerow({
                    "generationNo": generationNo,
                    "genotypeID": genotypeID,
                    "foodEaten" : telemetry["foodEaten"],
                    "timeAlive": telemetry["timeAlive"],
                    "groupingScore": telemetry["groupingScore"],
                    "meanCommMagnitude" : telemetry["meanCommMagnitude"],
                    "finalEnergy" : telemetry["finalEnergy"]
                })

    def ancestralOpponentContests(self, duration: float = 30.0, sampleRate: int = 10, savePath: str = None) -> dict:
        """
        Tests current generation against ancestral opponents to measure
        evolutionary progress. Tests both predators and prey separately.

        Args:
            duration: Simulation duration per match
            sampleRate: Test against every Nth generation (10 = every 10th)
            savePath: Optional path to save CSV results

        Returns:
            Dictionary containing:
                - currentGeneration: Current gen number
                - opponentGenerations: List of generations tested
                - predResults: Per-generation predator metrics
                - preyResults: Per-generation prey metrics
                - summary: Overall evolution metrics
        """
        print("\n" + "═" * 80)
        print(f"ANCESTRAL COMPETITION - Generation {self.generationNo}")
        print("═" * 80)

        # Save current state
        originalPredators = self.currentPredatorGeneration
        originalPrey = self.currentPreyGeneration
        originalGenNo = self.generationNo

        # Get available generations
        generations = self._scanGenerations()
        if not generations:
            print("! No ancestral generations found")
            return None

        # Sample generations (always include earliest and current)
        opponentGenerations = generations[::sampleRate]
        if generations[0] not in opponentGenerations:
            opponentGenerations.insert(0, generations[0])
        if originalGenNo not in opponentGenerations:
            opponentGenerations.append(originalGenNo)

        print(f"  Testing against {len(opponentGenerations)} ancestral generations")
        print(f"  Generations: {opponentGenerations}\n")

        # Storage for detailed results
        predResults = []
        preyResults = []

        for i, opponentGen in enumerate(opponentGenerations):
            print(f"\n  [{i + 1}/{len(opponentGenerations)}] Testing vs Generation {opponentGen}")
            print("  " + "─" * 76)

            # ─────────────────────────────────────────────────────────────
            # TEST 1: Current Predators vs Ancestral Prey
            # ─────────────────────────────────────────────────────────────
            try:
                _, ancestralPrey = self._loadGeneration(opponentGen)

                self.currentPredatorGeneration = originalPredators
                self.currentPreyGeneration = ancestralPrey

                _, predDiagnostics, predTelemetry, preyTelemetry = self._runSimulator(duration=duration)

                # Calculate individual fitnesses for variance analysis
                predFitnesses = [
                    FitnessFunctions.calculatePredatorFitness(t)
                    for t in predTelemetry.values()
                ]

                predResult = {
                    "opponentGeneration": opponentGen,
                    "avgFitness": sum(predFitnesses) / len(predFitnesses),
                    "maxFitness": max(predFitnesses),
                    "minFitness": min(predFitnesses),
                    "avgCatches": predDiagnostics["catches"],
                    "avgTeamHunt": predDiagnostics["teamHuntBonus"] / 800.0,  # Reverse weight
                    "avgSurvival": predDiagnostics["survivalBonus"] / 2.0,  # Reverse weight
                    "totalFitness": predDiagnostics["totalFitness"],
                }
                predResults.append(predResult)

                print(f"    Predators (Gen {originalGenNo}) vs Prey (Gen {opponentGen}):")
                print(
                    f"      Avg Catches: {predResult['avgCatches']:>6.2f} | Avg Fitness: {predResult['avgFitness']:>8.2f}")

            except Exception as e:
                print(f"    ! Error testing predators vs Gen {opponentGen}: {e}")
                continue

            # ─────────────────────────────────────────────────────────────
            # TEST 2: Ancestral Predators vs Current Prey
            # ─────────────────────────────────────────────────────────────
            try:
                ancestralPredators, _ = self._loadGeneration(opponentGen)

                self.currentPredatorGeneration = ancestralPredators
                self.currentPreyGeneration = originalPrey

                _, _, _, preyTelemetry = self._runSimulator(duration=duration)

                # Calculate individual prey fitnesses
                preyFitnesses = [
                    FitnessFunctions.calculatePreyFitness(t)
                    for t in preyTelemetry.values()
                ]

                # Calculate average food eaten and survival
                avgFoodEaten = sum(t.get("foodEaten", 0) for t in preyTelemetry.values()) / len(preyTelemetry)
                avgTimeAlive = sum(t["timeAlive"] for t in preyTelemetry.values()) / len(preyTelemetry)
                avgGrouping = sum(t["groupingScore"] for t in preyTelemetry.values()) / len(preyTelemetry)
                survivalRate = sum(1 for t in preyTelemetry.values() if t["timeAlive"] >= duration * 0.95) / len(
                    preyTelemetry)

                preyResult = {
                    "opponentGeneration": opponentGen,
                    "avgFitness": sum(preyFitnesses) / len(preyFitnesses),
                    "maxFitness": max(preyFitnesses),
                    "minFitness": min(preyFitnesses),
                    "avgFoodEaten": avgFoodEaten,
                    "avgTimeAlive": avgTimeAlive,
                    "avgGrouping": avgGrouping,
                    "survivalRate": survivalRate,
                }
                preyResults.append(preyResult)

                print(f"    Predators (Gen {opponentGen}) vs Prey (Gen {originalGenNo}):")
                print(
                    f"      Avg Food: {preyResult['avgFoodEaten']:>6.2f} | Survival: {preyResult['survivalRate'] * 100:>5.1f}% | Avg Fitness: {preyResult['avgFitness']:>8.2f}")

            except Exception as e:
                print(f"    ! Error testing prey vs Gen {opponentGen}: {e}")
                continue

        # Restore original state
        self.currentPredatorGeneration = originalPredators
        self.currentPreyGeneration = originalPrey
        self.generationNo = originalGenNo

        # ─────────────────────────────────────────────────────────────
        # Calculate Summary Statistics
        # ─────────────────────────────────────────────────────────────
        summary = self._calculateAncestralSummary(predResults, preyResults, originalGenNo)

        # Print comprehensive summary
        self._printAncestralSummary(summary, predResults, preyResults)

        # Save to CSV if requested
        if savePath:
            self._saveAncestralResults(predResults, preyResults, summary, savePath)

        return {
            "currentGeneration": originalGenNo,
            "opponentGenerations": opponentGenerations,
            "predResults": predResults,
            "preyResults": preyResults,
            "summary": summary
        }

    def _calculateAncestralSummary(self, predResults, preyResults, currentGen) -> dict:
        """Calculate evolutionary progress metrics from ancestral results."""
        if not predResults or not preyResults:
            return {}

        # Get oldest and current performance
        oldestPred = predResults[0]
        currentPred = predResults[-1]
        oldestPrey = preyResults[0]
        currentPrey = preyResults[-1]

        # Calculate improvements (how much current dominates ancestors)
        predImprovement = {
            "fitnessGain": currentPred["avgFitness"] - oldestPred["avgFitness"],
            "fitnessGainPct": ((currentPred["avgFitness"] / max(1.0, oldestPred["avgFitness"])) - 1) * 100,
            "catchesGain": currentPred["avgCatches"] - oldestPred["avgCatches"],
            "teamHuntGain": currentPred["avgTeamHunt"] - oldestPred["avgTeamHunt"],
        }

        preyImprovement = {
            "fitnessGain": currentPrey["avgFitness"] - oldestPrey["avgFitness"],
            "fitnessGainPct": ((currentPrey["avgFitness"] / max(1.0, oldestPrey["avgFitness"])) - 1) * 100,
            "foodGain": currentPrey["avgFoodEaten"] - oldestPrey["avgFoodEaten"],
            "survivalGain": currentPrey["survivalRate"] - oldestPrey["survivalRate"],
            "groupingGain": currentPrey["avgGrouping"] - oldestPrey["avgGrouping"],
        }

        # Calculate "Red Queen" coefficient
        # If both improving equally, ratio = 1.0 (balanced arms race)
        # If predator improving faster, ratio > 1.0
        # If prey improving faster, ratio < 1.0
        redQueenRatio = (predImprovement["fitnessGainPct"] + 1e-6) / (preyImprovement["fitnessGainPct"] + 1e-6)

        return {
            "currentGeneration": currentGen,
            "oldestGenerationTested": predResults[0]["opponentGeneration"],
            "predImprovement": predImprovement,
            "preyImprovement": preyImprovement,
            "redQueenRatio": redQueenRatio,
        }

    def _printAncestralSummary(self, summary, predResults, preyResults):
        """Print comprehensive summary of ancestral competition."""
        if not summary:
            print("\n  ! Insufficient data for summary")
            return

        print("\n" + "═" * 80)
        print("EVOLUTIONARY PROGRESS SUMMARY")
        print("═" * 80)

        pred = summary["predImprovement"]
        prey = summary["preyImprovement"]

        print(f"\n  Comparing Gen {summary['oldestGenerationTested']} → Gen {summary['currentGeneration']}")
        print(f"  ({summary['currentGeneration'] - summary['oldestGenerationTested']} generations of evolution)")

        print("\n" + "─" * 80)
        print("PREDATOR EVOLUTION")
        print("─" * 80)
        print(f"  Fitness Improvement:    {pred['fitnessGain']:>+8.2f} ({pred['fitnessGainPct']:>+6.1f}%)")
        print(f"  Catches Improvement:    {pred['catchesGain']:>+8.2f}")
        print(f"  Team Hunt Improvement:  {pred['teamHuntGain']:>+8.2f}")

        print("\n" + "─" * 80)
        print("PREY EVOLUTION")
        print("─" * 80)
        print(f"  Fitness Improvement:    {prey['fitnessGain']:>+8.2f} ({prey['fitnessGainPct']:>+6.1f}%)")
        print(f"  Food Eating Improvement:{prey['foodGain']:>+8.2f}")
        print(f"  Survival Rate Change:   {prey['survivalGain'] * 100:>+8.2f}%")
        print(f"  Grouping Improvement:   {prey['groupingGain']:>+8.2f}")

        print("\n" + "─" * 80)
        print("RED QUEEN DYNAMICS")
        print("─" * 80)
        rq = summary["redQueenRatio"]
        if 0.8 <= rq <= 1.25:
            status = "✓ BALANCED ARMS RACE"
        elif rq > 1.25:
            status = "⚠ PREDATORS DOMINATING"
        else:
            status = "⚠ PREY DOMINATING"
        print(f"  Red Queen Ratio: {rq:>6.2f} → {status}")
        print("  (Ratio of predator improvement to prey improvement)")
        print("  Ideal: 0.8-1.25 (both species co-evolving)")

        # Per-generation table
        print("\n" + "─" * 80)
        print("PER-GENERATION RESULTS")
        print("─" * 80)
        print(
            f"  {'Gen':>5} | {'Pred Fit':>10} | {'Pred Catch':>10} | {'Prey Fit':>10} | {'Prey Food':>10} | {'Survival':>9}")
        print("  " + "─" * 76)

        for pred, prey in zip(predResults, preyResults):
            print(f"  {pred['opponentGeneration']:>5} | "
                  f"{pred['avgFitness']:>10.2f} | "
                  f"{pred['avgCatches']:>10.2f} | "
                  f"{prey['avgFitness']:>10.2f} | "
                  f"{prey['avgFoodEaten']:>10.2f} | "
                  f"{prey['survivalRate'] * 100:>8.1f}%")

        print("═" * 80 + "\n")

    def _saveAncestralResults(self, predResults, preyResults, summary, savePath):
        """Save ancestral competition results to CSV files."""
        os.makedirs(savePath, exist_ok=True)

        # Save predator results
        predPath = os.path.join(savePath, f"ancestralPredators_Gen{summary['currentGeneration']}.csv")
        with open(predPath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "opponentGeneration", "avgFitness", "maxFitness", "minFitness",
                "avgCatches", "avgTeamHunt", "avgSurvival", "totalFitness"
            ])
            writer.writeheader()
            writer.writerows(predResults)

        # Save prey results
        preyPath = os.path.join(savePath, f"ancestralPrey_Gen{summary['currentGeneration']}.csv")
        with open(preyPath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "opponentGeneration", "avgFitness", "maxFitness", "minFitness",
                "avgFoodEaten", "avgTimeAlive", "avgGrouping", "survivalRate"
            ])
            writer.writeheader()
            writer.writerows(preyResults)

        # Save summary
        summaryPath = os.path.join(savePath, f"ancestralSummary_Gen{summary['currentGeneration']}.csv")
        with open(summaryPath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Current Generation", summary["currentGeneration"]])
            writer.writerow(["Oldest Generation Tested", summary["oldestGenerationTested"]])
            writer.writerow(["Predator Fitness Gain", summary["predImprovement"]["fitnessGain"]])
            writer.writerow(["Predator Fitness Gain %", summary["predImprovement"]["fitnessGainPct"]])
            writer.writerow(["Predator Catches Gain", summary["predImprovement"]["catchesGain"]])
            writer.writerow(["Predator Team Hunt Gain", summary["predImprovement"]["teamHuntGain"]])
            writer.writerow(["Prey Fitness Gain", summary["preyImprovement"]["fitnessGain"]])
            writer.writerow(["Prey Fitness Gain %", summary["preyImprovement"]["fitnessGainPct"]])
            writer.writerow(["Prey Food Gain", summary["preyImprovement"]["foodGain"]])
            writer.writerow(["Prey Survival Gain", summary["preyImprovement"]["survivalGain"]])
            writer.writerow(["Red Queen Ratio", summary["redQueenRatio"]])

        print(f"\n  ✓ Results saved to {savePath}")

    def _scanGenerations(self) -> list:
        generations = []
        for item in os.listdir("Generations"):
            if item.startswith("Generation") and os.path.isdir(os.path.join("Generations", item)):
                try:
                    generations.append(
                        int(item.replace("Generation", ""))
                    )
                except ValueError:
                    continue
        return sorted(generations)

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