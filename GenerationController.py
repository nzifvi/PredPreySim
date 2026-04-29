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

RETRIAL_AMOUNT      = 2
WORKER_COUNT        = 18

SIMULATION_BATCH_SIZE   = 4
SIMULATION_REPEAT_COUNT = 2

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
            tournamentSize = 3,
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
                    "timeAlive": telemetry["timeAlive"],
                    "groupingScore": telemetry["groupingScore"],
                    "meanCommMagnitude" : telemetry["meanCommMagnitude"],
                    "finalEnergy" : telemetry["finalEnergy"]
                })

    def ancestralOpponentContests(self, duration:float = 30.0, sampleRate:int = 10) -> dict:
        originalPredators = self.currentPredatorGeneration
        originalPrey      = self.currentPreyGeneration

        generations = self._scanGenerations()
        opponentGenerations = generations[::sampleRate]

        predPerformance = []
        preyPerformance = []

        for i, opponentGen in enumerate(opponentGenerations):
            opponentPredators, opponentPrey = self._loadGeneration(opponentGen)
            self.currentPredatorGeneration = originalPredators
            self.currentPreyGeneration = opponentPrey

            _, predDiagnostics, _, _ = self._runSimulator(duration = duration)
            predPerformance.append(
                predDiagnostics["totalFitness"]
            )

            self.currentPredatorGeneration = opponentPredators
            self.currentPreyGeneration = originalPrey
            _, _, _, preyTelemetry = self._runSimulator(duration = duration)
            preyFitnesses = []
            for gID, telemetry in preyTelemetry.items():
                preyFitnesses.append(FitnessFunctions.calculatePreyFitness(telemetry))
            preyPerformance.append(
                sum(preyFitnesses) / len(preyFitnesses)
            )

        self.currentPredatorGeneration = originalPredators
        self.currentPreyGeneration     = originalPrey

        return {
            "currentGeneration"   : self.generationNo,
            "opponentGenerations" : opponentGenerations,
            "predPerformance"     : predPerformance,
            "preyPerformance"     : preyPerformance
        }

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