import matplotlib
import matplotlib.colors as colors

import FitnessFunctions
from DataAnalysers import analyseAutocorrelation

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas
import torch
import numpy

from GenerationController import GenerationController
import Simulator
from NeuralNetwork import NeuralNetwork
import DataAnalysers

def train(duration):
    runForNGenerations = 100

    generationController = GenerationController(
        predPopSize=16,
        preyPopSize=16,
        checkpointControl=1
    )

    predAvgHistory = []
    predBestHistory = []
    predWorstHistory = []

    preyAvgHistory = []
    preyBestHistory = []
    preyWorstHistory = []

    startGen = generationController.generationNo

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))

    line_pred, = ax.plot([], [], "b-", linewidth=2, label="Predator Avg Fitness")
    line_prey, = ax.plot([], [], "r-", linewidth=2, label="Prey Avg Fitness")

    predator_errorbars = None
    prey_errorbars = None

    ax.set_title("evolution? ahhhh")
    ax.set_xlabel("gen")
    ax.set_ylabel("fitness")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")

    for gen in range(runForNGenerations):
        predData, preyData = generationController.run(duration)

        predAvgFitness, predBestFitness, predWorstFitness = predData
        preyAvgFitness, preyBestFitness, preyWorstFitness = preyData

        predAvgHistory.append(predAvgFitness)
        predBestHistory.append(predBestFitness)
        predWorstHistory.append(predWorstFitness)

        preyAvgHistory.append(preyAvgFitness)
        preyBestHistory.append(preyBestFitness)
        preyWorstHistory.append(preyWorstFitness)

        print(
            f"Generation {startGen + gen} | "
            f"pred avg {predAvgFitness:.2f} | pred best {predBestFitness:.2f} | pred worst {predWorstFitness:.2f}"
        )
        print(
            f"Generation {startGen + gen} | "
            f"prey avg {preyAvgFitness:.2f} | prey best {preyBestFitness:.2f} | prey worst {preyWorstFitness:.2f}"
        )

        current_x = list(range(startGen, startGen + len(predAvgHistory)))

        line_pred.set_data(current_x, predAvgHistory)
        line_prey.set_data(current_x, preyAvgHistory)

        pred_lower_err = [avg - worst for avg, worst in zip(predAvgHistory, predWorstHistory)]
        pred_upper_err = [best - avg for avg, best in zip(predAvgHistory, predBestHistory)]
        pred_yerr = [pred_lower_err, pred_upper_err]

        prey_lower_err = [avg - worst for avg, worst in zip(preyAvgHistory, preyWorstHistory)]
        prey_upper_err = [best - avg for avg, best in zip(preyAvgHistory, preyBestHistory)]
        prey_yerr = [prey_lower_err, prey_upper_err]

        if predator_errorbars is not None:
            predator_errorbars.remove()
        if prey_errorbars is not None:
            prey_errorbars.remove()

        predator_errorbars = ax.errorbar(
            current_x,
            predAvgHistory,
            yerr=pred_yerr,
            fmt="none",
            ecolor="blue",
            elinewidth=1,
            capsize=3,
            alpha=0.5
        )

        prey_errorbars = ax.errorbar(
            current_x,
            preyAvgHistory,
            yerr=prey_yerr,
            fmt="none",
            ecolor="red",
            elinewidth=1,
            capsize=3,
            alpha=0.5
        )

        ax.relim()
        ax.autoscale_view()

        plt.draw()
        plt.pause(0.1)

    plt.ioff()
    plt.show()

def observe(generationNo:int, predators:list, prey:list, duration:float):
    predPopSize = len(predators)
    preyPopSize = len(prey)

    predatorNNs = []
    preyNNs = []

    for i in range(0, len(predators)):
        predatorNNs.append(
            NeuralNetwork(
                generationNo = generationNo,
                genotypeID   = predators[i]
            )
        )

    for i in range(0, len(prey)):
        preyNNs.append(
            NeuralNetwork(
                generationNo = generationNo,
                genotypeID   = prey[i]
            )
        )

    sim = Simulator.Simulator(
            numPredators = predPopSize,
            numPrey = preyPopSize,
            simDuration = duration,
            gui = True
    )

    sim.runSimulation(
        predatorNNs = predatorNNs,
        preyNNs = preyNNs,
    )

def predatorAndPreyMessageHeatmaps(generationNo:int, bins:int) -> None:
    processor = DataAnalysers.MessageLogProcessor.fromCsv(
        f"Generations/Generation{generationNo}/messageLog.csv"
    )

    pred = processor.species("predator").df
    prey = processor.species("prey").df

    plt.subplot(1, 2, 1)
    plt.hist2d(
        pred["msg0"],
        pred["msg1"],
        bins=bins,
        norm=colors.LogNorm()
    )
    plt.xlabel("msg0")
    plt.ylabel("msg1")

    plt.subplot(1, 2, 2)
    plt.hist2d(
        prey["msg0"],
        prey["msg1"],
        bins=bins,
        norm=colors.LogNorm()
    )
    plt.xlabel("msg0")
    plt.ylabel("msg1")

    plt.tight_layout()
    plt.show()

def contextHeatmaps(generationNo:int, bins:int) -> None:
    processor = DataAnalysers.MessageLogProcessor.fromCsv(
        f"Generations/Generation{generationNo}/messageLog.csv"
    )

    enemy = processor.enemyVisibleOnly().df
    noEnemy = processor.df[~processor.df["seesEnemy"]]

    fig, axes = plt.subplots(1, 2, figsize = (12, 5))

    h1 = axes[0].hist2d(
        noEnemy["msg0"],
        noEnemy["msg1"],
        bins = bins,
        norm=colors.LogNorm()
    )
    axes[0].set_title("no enemy visible")
    axes[0].set_xlabel("msg0")
    axes[0].set_ylabel("msg1")

    h2 = axes[1].hist2d(
        enemy["msg0"],
        enemy["msg1"],
        bins = bins,
        norm = colors.LogNorm()
    )
    axes[1].set_title("enemy visible")
    axes[1].set_xlabel("msg0")
    axes[1].set_ylabel("msg1")

    plt.tight_layout()
    plt.show()

def contextMessageCorrelations(generationNo:int) -> None:
    processor = DataAnalysers.MessageLogProcessor.fromCsv(
        f"Generations/Generation{generationNo}/messageLog.csv"
    )

    corr = processor.messageCorrelationTable()

    print("\n=== Context -> Message Correlations ===")
    print(
        corr.loc[
            ["nearestEnemyDist", "nearestAllyDist", "visibleEnemyCount", "visibleAllyCount"],
            ["msg0", "msg1", "receivedMessage0", "receivedMessage1"]
        ]
    )
    print("\n === Mean messages by visibleEnemyCount ===")
    print(
        processor.df.groupby("visibleEnemyCount")[["msg0", "msg1"]].mean().sort_index()
    )
    print("\n === Mean messages by visibleAllyCount ===")
    print(
        processor.df.groupby("visibleAllyCount")[["msg0", "msg1"]].mean().sort_index()
    )

def messageVsMovementHeatmaps(generationNo:int, bins:int = 100) -> None:
    processor = DataAnalysers.MessageLogProcessor.fromCsv(
        f"Generations/Generation{generationNo}/messageLog.csv"
    )

    df = processor.df.copy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].hist2d(df["receivedMessage0"], df["move0"], bins = bins, norm = colors.LogNorm())
    axes[0, 0].set_title("receivedMessage0 vs move0")
    axes[0, 0].set_xlabel("msg0")
    axes[0, 0].set_ylabel("move0")

    axes[0, 1].hist2d(df["receivedMessage0"], df["move1"], bins = bins, norm = colors.LogNorm())
    axes[0, 1].set_title("receivedMessage0 vs move1")
    axes[0, 1].set_xlabel("msg0")
    axes[0, 1].set_ylabel("move1")

    axes[1, 0].hist2d(df["receivedMessage1"], df["move0"], bins = bins, norm = colors.LogNorm())
    axes[1, 0].set_title("receivedMessage1 vs move0")
    axes[1, 0].set_xlabel("msg1")
    axes[1, 0].set_ylabel("move0")

    axes[1, 1].hist2d(df["receivedMessage1"], df["move1"], bins = bins, norm = colors.LogNorm())
    axes[1, 1].set_title("receivedMessage1 vs move1")
    axes[1, 1].set_xlabel("msg1")
    axes[1, 1].set_ylabel("move1")

    plt.tight_layout()
    plt.show()

def plotGenerationalFitness() -> None:
    f = open("Generations/GenerationCount.txt", "r")
    genCount = int(f.read())
    f.close()

    generations = []
    i = 0
    while i <= genCount:
        generations.append(i)
        if i < 200:
            i = i +5
        else:
            i = i + 1

    preyGenerationFitness = []
    predGenerationFitness = []

    for j in generations:
        predDF = pandas.read_csv(f"Generations/Generation{j}/predatorTelemetry.csv")
        preyDF = pandas.read_csv(f"Generations/Generation{j}/preyTelemetry.csv")
        predFitnesses = []
        preyFitnesses = []
        for k in range(0, len(predDF)):
            predRow = predDF.iloc[k]
            preyRow = preyDF.iloc[k]

            predFitnesses.append(
                FitnessFunctions.calculatePredatorFitness(
                    FitnessFunctions.calculatePredatorFitnessBreakdown(
                        {
                            "catches" : predRow["catches"],
                            "teamHuntScore" : predRow["teamHuntScore"],
                            "meanNearestPreyDistance" : predRow["meanNearestPreyDistance"]
                        }
                    )
                )
            )
            preyFitnesses.append(
                FitnessFunctions.calculatePreyFitness(
                    {
                        "timeAlive" : preyRow["timeAlive"],
                        "alive" : preyRow["alive"],
                        "groupingScore" : preyRow["groupingScore"]
                    }
                )
            )

        predAvgFitness = sum(predFitnesses) / len(predFitnesses)
        minPredFitness = min(predFitnesses)
        maxPredFitness = max(predFitnesses)

        preyAvgFitness = sum(preyFitnesses) / len(preyFitnesses)
        minPreyFitness = min(preyFitnesses)
        maxPreyFitness = max(preyFitnesses)
        predGenerationFitness.append(
            (predAvgFitness, minPredFitness, maxPredFitness)
        )
        preyGenerationFitness.append(
            (preyAvgFitness, minPreyFitness, maxPreyFitness)
        )

    for i in range(0, len(generations)):
        print(f"Generation {generations[i]}:")
        avgPredFitness, minPredFitness, maxPredFitness = predGenerationFitness[i]
        avgPreyFitness, minPreyFitness, maxPreyFitness = preyGenerationFitness[i]
        print(f" |    avg pred. fitness: {avgPredFitness:.4f}, min pred. fitness: {minPredFitness:.4f}, max pred. fitness: {maxPreyFitness:.4f}")
        print(f" |    avg prey fitness: {avgPreyFitness:.4f}, min prey fitness: {minPreyFitness:.4f}, max prey fitness: {maxPreyFitness:.4f}\n")

    generations_array = numpy.array(generations)

    pred_avg = numpy.array([f[0] for f in predGenerationFitness])
    pred_min = numpy.array([f[1] for f in predGenerationFitness])
    pred_max = numpy.array([f[2] for f in predGenerationFitness])

    prey_avg = numpy.array([f[0] for f in preyGenerationFitness])
    prey_min = numpy.array([f[1] for f in preyGenerationFitness])
    prey_max = numpy.array([f[2] for f in preyGenerationFitness])

    # Calculate error bars
    pred_error_lower = pred_avg - pred_min
    pred_error_upper = pred_max - pred_avg

    prey_error_lower = prey_avg - prey_min
    prey_error_upper = prey_max - prey_avg

    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Plot predator fitness
    ax.errorbar(
        generations_array,
        pred_avg,
        yerr=[pred_error_lower, pred_error_upper],
        fmt='o-',
        linewidth=2.5,
        markersize=7,
        capsize=5,
        capthick=2,
        color='#2E86AB',
        ecolor='#2E86AB',
        label='Predator Mean',
        alpha=0.8
    )

    # Plot prey fitness
    ax.errorbar(
        generations_array,
        prey_avg,
        yerr=[prey_error_lower, prey_error_upper],
        fmt='s-',
        linewidth=2.5,
        markersize=7,
        capsize=5,
        capthick=2,
        color='#F18F01',
        ecolor='#F18F01',
        label='Prey Mean',
        alpha=0.8
    )

    # Fill between for visual effect
    ax.fill_between(generations_array, pred_min, pred_max, alpha=0.15, color='#2E86AB')
    ax.fill_between(generations_array, prey_min, prey_max, alpha=0.15, color='#F18F01')

    ax.set_xlabel('Generation', fontsize=13)
    ax.set_ylabel('Fitness', fontsize=13)
    ax.set_title('Predator-Prey Fitness Evolution Over Generations', fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig('generational_fitness_combined.png', dpi=300, bbox_inches='tight')
    print("✓ Saved plot: generational_fitness_combined.png")
    plt.show()

def plotAncestralContest(results:dict) -> None:
    currentGen = results["currentGeneration"]
    opponentGenerations = results["opponentGenerations"]
    predPerformance = results["predPerformance"]
    preyPerformance = results["preyPerformance"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14, 6))
    ax1.plot(
        opponentGenerations,
        predPerformance
    )

    ax2.plot(
        opponentGenerations,
        preyPerformance
    )

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    plotGenerationalFitness()