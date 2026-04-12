import matplotlib
import matplotlib.colors as colors
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas
import torch

from GenerationController import GenerationController
import Simulator
from NeuralNetwork import NeuralNetwork
import DataAnalysers

def train(duration):
    runForNGenerations = 100

    generationController = GenerationController(
        predPopSize=16,
        preyPopSize=16,
        checkpointControl=5
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

    ax.set_title("Evolutionary Progress: Predator vs Prey Fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness Score")
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

if __name__ == "__main__":
    train(
        duration = 15.0
    )

# generation 185 has interesting emergent behaviour from prey population
# - prey population move like one big organism. movement and direction of movement is synced.
# - all prey members participating in the big organism are equally spaced.
# - distance between prey members is equal to the communication radius limit. done to maintain max distance without losing connection from network?