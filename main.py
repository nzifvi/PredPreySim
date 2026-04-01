import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from GenerationController import GenerationController
import Simulator
from NeuralNetwork import NeuralNetwork

def train():
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
        predData, preyData = generationController.run()

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

def observe():
    generationCount = 5
    predPopSize = 16
    preyPopSize = 16
    simDuration = 30.0
    i = 0


    while i <= generationCount:
        predatorNNs = []
        preyNNs = []
        for j in range(predPopSize):
            predatorNNs.append(
                NeuralNetwork(
                    generationNo = i,
                    genotypeID = j,
                    isPredator = True
                )
            )
        for j in range(preyPopSize):
            preyNNs.append(
                NeuralNetwork(
                    generationNo = i,
                    genotypeID = j,
                    isPredator = False
                )
            )

        sim = Simulator.Simulator(
            numPredators = predPopSize,
            numPrey = preyPopSize,
            simDuration = simDuration,
            gui = True
        )

        print(
            sim.runSimulation(
                predatorNNs,
                preyNNs
            )
        )

        i = i + 5


if __name__ == "__main__":
    train()