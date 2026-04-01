import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from GenerationController import GenerationController

if __name__ == "__main__":
    runForNGenerations      = 100

    generationController = GenerationController(
        predPopSize = 16,
        preyPopSize = 16,
        checkpointControl = 5
    )

    predData = {
        "avgFitness"   : [],
        "bestFitness"  : [],
        "worstFitness" : []
    }
    preyData = {
        "avgFitness"   : [],
        "bestFitness"  : [],
        "worstFitness" : []
    }
    averageFitnessOverTime = []
    bestFitnessOverTime = []
    worstFitnessOverTime = []
    startGen = generationController.generationNo

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))

    line_avg, = ax.plot([], [], 'b-', linewidth=2, label='Average Fitness')

    error_bars = None

    ax.set_title("Evolutionary Progress: Population Fitness Variance")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness Score")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left')

    for gen in range(runForNGenerations):
        predData, preyData = generationController.run(

        )
        predAvgFitness, predBestFitness, predWorstFitness = predData
        preyAvgFitness, preyBestFitness, preyWorstFitness = preyData

        print(f"pred avg fitness {predAvgFitness} | pred best fitness {predBestFitness} | pred worst fitness {predWorstFitness}")
        print(f"prey avg fitness {preyAvgFitness} | prey best fitness {preyBestFitness} | prey worst fitness {preyWorstFitness}")

        current_x = list(range(startGen, startGen + len(averageFitnessOverTime)))

        line_avg.set_data(current_x, averageFitnessOverTime)

        lower_err = [a - w for a, w in zip(averageFitnessOverTime, worstFitnessOverTime)]
        upper_err = [b - a for a, b in zip(averageFitnessOverTime, bestFitnessOverTime)]
        yerr = [lower_err, upper_err]

        if error_bars:
            error_bars.remove()

        #error_bars = ax.errorbar(
        #    current_x,
        #    averageFitnessOverTime,
        #    yerr=yerr,
        #    fmt='none',
        #    ecolor='skyblue',
        #    elinewidth=1,
        #    capsize=3,
        #    alpha=0.7
        #)

        ax.relim()
        ax.autoscale_view()

        plt.draw()
        plt.pause(0.1)

    plt.ioff()
    plt.show()