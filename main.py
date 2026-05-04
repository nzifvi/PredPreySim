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

import os
import csv
import statistics


def train(duration):
    runForNGenerations = 100

    generationController = GenerationController(
        predPopSize=32,
        preyPopSize=32,
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


def observe(generationNo: int, predators: list, prey: list, duration: float):
    predPopSize = len(predators)
    preyPopSize = len(prey)

    predatorNNs = []
    preyNNs = []

    for i in range(0, len(predators)):
        predatorNNs.append(
            NeuralNetwork(
                generationNo=generationNo,
                genotypeID=predators[i]
            )
        )

    for i in range(0, len(prey)):
        preyNNs.append(
            NeuralNetwork(
                generationNo=generationNo,
                genotypeID=prey[i]
            )
        )

    sim = Simulator.Simulator(
        numPredators=predPopSize,
        numPrey=preyPopSize,
        simDuration=duration,
        gui=True
    )

    sim.runSimulation(
        predatorNNs=predatorNNs,
        preyNNs=preyNNs,
    )


def predatorAndPreyMessageHeatmaps(generationNo: int, bins: int) -> None:
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


def contextHeatmaps(generationNo: int, bins: int) -> None:
    processor = DataAnalysers.MessageLogProcessor.fromCsv(
        f"Generations/Generation{generationNo}/messageLog.csv"
    )

    enemy = processor.enemyVisibleOnly().df
    noEnemy = processor.df[~processor.df["seesEnemy"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    h1 = axes[0].hist2d(
        noEnemy["msg0"],
        noEnemy["msg1"],
        bins=bins,
        norm=colors.LogNorm()
    )
    axes[0].set_title("no enemy visible")
    axes[0].set_xlabel("msg0")
    axes[0].set_ylabel("msg1")

    h2 = axes[1].hist2d(
        enemy["msg0"],
        enemy["msg1"],
        bins=bins,
        norm=colors.LogNorm()
    )
    axes[1].set_title("enemy visible")
    axes[1].set_xlabel("msg0")
    axes[1].set_ylabel("msg1")

    plt.tight_layout()
    plt.show()


def contextMessageCorrelations(generationNo: int) -> None:
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


def messageVsMovementHeatmaps(generationNo: int, bins: int = 100) -> None:
    processor = DataAnalysers.MessageLogProcessor.fromCsv(
        f"Generations/Generation{generationNo}/messageLog.csv"
    )

    df = processor.df.copy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].hist2d(df["receivedMessage0"], df["move0"], bins=bins, norm=colors.LogNorm())
    axes[0, 0].set_title("receivedMessage0 vs move0")
    axes[0, 0].set_xlabel("msg0")
    axes[0, 0].set_ylabel("move0")

    axes[0, 1].hist2d(df["receivedMessage0"], df["move1"], bins=bins, norm=colors.LogNorm())
    axes[0, 1].set_title("receivedMessage0 vs move1")
    axes[0, 1].set_xlabel("msg0")
    axes[0, 1].set_ylabel("move1")

    axes[1, 0].hist2d(df["receivedMessage1"], df["move0"], bins=bins, norm=colors.LogNorm())
    axes[1, 0].set_title("receivedMessage1 vs move0")
    axes[1, 0].set_xlabel("msg1")
    axes[1, 0].set_ylabel("move0")

    axes[1, 1].hist2d(df["receivedMessage1"], df["move1"], bins=bins, norm=colors.LogNorm())
    axes[1, 1].set_title("receivedMessage1 vs move1")
    axes[1, 1].set_xlabel("msg1")
    axes[1, 1].set_ylabel("move1")

    plt.tight_layout()
    plt.show()


def plotGenerationalFitness() -> None:
    """✅ UPDATED: Uses new fitness metrics"""
    f = open("Generations/GenerationCount.txt", "r")
    genCount = int(f.read())
    f.close()

    generations = list(range(0, genCount + 1))
    preyGenerationFitness = []
    predGenerationFitness = []

    for i in generations:
        predDF = pandas.read_csv(f"Generations/Generation{i}/predatorTelemetry.csv")
        preyDF = pandas.read_csv(f"Generations/Generation{i}/preyTelemetry.csv")
        predFitnesses = []
        preyFitnesses = []

        for k in range(0, len(predDF)):
            predRow = predDF.iloc[k]
            preyRow = preyDF.iloc[k]

            # ✅ UPDATED: Use new telemetry fields
            predFitnesses.append(
                FitnessFunctions.calculatePredatorFitnessBreakdown(
                    {
                        "catches": predRow["catches"],
                        "teamHuntScore": predRow["teamHuntScore"],
                        "meanCommMagnitude": predRow["meanCommMagnitude"],
                        "timeAlive": predRow["timeAlive"],
                        "finalEnergy": predRow["finalEnergy"]
                    }
                )["totalFitness"]  # ✅ Get totalFitness from breakdown
            )

            preyFitnesses.append(
                FitnessFunctions.calculatePreyFitness(
                    {
                        "timeAlive": preyRow["timeAlive"],
                        "groupingScore": preyRow["groupingScore"],
                        "meanCommMagnitude": preyRow["meanCommMagnitude"],
                        "finalEnergy": preyRow["finalEnergy"]
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
        print(
            f" |    avg pred. fitness: {avgPredFitness:.4f}, min pred. fitness: {minPredFitness:.4f}, max pred. fitness: {maxPredFitness:.4f}")
        print(
            f" |    avg prey fitness: {avgPreyFitness:.4f}, min prey fitness: {minPreyFitness:.4f}, max prey fitness: {maxPreyFitness:.4f}\n")

    generations_array = numpy.array(generations)

    pred_avg = numpy.array([f[0] for f in predGenerationFitness])
    pred_min = numpy.array([f[1] for f in predGenerationFitness])
    pred_max = numpy.array([f[2] for f in predGenerationFitness])

    prey_avg = numpy.array([f[0] for f in preyGenerationFitness])
    prey_min = numpy.array([f[1] for f in preyGenerationFitness])
    prey_max = numpy.array([f[2] for f in preyGenerationFitness])

    pred_error_lower = pred_avg - pred_min
    pred_error_upper = pred_max - pred_avg

    prey_error_lower = prey_avg - prey_min
    prey_error_upper = prey_max - prey_avg

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

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


def plotAncestralContest(results: dict) -> None:
    currentGen = results["currentGeneration"]
    opponentGenerations = results["opponentGenerations"]
    predPerformance = results["predPerformance"]
    preyPerformance = results["preyPerformance"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
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


def analyze_all_generations():
    """✅ UPDATED: Analyze all generation data with new metrics."""

    generations = []

    # Predator metrics
    pred_mean_catches = []
    pred_max_catches = []
    pred_mean_teamwork = []
    pred_max_teamwork = []
    pred_mean_comm = []  # ✅ NEW
    pred_mean_survival_time = []  # ✅ NEW
    pred_mean_energy = []  # ✅ NEW

    # Prey metrics
    prey_mean_survival_time = []  # ✅ CHANGED from survival_rate
    prey_mean_grouping = []
    prey_max_grouping = []
    prey_mean_comm = []  # ✅ NEW
    prey_mean_energy = []  # ✅ NEW

    print("Scanning generations...")

    for gen in range(1000):  # Scan up to gen 1000
        pred_path = f"Generations/Generation{gen}/predatorTelemetry.csv"
        prey_path = f"Generations/Generation{gen}/preyTelemetry.csv"

        if not os.path.exists(pred_path):
            break  # Stop when no more generations

        generations.append(gen)

        # ✅ UPDATED: Read new predator fields
        with open(pred_path, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)

            catches = [float(r['catches']) for r in data]
            teamwork = [float(r['teamHuntScore']) for r in data]
            comm = [float(r['meanCommMagnitude']) for r in data]  # ✅ NEW
            timeAlive = [float(r['timeAlive']) for r in data]  # ✅ NEW
            energy = [float(r['finalEnergy']) for r in data]  # ✅ NEW

            pred_mean_catches.append(statistics.mean(catches))
            pred_max_catches.append(max(catches))
            pred_mean_teamwork.append(statistics.mean(teamwork))
            pred_max_teamwork.append(max(teamwork))
            pred_mean_comm.append(statistics.mean(comm))  # ✅ NEW
            pred_mean_survival_time.append(statistics.mean(timeAlive))  # ✅ NEW
            pred_mean_energy.append(statistics.mean(energy))  # ✅ NEW

        # ✅ UPDATED: Read new prey fields
        if os.path.exists(prey_path):
            with open(prey_path, 'r') as f:
                reader = csv.DictReader(f)
                data = list(reader)

                timeAlive = [float(r['timeAlive']) for r in data]  # ✅ CHANGED
                grouping = [float(r['groupingScore']) for r in data]
                comm = [float(r['meanCommMagnitude']) for r in data]  # ✅ NEW
                energy = [float(r['finalEnergy']) for r in data]  # ✅ NEW

                prey_mean_survival_time.append(statistics.mean(timeAlive))  # ✅ CHANGED
                prey_mean_grouping.append(statistics.mean(grouping))
                prey_max_grouping.append(max(grouping))
                prey_mean_comm.append(statistics.mean(comm))  # ✅ NEW
                prey_mean_energy.append(statistics.mean(energy))  # ✅ NEW

    return {
        'generations': generations,
        'pred_mean_catches': pred_mean_catches,
        'pred_max_catches': pred_max_catches,
        'pred_mean_teamwork': pred_mean_teamwork,
        'pred_max_teamwork': pred_max_teamwork,
        'pred_mean_comm': pred_mean_comm,  # ✅ NEW
        'pred_mean_survival_time': pred_mean_survival_time,  # ✅ NEW
        'pred_mean_energy': pred_mean_energy,  # ✅ NEW
        'prey_mean_survival_time': prey_mean_survival_time,  # ✅ CHANGED
        'prey_mean_grouping': prey_mean_grouping,
        'prey_max_grouping': prey_max_grouping,
        'prey_mean_comm': prey_mean_comm,  # ✅ NEW
        'prey_mean_energy': prey_mean_energy,  # ✅ NEW
    }


def plot_evolution(data):
    """✅ UPDATED: Create visualization of evolution with new metrics."""

    fig, axes = plt.subplots(4, 2, figsize=(15, 16))

    # Predator catches
    axes[0, 0].plot(data['generations'], data['pred_mean_catches'], label='Mean', alpha=0.7)
    axes[0, 0].plot(data['generations'], data['pred_max_catches'], label='Best', alpha=0.7)
    axes[0, 0].set_title('Predator Catches per Generation')
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Catches')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Predator teamwork
    axes[0, 1].plot(data['generations'], data['pred_mean_teamwork'], label='Mean', alpha=0.7)
    axes[0, 1].plot(data['generations'], data['pred_max_teamwork'], label='Best', alpha=0.7)
    axes[0, 1].set_title('Predator Teamwork Score')
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Teamwork Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # ✅ NEW: Predator communication
    axes[1, 0].plot(data['generations'], data['pred_mean_comm'], alpha=0.7, color='purple')
    axes[1, 0].set_title('Predator Communication Magnitude')
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Comm Magnitude')
    axes[1, 0].grid(True, alpha=0.3)

    # ✅ NEW: Predator survival time
    axes[1, 1].plot(data['generations'], data['pred_mean_survival_time'], alpha=0.7, color='blue')
    axes[1, 1].set_title('Predator Mean Survival Time')
    axes[1, 1].set_xlabel('Generation')
    axes[1, 1].set_ylabel('Time Alive (s)')
    axes[1, 1].grid(True, alpha=0.3)

    # ✅ NEW: Predator energy
    axes[2, 0].plot(data['generations'], data['pred_mean_energy'], alpha=0.7, color='orange')
    axes[2, 0].set_title('Predator Mean Final Energy')
    axes[2, 0].set_xlabel('Generation')
    axes[2, 0].set_ylabel('Energy')
    axes[2, 0].grid(True, alpha=0.3)

    # Prey grouping
    axes[2, 1].plot(data['generations'], data['prey_mean_grouping'], label='Mean', alpha=0.7)
    axes[2, 1].plot(data['generations'], data['prey_max_grouping'], label='Best', alpha=0.7)
    axes[2, 1].set_title('Prey Grouping Score')
    axes[2, 1].set_xlabel('Generation')
    axes[2, 1].set_ylabel('Grouping Score')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    # ✅ CHANGED: Prey survival time (not percentage)
    axes[3, 0].plot(data['generations'], data['prey_mean_survival_time'], alpha=0.7, color='green')
    axes[3, 0].set_title('Prey Mean Survival Time')
    axes[3, 0].set_xlabel('Generation')
    axes[3, 0].set_ylabel('Time Alive (s)')
    axes[3, 0].grid(True, alpha=0.3)

    # ✅ NEW: Prey energy
    axes[3, 1].plot(data['generations'], data['prey_mean_energy'], alpha=0.7, color='red')
    axes[3, 1].set_title('Prey Mean Final Energy')
    axes[3, 1].set_xlabel('Generation')
    axes[3, 1].set_ylabel('Energy')
    axes[3, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('evolution_analysis_updated.png', dpi=150)
    print("✓ Saved plot: evolution_analysis_updated.png")
    plt.show()


def detect_plateau(data):
    """Detect when evolution plateaued."""

    catches = data['pred_mean_catches']

    window = 50
    threshold = 0.05

    for i in range(len(catches) - window):
        start_val = catches[i]
        end_val = catches[i + window]

        if start_val > 0:
            improvement = (end_val - start_val) / start_val

            if abs(improvement) < threshold:
                gen = data['generations'][i]
                print(f"\n⚠️  PLATEAU DETECTED around Generation {gen}")
                print(f"   Improvement over next 50 gens: {improvement * 100:.1f}%")
                return gen

    print("\n✓ No plateau detected - still improving")
    return None


def analyze_latest_generation(gen_no=None):
    """✅ UPDATED: Deep dive into latest generation with new metrics."""

    if gen_no is None:
        # Find latest generation
        f = open("Generations/GenerationCount.txt", "r")
        gen_no = int(f.read())
        f.close()

    print(f"\n{'=' * 60}")
    print(f"DETAILED ANALYSIS: GENERATION {gen_no}")
    print(f"{'=' * 60}")

    # ✅ UPDATED: Predators with new fields
    pred_path = f"Generations/Generation{gen_no}/predatorTelemetry.csv"

    if os.path.exists(pred_path):
        with open(pred_path, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)

            catches = [float(r['catches']) for r in data]
            teamwork = [float(r['teamHuntScore']) for r in data]
            comm = [float(r['meanCommMagnitude']) for r in data]  # ✅ NEW
            timeAlive = [float(r['timeAlive']) for r in data]  # ✅ NEW
            energy = [float(r['finalEnergy']) for r in data]  # ✅ NEW

            print("\nPREDATORS:")
            print(f"  Population Size: {len(data)}")
            print(f"  Mean Catches:    {statistics.mean(catches):.2f}")
            print(f"  Best Catches:    {max(catches):.2f}")
            print(f"  Worst Catches:   {min(catches):.2f}")
            print(f"  Std Dev:         {statistics.stdev(catches) if len(catches) > 1 else 0:.2f}")

            print(f"\n  Mean Teamwork Score: {statistics.mean(teamwork):.2f}")
            print(f"  Best Teamwork Score: {max(teamwork):.2f}")

            if max(teamwork) < 5.0:
                print("  ⚠️  LOW TEAMWORK - Predators NOT coordinating effectively")
            elif max(teamwork) < 20.0:
                print("  ⚠️  MODERATE TEAMWORK - Some coordination but room to improve")
            else:
                print("  ✓ HIGH TEAMWORK - Good coordination")

            # ✅ NEW: Communication analysis
            print(f"\n  Mean Communication: {statistics.mean(comm):.3f}")
            print(f"  Best Communication: {max(comm):.3f}")

            if statistics.mean(comm) < 0.1:
                print("  ⚠️  LOW COMMUNICATION - Barely using communication channel")
            elif statistics.mean(comm) < 0.5:
                print("  ⚠️  MODERATE COMMUNICATION - Some signaling")
            else:
                print("  ✓ ACTIVE COMMUNICATION - Frequent signaling")

            # ✅ NEW: Survival analysis
            print(f"\n  Mean Survival Time: {statistics.mean(timeAlive):.2f}s / 30s")
            survival_rate = (statistics.mean(timeAlive) / 30.0) * 100
            print(f"  Survival Rate: {survival_rate:.1f}%")

            if survival_rate < 50:
                print("  ⚠️  HIGH DEATH RATE - Many predators dying")
            elif survival_rate < 80:
                print("  ⚠️  MODERATE DEATHS - Some predators dying")
            else:
                print("  ✓ GOOD SURVIVAL - Most predators survive")

            # ✅ NEW: Energy analysis
            print(f"\n  Mean Final Energy: {statistics.mean(energy):.2f}")
            print(f"  Best Final Energy: {max(energy):.2f}")
            print(f"  Worst Final Energy: {min(energy):.2f}")

            if statistics.mean(energy) < 20:
                print("  ⚠️  ENERGY CRITICAL - Agents barely surviving")
            elif statistics.mean(energy) < 50:
                print("  ⚠️  LOW ENERGY - Agents struggling with energy management")
            else:
                print("  ✓ GOOD ENERGY - Agents managing energy well")

    # ✅ UPDATED: Prey with new fields
    prey_path = f"Generations/Generation{gen_no}/preyTelemetry.csv"

    if os.path.exists(prey_path):
        with open(prey_path, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)

            timeAlive = [float(r['timeAlive']) for r in data]  # ✅ CHANGED
            grouping = [float(r['groupingScore']) for r in data]
            comm = [float(r['meanCommMagnitude']) for r in data]  # ✅ NEW
            energy = [float(r['finalEnergy']) for r in data]  # ✅ NEW

            print("\nPREY:")
            print(f"  Population Size: {len(data)}")
            print(f"  Mean Survival Time: {statistics.mean(timeAlive):.2f}s / 30s")

            # Calculate survival percentage
            survival_rate = (statistics.mean(timeAlive) / 30.0) * 100
            print(f"  Survival Rate: {survival_rate:.1f}%")

            print(f"\n  Mean Grouping Score: {statistics.mean(grouping):.2f}")
            print(f"  Best Grouping Score: {max(grouping):.2f}")

            if max(grouping) < 5.0:
                print("  ⚠️  LOW GROUPING - Prey NOT coordinating")
            elif max(grouping) < 20.0:
                print("  ⚠️  MODERATE GROUPING - Some coordination")
            else:
                print("  ✓ HIGH GROUPING - Strong coordination")

            # ✅ NEW: Communication analysis
            print(f"\n  Mean Communication: {statistics.mean(comm):.3f}")
            print(f"  Best Communication: {max(comm):.3f}")

            if statistics.mean(comm) < 0.1:
                print("  ⚠️  LOW COMMUNICATION - Barely using communication channel")
            elif statistics.mean(comm) < 0.5:
                print("  ⚠️  MODERATE COMMUNICATION - Some signaling")
            else:
                print("  ✓ ACTIVE COMMUNICATION - Frequent signaling")

            # ✅ NEW: Energy analysis
            print(f"\n  Mean Final Energy: {statistics.mean(energy):.2f}")
            print(f"  Best Final Energy: {max(energy):.2f}")
            print(f"  Worst Final Energy: {min(energy):.2f}")

            if statistics.mean(energy) < 20:
                print("  ⚠️  ENERGY CRITICAL - Prey barely surviving")
            elif statistics.mean(energy) < 50:
                print("  ⚠️  LOW ENERGY - Prey struggling with energy management")
            else:
                print("  ✓ GOOD ENERGY - Prey managing energy well")


if __name__ == "__main__":
    train(
        duration=45.0
    )