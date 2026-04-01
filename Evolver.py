import copy
import random
import torch
import numpy as np


class Evolver:
    def __init__(self, tournamentSize: int = 3, mutationRate: float = 0.15, sigma: float = 0.1):
        # REDUCED tournament size from 5 to 3 for less selection pressure
        self.tournamentSize = tournamentSize

        # Keep track of base parameters
        self.baseMutationRate = mutationRate
        self.currentMutationRate = mutationRate
        self.baseSigma = sigma
        self.currentSigma = sigma

        # Stagnation tracking
        self.previousBestFitness = -float("inf")
        self.stagnationCounter = 0

        # Diversity tracking
        self.diversityHistory = []
        self.fitnessVarianceHistory = []

    def calculateGenotypeDiversity(self, population: list) -> float:
        """Calculate average pairwise distance between genotypes"""
        if len(population) < 2:
            return 0.0

        genotypes = [ind["genotype"] for ind in population]
        distances = []

        # Sample 100 random pairs to avoid O(n²) complexity
        sample_size = min(100, len(population) * (len(population) - 1) // 2)
        for _ in range(sample_size):
            i, j = random.sample(range(len(population)), 2)
            dist = torch.dist(genotypes[i], genotypes[j]).item()
            distances.append(dist)

        return np.mean(distances) if distances else 0.0

    def produceNextGeneration(self, currentGeneration: list, kFittest=2) -> list:
        # INCREASED elites from 2 to 5 to preserve more genetic material
        nextGen = []
        populationSize = len(currentGeneration)

        # Calculate diversity metrics
        diversity = self.calculateGenotypeDiversity(currentGeneration)
        fitness_values = [ind["fitness"] for ind in currentGeneration]
        fitness_variance = np.var(fitness_values)

        self.diversityHistory.append(diversity)
        self.fitnessVarianceHistory.append(fitness_variance)

        sortedPop = sorted(currentGeneration, key=lambda x: x["fitness"], reverse=True)

        # Check for stagnation AND low diversity
        bestFitness = sortedPop[0]["fitness"]
        is_stagnant = self._checkStagnation(bestFitness)
        is_low_diversity = diversity < 5.0  # Threshold depends on your parameter scale

        if is_stagnant or is_low_diversity:
            # ADAPTIVE RESPONSE: Boost both mutation rate AND sigma
            self.currentMutationRate = min(self.baseMutationRate * 2.0, 0.6)
            self.currentSigma = min(self.baseSigma * 3.0, 0.5)
            print(f" ! Diversity crisis: diversity={diversity:.2f}, variance={fitness_variance:.2f}")
            print(f" ! Boosting mutation_rate={self.currentMutationRate:.3f}, sigma={self.currentSigma:.3f}")
        else:
            # Gradually return to base values
            self.currentMutationRate = max(self.currentMutationRate * 0.9, self.baseMutationRate)
            self.currentSigma = max(self.currentSigma * 0.9, self.baseSigma)

        # Preserve top performers (elitism)
        for i in range(0, kFittest):
            eliteGenotype = copy.deepcopy(sortedPop[i])
            eliteGenotype["generationNo"] += 1
            eliteGenotype["fitness"] = None
            nextGen.append(eliteGenotype)

        # EXPANDED parent pool from 70% to 90% for more diversity
        parentPoolSize = int(populationSize * 0.9)
        parentPool = sortedPop[:parentPoolSize]

        # Fill the rest with offspring
        while len(nextGen) < populationSize:
            parent1 = self._tournamentSelection(parentPool)
            parent2 = self._tournamentSelection([ind for ind in parentPool if ind != parent1])

            # Use blend crossover instead of uniform for exploration
            child = self._mutate(
                self._reproduce(
                    parent1["genotype"],
                    parent2["genotype"]
                )
            )

            nextGen.append(
                {
                    "generationNo": parent1.get("generationNo", 0) + 1,
                    "genotypeID": None,
                    "genotypeNN": None,
                    "genotype": child,
                    "fitness": None
                }
            )

        # DIVERSITY INJECTION: Replace worst 10% with random immigrants every 5 generations
        if len(self.diversityHistory) % 5 == 0 and diversity < 10.0:
            num_immigrants = max(1, populationSize // 10)
            print(f" ! Injecting {num_immigrants} random immigrants")
            for i in range(num_immigrants):
                # Replace from the end (worst performers that weren't selected as elites)
                idx = -(i + 1)
                if abs(idx) <= len(nextGen) - kFittest:  # Don't replace elites
                    nextGen[idx]["genotype"] = torch.randn_like(nextGen[idx]["genotype"])

        return nextGen

    def _tournamentSelection(self, pop: list):
        potentialParents = random.sample(pop, min(self.tournamentSize, len(pop)))
        return max(potentialParents, key=lambda individual: individual["fitness"])

    def _mutate(self, genotype: torch.Tensor) -> torch.Tensor:
        """Gaussian mutation with per-gene masking"""
        noise = torch.randn(genotype.shape) * self.currentSigma
        mutationMask = (torch.rand(genotype.shape) < self.currentMutationRate).float()
        return genotype + (noise * mutationMask)

    def _reproduce(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """Blend crossover for better exploration"""
        # Use blend crossover with alpha in [0.3, 0.7] range
        alpha = random.uniform(0.3, 0.7)
        return alpha * p1 + (1 - alpha) * p2

        # Alternative: Keep uniform crossover but add perturbation
        # mask = (torch.rand(p1.shape) > 0.5).float()
        # child = (mask * p1) + ((1 - mask) * p2)
        # # Add small random perturbation to introduce novelty
        # perturbation = torch.randn_like(child) * 0.01
        # return child + perturbation

    def _checkStagnation(self, currentBest: float, tolerance=0.01):
        """Check if fitness improvement has stalled"""
        # INCREASED tolerance from 0.001 to 0.01 - less sensitive
        if currentBest > self.previousBestFitness + tolerance:
            self.previousBestFitness = currentBest
            self.stagnationCounter = 0
            return False

        self.stagnationCounter += 1

        # REDUCED threshold from 10 to 5 generations - faster response
        if self.stagnationCounter >= 5:
            return True

        return False

    def getDiversityMetrics(self):
        """Return diversity history for plotting"""
        return {
            "diversity": self.diversityHistory,
            "fitness_variance": self.fitnessVarianceHistory
        }