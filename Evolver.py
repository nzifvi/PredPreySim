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

    def produceNextGeneration(self, currentGeneration: list, kFittest=3) -> list:
        nextGen = []
        populationSize = len(currentGeneration)

        sortedPop = sorted(currentGeneration, key=lambda x: x["fitness"], reverse=True)

        # Preserve top performers (elitism)
        for i in range(0, kFittest):
            eliteGenotype = copy.deepcopy(sortedPop[i])
            eliteGenotype["generationNo"] += 1
            eliteGenotype["fitness"] = None
            nextGen.append(eliteGenotype)

        parentPoolSize = int(populationSize * 0.9)
        parentPool = sortedPop[:parentPoolSize]

        while len(nextGen) < populationSize:
            parent1 = self._tournamentSelection(parentPool)
            parent2 = self._tournamentSelection([ind for ind in parentPool if ind != parent1])

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
