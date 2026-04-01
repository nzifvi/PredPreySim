"""
Fitness functions for predator-prey co-evolution
"""


def calculatePredatorFitness(telemetry, predatorIndex):
    """
    Calculate fitness for a single predator

    Rewards:
    - Catching prey (primary goal)
    - Time spent active/hunting
    - Proximity to prey (encourages chasing even if no catch)

    Args:
        telemetry: Dict from simulation
        predatorIndex: Which predator to evaluate (0-indexed)

    Returns:
        float: Fitness score
    """
    catches = telemetry['predator_catches'][predatorIndex]
    timeActive = telemetry['predator_time_active'][predatorIndex]

    # Base fitness from catches
    catchReward = catches * 100.0

    # Survival bonus (stayed in sim)
    survivalBonus = timeActive * 2.0

    # Total fitness
    fitness = catchReward + survivalBonus

    return max(0.0, fitness)


def calculatePreyFitness(telemetry, preyIndex):
    """
    Calculate fitness for a single prey

    Rewards:
    - Survival time (primary goal)
    - Bonus for surviving entire simulation

    Args:
        telemetry: Dict from simulation
        preyIndex: Which prey to evaluate (0-indexed)

    Returns:
        float: Fitness score
    """
    timeAlive = telemetry['prey_time_alive'][preyIndex]
    survived = telemetry['prey_alive'][preyIndex]

    # Base fitness from survival time
    survivalReward = timeAlive * 10.0

    # Big bonus for making it to the end
    completionBonus = 500.0 if survived else 0.0

    # Total fitness
    fitness = survivalReward + completionBonus

    return max(0.0, fitness)


def calculatePredatorFitnessAdvanced(telemetry, predatorIndex):
    """
    Advanced predator fitness with more nuanced rewards

    Use this if basic fitness leads to boring behavior
    """
    catches = telemetry['predator_catches'][predatorIndex]
    timeActive = telemetry['predator_time_active'][predatorIndex]

    # Diminishing returns on catches (prevent camping)
    if catches == 0:
        catchReward = 0.0
    elif catches == 1:
        catchReward = 150.0
    elif catches == 2:
        catchReward = 250.0
    else:
        catchReward = 250.0 + (catches - 2) * 50.0

    # Efficiency bonus (catches per unit time)
    efficiency = catches / (timeActive + 0.1)
    efficiencyBonus = efficiency * 100.0

    # Survival bonus
    survivalBonus = timeActive * 1.0

    fitness = catchReward + efficiencyBonus + survivalBonus

    return max(0.0, fitness)


def calculatePreyFitnessAdvanced(telemetry, preyIndex):
    """
    Advanced prey fitness with more nuanced rewards

    Use this if basic fitness leads to boring behavior
    """
    timeAlive = telemetry['prey_time_alive'][preyIndex]
    survived = telemetry['prey_alive'][preyIndex]

    # Exponential reward for longer survival
    survivalReward = (timeAlive ** 1.5) * 5.0

    # Massive bonus for full survival
    completionBonus = 1000.0 if survived else 0.0

    fitness = survivalReward + completionBonus

    return max(0.0, fitness)