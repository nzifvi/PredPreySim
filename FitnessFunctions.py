CATCH_REWARD_WEIGHT = 100.0
PREDATOR_SURVIVAL_BONUS_WEIGHT = 2.0
PREY_SURVIVAL_BONUS_WEIGHT = 10.0
COMPLETION_BONUS_WEIGHT = 500.0

def calculatePredatorFitness(predatorTelemetry) -> float:
    catchReward = predatorTelemetry["catches"] * CATCH_REWARD_WEIGHT
    survivalBonus = predatorTelemetry["timeActive"] * PREDATOR_SURVIVAL_BONUS_WEIGHT

    fitness = catchReward + survivalBonus

    return max(0.0, fitness)

def calculatePreyFitness(preyTelemetry) -> float:
    survivalReward = preyTelemetry["timeAlive"] * PREY_SURVIVAL_BONUS_WEIGHT
    completionBonus = 0.0
    if preyTelemetry["alive"]:
        completionBonus = COMPLETION_BONUS_WEIGHT

    fitness = survivalReward + completionBonus

    return max(0.0, fitness)