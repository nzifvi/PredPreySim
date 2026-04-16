# predator fitness function term weights
CATCH_REWARD_WEIGHT = 150.0
TEAM_HUNT_BONUS_WEIGHT = 85.0
PREDATOR_PREY_PRESSURE_WEIGHT = 75.0

# prey fitness function term weights
GROUPING_BONUS_WEIGHT = 1.5
COMPLETION_BONUS_WEIGHT = 250
PREY_SURVIVAL_BONUS_WEIGHT = 10.0
ESCAPE_BONUS_WEIGHT = 25.0

def calculatePredatorFitnessBreakdown(predatorTelemetry) -> dict:
    catches = predatorTelemetry["catches"]
    teamHuntScore = predatorTelemetry.get("teamHuntScore", 0.0)
    meanNearestPreyDistance = predatorTelemetry.get("meanNearestPreyDistance", 0.0)

    catchReward = catches * CATCH_REWARD_WEIGHT
    teamHuntBonus = teamHuntScore * TEAM_HUNT_BONUS_WEIGHT

    preyPressureBonus = 0.0
    if meanNearestPreyDistance is not None:
        preyPressureBonus = (1.0 / (1.0 + meanNearestPreyDistance)) * PREDATOR_PREY_PRESSURE_WEIGHT

    totalFitness = catchReward + teamHuntBonus + preyPressureBonus
    totalFitness = max(0.0, totalFitness)
    return {
        "catches" : catches,
        "catchReward" : catchReward,
        "teamHuntBonus" : teamHuntBonus,
        "preyPressureBonus" : preyPressureBonus,
        "totalFitness" : totalFitness
    }

def calculatePredatorFitness(predatorTelemetry) -> float:
    return calculatePredatorFitnessBreakdown(predatorTelemetry)["totalFitness"]

def calculatePreyFitness(preyTelemetry) -> float:
    survivalReward = preyTelemetry["timeAlive"] * PREY_SURVIVAL_BONUS_WEIGHT
    completionBonus = 0.0
    if preyTelemetry["alive"]:
        completionBonus = COMPLETION_BONUS_WEIGHT
    groupingBonus = preyTelemetry.get("groupingScore", 0.0) * GROUPING_BONUS_WEIGHT

    meanNearestPredatorDistance = preyTelemetry.get("meanNearestPredatorDistance", 0.0)
    escapeBonus = 0.0
    if meanNearestPredatorDistance is not None:
        escapeBonus = (meanNearestPredatorDistance / 4.0) * ESCAPE_BONUS_WEIGHT

    fitness = survivalReward + completionBonus + groupingBonus + escapeBonus

    return max(0.0, fitness)