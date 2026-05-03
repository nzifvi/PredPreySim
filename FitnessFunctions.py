# predator fitness function term weights
CATCH_REWARD_WEIGHT            = 150
TEAM_HUNT_BONUS_WEIGHT         = 300.0
PREDATOR_COMM_BONUS_WEIGHT     = 1.0
PREDATOR_ENERGY_BONUS_WEIGHT   = 2.0
PREDATOR_SURVIVAL_BONUS_WEIGHT = 10.0
PREDATOR_STARVATION_PENALTY    = -300.0

# prey fitness function term weights
FOOD_EATEN_WEIGHT          = 100.0
GROUPING_BONUS_WEIGHT      = 30.0
PREY_SURVIVAL_BONUS_WEIGHT = 2.0
PREY_COMM_BONUS_WEIGHT     = 1.0
PREY_ENERGY_BONUS_WEIGHT   = 15.0
PREY_STARVATION_PENALTY    = -150.0

def calculatePredatorFitnessBreakdown(predatorTelemetry) -> dict:
    catches = predatorTelemetry.get("catches", 0)
    teamHuntScore = predatorTelemetry.get("teamHuntScore", 0.0)
    meanCommMagnitude = predatorTelemetry.get("meanCommMagnitude", 0.0)
    timeAlive = predatorTelemetry.get("timeAlive", 0.0)
    finalEnergy = predatorTelemetry.get("finalEnergy", 0.0)

    catchReward = catches * CATCH_REWARD_WEIGHT
    teamHuntBonus = teamHuntScore * TEAM_HUNT_BONUS_WEIGHT
    commBonus = meanCommMagnitude * PREDATOR_COMM_BONUS_WEIGHT

    # Survival: reward per second alive
    survivalBonus = timeAlive * PREDATOR_SURVIVAL_BONUS_WEIGHT

    # Energy: reward for ending with high energy
    energyBonus = (finalEnergy / 100.0) * PREDATOR_ENERGY_BONUS_WEIGHT

    # Starvation: penalty if very low energy
    starvationPenalty = 0.0
    if finalEnergy < 5.0:  # Consider < 5 energy as "starved"
        starvationPenalty = PREDATOR_STARVATION_PENALTY

    totalFitness = (
            catchReward +
            teamHuntBonus +
            commBonus +
            survivalBonus +
            energyBonus +
            starvationPenalty
    )

    totalFitness = max(0.0, totalFitness)

    return {
        "catches": catches,
        "catchReward": catchReward,
        "teamHuntBonus": teamHuntBonus,
        "commBonus": commBonus,
        "survivalBonus": survivalBonus,
        "energyBonus": energyBonus,
        "starvationPenalty": starvationPenalty,
        "totalFitness": totalFitness,
    }

def calculatePredatorFitness(predatorTelemetry) -> float:
    return calculatePredatorFitnessBreakdown(predatorTelemetry)["totalFitness"]

def calculatePreyFitnessBreakdown(preyTelemetry) -> dict:
    foodEaten = preyTelemetry.get("foodEaten", 0)
    timeAlive = preyTelemetry.get("timeAlive", 0.0)
    groupingScore = preyTelemetry.get("groupingScore", 0.0)
    meanCommMagnitude = preyTelemetry.get("meanCommMagnitude", 0.0)
    finalEnergy = preyTelemetry.get("finalEnergy", 0.0)

    eatingReward = foodEaten * FOOD_EATEN_WEIGHT

    # Survival: per-second reward (automatically higher if survived full 30s)
    survivalReward = timeAlive * PREY_SURVIVAL_BONUS_WEIGHT

    groupingBonus = groupingScore * GROUPING_BONUS_WEIGHT
    commBonus = meanCommMagnitude * PREY_COMM_BONUS_WEIGHT

    # Energy: reward for ending with high energy
    energyBonus = (finalEnergy / 100.0) * PREY_ENERGY_BONUS_WEIGHT

    # Starvation: penalty if very low energy
    starvationPenalty = 0.0
    if finalEnergy < 5.0:  # Consider < 5 energy as "starved"
        starvationPenalty = PREY_STARVATION_PENALTY

    totalFitness = (
            eatingReward +
            survivalReward +
            groupingBonus +
            commBonus +
            energyBonus +
            starvationPenalty
    )

    totalFitness = max(0.0, totalFitness)

    return {
        "eatingReward" : eatingReward,
        "timeAlive": timeAlive,
        "survivalReward": survivalReward,
        "groupingBonus": groupingBonus,
        "commBonus": commBonus,
        "energyBonus": energyBonus,
        "starvationPenalty": starvationPenalty,
        "totalFitness": totalFitness,
    }

def calculatePreyFitness(preyTelemetry) -> float:
    return calculatePreyFitnessBreakdown(preyTelemetry)["totalFitness"]

