from dataclasses import dataclass, field


@dataclass
class SimulationConfig:
    arenaSize: float = 60  # Keep
    catchDistance: float = 0.7  # Was 0.4
    eatDistance: float = 0.7  # Was 1.0 (symmetric!)
    boundaryMargin: float = 1.0  # Keep
    boundaryForce: float = 50.0  # Keep
    visionRadius: float = 15.0  # Was 20.0
    communicationRadius: float = 12.0  # Was 22.5
    foodClusterNo: int = 4  # Was 5
    minClusterFood: int = 8  # Was 10
    maxClusterFood: int = 10  # Was 12
    preyEnergyValue: float = 50.0  # Was 60.0
    arenaFoodEnergyValue: float = 30.0  # Was 50.0
    predatorCatchSupportRadius: float = 3.0  # Keep
    predatorTeamHuntRadius: float = 3.0  # Keep
    minRequiredTeamMembers: int = 2  # Keep


@dataclass
class PreyConfig:
    arenaSize: float = 60.0
    baseSpeed: float = 15.0  # Keep
    eatDistance: float = 0.5  # Match SimConfig
    predatorRadius: float = 0.3  # Match Predator
    preyRadius: float = 0.2  # Keep
    boundaryMargin: float = 1.0
    boundaryForce: float = 50.0
    maxEnergy: float = 100.0
    energyDrainRate: float = 1.5  # Was 5.0 ← CRITICAL FIX
    sprintDrainRate: float = 3.0  # Was 8.5 ← CRITICAL FIX
    sprintMultiplier: float = 1.5  # Keep
    maxStomachCapacity: float = 90.0

    radius: float = 0.2  # Was 0.1 (matches preyRadius)
    colour: list = field(
        default_factory=lambda: [0.0, 0.0, 1.0, 1.0]
    )


@dataclass
class PredatorConfig:
    arenaSize: float = 60.0
    baseSpeed: float = 16.0  # Was 15.0 (slight advantage)
    catchDistance: float = 0.5  # Match SimConfig
    predatorRadius: float = 0.3
    preyRadius: float = 0.2
    boundaryMargin: float = 1.0
    boundaryForce: float = 50.0
    maxEnergy: float = 100.0
    energyDrainRate: float = 1.5  # Was 5.0 ← CRITICAL FIX
    sprintDrainRate: float = 3.0  # Was 8.5 ← CRITICAL FIX
    sprintMultiplier: float = 1.5
    maxStomachCapacity: float = 90.0

    radius: float = 0.3  # Was 0.1 (matches predatorRadius)
    colour: list = field(
        default_factory=lambda: [1.0, 0.0, 0.0, 1.0]
    )


@dataclass
class FoodConfig:
    energyValue: float = 30.0  # Was 50.0
    respawnTime: float = 8.0  # Was 10.0 (faster respawn)
    radius: float = 0.15
    readyColour: list = field(
        default_factory=lambda: [1.0, 0.713, 0.756, 1.0]
    )
    consumedColour: list = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.0]
    )