from dataclasses import dataclass, field

@dataclass
class SimulationConfig:
    arenaSize            : float = 85.0
    catchDistance        : float = 0.4
    eatDistance          : float = 1.0
    boundaryMargin       : float  = 1.0
    boundaryForce        : float = 50.0
    visionRadius         : float = 20.0
    communicationRadius  : float = 22.5
    foodClusterNo        : int = 5
    minClusterFood       : int = 10
    maxClusterFood       : int = 12
    preyEnergyValue      : float = 60.0
    arenaFoodEnergyValue : float = 50.0

@dataclass
class PreyConfig:
    arenaSize          : float = 85.0
    baseSpeed          : float = 4.2
    eatDistance        : float = 1.0
    predatorRadius     : float = 0.3
    preyRadius         : float = 0.3
    boundaryMargin     : float = 1.0
    boundaryForce      : float = 50.0
    maxEnergy          : float = 100
    energyDrainRate    : float = 1.5
    sprintDrainRate    : float = 2.0
    sprintMultiplier   : float = 1.4
    maxStomachCapacity : float = 90.0

    radius : float = 0.1
    colour : list  = field(
        default_factory=lambda:[0.0, 0.0, 1.0, 1.0]
    )

@dataclass
class PredatorConfig:
    arenaSize          : float = 85.0
    baseSpeed          : float = 3.0
    catchDistance      : float = 0.3
    predatorRadius     : float = 0.4
    preyRadius         : float = 0.3
    boundaryMargin     : float = 1.0
    boundaryForce      : float = 50.0
    maxEnergy          : float = 100.0
    energyDrainRate    : float = 1.5
    sprintDrainRate    : float = 2.0
    sprintMultiplier   : float = 1.4
    maxStomachCapacity : float = 90.0

    radius : float = 0.1
    colour : list  = field(
        default_factory=lambda: [1.0, 0.0, 0.0, 1.0]
    )