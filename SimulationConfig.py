from dataclasses import dataclass, field

@dataclass
class SimulationConfig:
    arenaSize            : float = 60.0
    catchDistance        : float = 0.4
    predatorRadius       : float = 0.4
    preyRadius           : float = 0.3
    boundaryMargin       : float  = 1.0
    boundaryForce        : float = 50.0
    visionRadius         : float = 4.0
    communicationRadius  : float = 7.5
    maxFoodAmount        : float = 64
    minFoodAmount        : float = 16
    preyEnergyValue      : float = 40.0
    arenaFoodEnergyValue : float = 30.0

@dataclass
class PreyConfig:
    arenaSize          : float = 60.0
    baseSpeed          : float = 4.2
    catchDistance      : float = 0.4
    predatorRadius     : float = 0.2
    preyRadius         : float = 0.1
    boundaryMargin     : float = 1.0
    boundaryForce      : float = 50.0
    maxEnergy          : float = 100
    energyDrainRate    : float = 2.0
    sprintDrainRate    : float = 3.0
    sprintMultiplier   : float = 1.3
    maxStomachCapacity : float = 50.0

    radius : float = 0.1
    colour : list  = field(
        default_factory=lambda:[1.0, 0.0, 1.0, 1.0]
    )

@dataclass
class PredatorConfig:
    arenaSize          : float = 60.0
    baseSpeed          : float = 3.0
    catchDistance      : float = 0.4
    predatorRadius     : float = 0.2
    preyRadius         : float = 0.1
    boundaryMargin     : float = 1.0
    boundaryForce      : float = 50.0
    maxEnergy          : float  = 100.0
    energyDrainRate    : float = 2.0
    sprintDrainRate    : float = 3.0
    sprintMultiplier   : float = 1.5
    maxStomachCapacity : float = 90.0

    radius : float = 0.2
    colour : list  = field(
        default_factory=lambda: [0.0, 1.0, 0.0, 1.0]
    )