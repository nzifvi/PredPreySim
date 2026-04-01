from dataclasses import dataclass

@dataclass
class SimulationConfig:
    arenaSize: float      = 20.0
    maxSpeed: float       = 3.0
    catchDistance: float  = 0.8
    predatorRadius: float = 0.4
    preyRadius: float     = 0.3
    boundaryMargin: float = 1.0
    boundaryForce: float  = 50.0

@dataclass
class PreyConfig:
    arenaSize: float      = 20.0
    maxSpeed: float       = 3.0
    catchDistance: float  = 0.8
    predatorRadius: float = 0.4
    preyRadius: float     = 0.3
    boundaryMargin: float = 1.0
    boundaryForce: float  = 50.0
    radius = 3.0
    colour = [1, 0, 1, 0]

@dataclass
class PredatorConfig:
    arenaSize: float      = 20.0
    maxSpeed: float       = 3.0
    catchDistance: float  = 0.8
    predatorRadius: float = 0.4
    preyRadius: float     = 0.3
    boundaryMargin: float = 1.0
    boundaryForce: float  = 50.0
    radius = 3.0
    colour = [0, 1, 0, 1]
