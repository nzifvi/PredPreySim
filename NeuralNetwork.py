import os
import torch
import torch.nn as nn
from dataclasses import dataclass, field

OBSERVATION_SIZE = 19
COMM_IN_SIZE     = 2
INPUT_SIZE       = OBSERVATION_SIZE + COMM_IN_SIZE

HIDDEN_STATES = 64
SHARED_SIZE   = 32

MOVE_OUT = 2
COMM_OUT = 2

@dataclass
class NeuralNetworkConfig:
    nnBlueprint = [
        {
            "name"         : "encoderLayer",
            "inputs"       : INPUT_SIZE,
            "outputs"      : HIDDEN_STATES,
            "totalWeights" : INPUT_SIZE * HIDDEN_STATES,
            "biases"       : HIDDEN_STATES
        },
        {
            "name"         : "rnnIHLayer",
            "inputs"       : HIDDEN_STATES,
            "outputs"      : HIDDEN_STATES,
            "totalWeights" : HIDDEN_STATES ** 2,
            "biases"       : HIDDEN_STATES
        },
        {
            "name"         : "rnnHHLayer",
            "inputs"       : HIDDEN_STATES,
            "outputs"      : HIDDEN_STATES,
            "totalWeights" : HIDDEN_STATES ** 2,
            "biases"       : HIDDEN_STATES
        },
        {
            "name"         : "sharedLayer",
            "inputs"       : HIDDEN_STATES,
            "outputs"      : SHARED_SIZE,
            "totalWeights" : HIDDEN_STATES * SHARED_SIZE,
            "biases"       : SHARED_SIZE
        },
        {
            "name"         : "movementHead",
            "inputs"       : SHARED_SIZE,
            "outputs"      : MOVE_OUT,
            "totalWeights" : SHARED_SIZE * MOVE_OUT,
            "biases"       : MOVE_OUT
        },
        {
            "name"         : "communicationHead",
            "inputs"       : SHARED_SIZE,
            "outputs"      : COMM_OUT,
            "totalWeights" : SHARED_SIZE * COMM_OUT,
            "biases"       : COMM_OUT
        }
    ]

class NeuralNetwork(nn.Module):
    def __init__(self, generationNo = None, genotypeID = None, *, weights = None, biases = None, isPredator = True):
        super().__init__()

        self.isPredator = isPredator

        self.encoder = nn.Linear(
            INPUT_SIZE,
            HIDDEN_STATES
        )
        self.rnn = nn.RNNCell(
            HIDDEN_STATES,
            HIDDEN_STATES,
            nonlinearity = "tanh"
        )
        self.shared = nn.Linear(
            HIDDEN_STATES,
            SHARED_SIZE
        )
        self.movementHead = nn.Linear(
            SHARED_SIZE,
            MOVE_OUT
        )
        self.communicationHead = nn.Linear(
            SHARED_SIZE,
            COMM_OUT
        )

        if weights is not None and biases is not None:
            self._loadFromParameters(weights, biases)
        else:
            self._loadFromDisk(generationNo, genotypeID)

    def _loadFromParameters(self, weights, biases):
        encoderWeight, rnnIHWeight, rnnHHWeight, sharedWeight, movementHeadWeight, commHeadWeight = weights
        encoderBias, rnnIHBias, rnnHHBias, sharedBias, movementHeadBias, commHeadBias = biases

        with torch.no_grad():
            self.encoder.weight.copy_(encoderWeight.T)
            self.encoder.bias.copy_(encoderBias.flatten())

            self.rnn.weight_ih.copy_(rnnIHWeight.T)
            self.rnn.bias_ih.copy_(rnnIHBias.flatten())

            self.rnn.weight_hh.copy_(rnnHHWeight.T)
            self.rnn.bias_hh.copy_(rnnHHBias.flatten())

            self.shared.weight.copy_(sharedWeight.T)
            self.shared.bias.copy_(sharedBias.flatten())

            self.movementHead.weight.copy_(movementHeadWeight.T)
            self.movementHead.bias.copy_(movementHeadBias.flatten())

            self.communicationHead.weight.copy_(commHeadWeight.T)
            self.communicationHead.bias.copy_(commHeadBias.flatten())

    def _loadFromDisk(self, generationNo, genotypeID):
        speciesType = "Predators" if self.isPredator else "Prey"

        basePath = os.path.join(
            "Generations", f"Generation{generationNo}", speciesType, "NeuralNetworks", f"Genotype{genotypeID}"
        )

        weightsPath = os.path.join(basePath, "Weights", "weights.pt")
        biasesPath = os.path.join(basePath, "Biases", "biases.pt")

        if not os.path.exists(weightsPath):
            raise FileNotFoundError(f"! cannot locate weights file at {weightsPath}")
        if not os.path.exists(biasesPath):
            raise FileNotFoundError(f"! cannot locate biases file at {biasesPath}")

        weights = torch.load(weightsPath, map_location="cpu")
        biases = torch.load(biasesPath, map_location="cpu")

        self._loadFromParameters(weights, biases)

    def initHidden(self) -> torch.Tensor:
        return torch.zeros(1, HIDDEN_STATES, dtype=torch.float32)

    def forward(self, observation, hiddenState, commInput=None):
        if observation.dim() != 2 or observation.shape[1] != OBSERVATION_SIZE:
            raise ValueError(f"! Observation mismatch. Expected [batch, {OBSERVATION_SIZE}], got {tuple(observation.shape)}")
        if hiddenState.dim() != 2 or hiddenState.shape[1] != HIDDEN_STATES:
            raise ValueError(f"! hiddenState mismatch. Expected [batch, {HIDDEN_STATES}], got {tuple(hiddenState.shape)}")

        if commInput is None:
            commInput = torch.zeros(
                observation.shape[0],
                COMM_IN_SIZE,
                dtype=torch.float32
            )

        x = torch.cat(
            [observation, commInput],
            dim = 1
        )
        x = torch.tanh(
            self.encoder(x)
        )

        newHiddenState = self.rnn(
            x, hiddenState
        )
        sharedFeatures = torch.tanh(
            self.shared(newHiddenState)
        )

        movement      = self.movementHead(sharedFeatures)
        communication = self.communicationHead(sharedFeatures)

        return movement, communication, newHiddenState

    @property
    def weights(self):
        return [
            self.encoder.weight.detach().T.clone(),
            self.rnn.weight_ih.detach().T.clone(),
            self.rnn.weight_hh.detach().T.clone(),
            self.shared.weight.detach().T.clone(),
            self.movementHead.weight.detach().T.clone(),
            self.communicationHead.weight.detach().T.clone(),
        ]

    @property
    def biases(self):
        return [
            self.encoder.bias.detach().view(1, -1).clone(),
            self.rnn.bias_ih.detach().view(1, -1).clone(),
            self.rnn.bias_hh.detach().view(1, -1).clone(),
            self.shared.bias.detach().view(1, -1).clone(),
            self.movementHead.bias.detach().view(1, -1).clone(),
            self.communicationHead.bias.detach().view(1, -1).clone(),
        ]