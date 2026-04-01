import os
import torch
import torch.nn as nn

OBSERVATION_SIZE = 14
COMM_IN_SIZE = 2
INPUT_SIZE = OBSERVATION_SIZE + COMM_IN_SIZE

HIDDEN_STATES = 32
MOVE_OUT = 2
COMM_OUT = 2
OUTPUT_SIZE = MOVE_OUT * COMM_OUT

class NeuralNetwork(nn.Module):
    def __init__(self, generationNo = None, genotypeID = None, *, weights = None, biases = None, isPredator = True):
        super().__init__()

        self.isPredator = isPredator
        self.encoder = nn.Linear(INPUT_SIZE, HIDDEN_STATES)
        self.rnn = nn.RNNCell(HIDDEN_STATES, HIDDEN_STATES, nonlinearity = "tanh")
        self.head = nn.Linear(HIDDEN_STATES, OUTPUT_SIZE)

        if weights is not None and biases is not None:
            self._loadFromParameters(weights, biases)
        else:
            self._loadFromDisk(generationNo, genotypeID)

    def _loadFromParameters(self, weights, biases):
        encoderWeight, rnnWeightIh, rnnWeightHh, headWeight = weights
        encoderBias, rnnBiasIh, rnnBiasHh, headBias = biases

        with torch.no_grad():
            self.encoder.weight.copy_(encoderWeight.T)
            self.encoder.bias.copy_(encoderBias.flatten())

            self.rnn.weight_ih.copy_(rnnWeightIh.T)
            self.rnn.bias_ih.copy_(rnnBiasIh.flatten())

            self.head.weight.copy_(headWeight.T)
            self.head.bias.copy_(headBias.flatten())

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
            raise ValueError(f"! Expected observation shape [batch, {OBSERVATION_SIZE}, got {tuple(observation.shape)}")

        if hiddenState.dim() != 2 or hiddenState.shape[1] != HIDDEN_STATES:
            raise ValueError(f"! Expected hiddenState shape [batch, {HIDDEN_STATES}, got {tuple(hiddenState.shape)}")

        if commInput is None:
            commInput = torch.zeros(observation.shape[0], COMM_IN_SIZE, dtype=torch.float32)

        x = torch.cat(
            [observation, commInput], dim = 1
        )
        x = torch.tanh(
            self.encoder(x)
        )
        newHiddenState = self.rnn(x, hiddenState)
        output = self.head(newHiddenState)

        return output[:, :MOVE_OUT], output[:, MOVE_OUT:], newHiddenState

    @property
    def weights(self):
        return [
            self.encoder.weight.detach().T.clone(),
            self.rnn.weight_ih.detach().T.clone(),
            self.rnn.weight_hh.detach().T.clone(),
            self.head.weight.detach().T.clone()
        ]

    @property
    def biases(self):
        return [
            self.encoder.bias.detach().view(1, -1).clone(),
            self.rnn.bias_ih.detach().view(1, -1).clone(),
            self.rnn.bias_hh.detach().view(1, -1).clone(),
            self.head.bias.detach().view(1, -1).clone()
        ]