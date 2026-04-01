import torch
import os
import pandas

class NeuralNetwork:
    def __init__(self, isPredator:bool, generationNo:int = None, genotypeID:int = None, weights:list = None, biases:list = None):
        self.weights = []
        self.biases = []

        self.isPredator = isPredator

        self.inputNeurons = 14
        self.outputNeurons = 2
        self.hiddenLayers = [16]
        self.activationFunctions = ["ReLU", "tanh"]

        if generationNo is not None and genotypeID is not None:
            self.generationNo = generationNo
            self.genotypeID = genotypeID
            self.loadWeights()
            self.loadBiases()
        elif generationNo is None and genotypeID is None and weights is not None and biases is not None:
            self.weights = weights
            self.biases  = biases
        else:
            raise RuntimeError("! Error: unknown issue in NeuralNetwork constructor")


    def loadWeights(self):
        direcPath = "Generations/Generation" + str(self.generationNo)
        if self.isPredator:
            direcPath += "/Predators"
        else:
            direcPath += "/Prey"
        direcPath += "/NeuralNetworks/Genotype" + str(self.genotypeID) + "/Weights"
        if os.path.exists(direcPath):
            self.weights = torch.load(os.path.join(direcPath, "weights.pt"))
        else:
            raise FileNotFoundError(f"! Binary weights missing: {direcPath}")


    def loadBiases(self):
        direcPath = "Generations/Generation" + str(self.generationNo)
        if self.isPredator:
            direcPath += "/Predators"
        else:
            direcPath += "/Prey"
        direcPath += "/NeuralNetworks/Genotype" + str(self.genotypeID) + "/Biases"
        if os.path.exists(direcPath):
            self.biases = torch.load(os.path.join(direcPath, "biases.pt"))
        else:
            raise FileNotFoundError(f"! Binary biases missing: {direcPath}")

    @torch.no_grad()
    def inference(self, input):
        if input is None or input.shape != torch.Size([1, self.inputNeurons]):
            raise ValueError(f"  ! Input to NN has incorrect dimensions")

        functions = {
            "tanh" : torch.tanh,
            "ReLU" : torch.relu,
            "sigmoid" : torch.sigmoid
        }

        output = input
        for i in range(0, len(self.weights)):
            funcName = self.activationFunctions[i]
            output = functions[funcName](
                torch.mm(output, self.weights[i]) + self.biases[i]
            )
        return output