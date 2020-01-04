# generate random floating point values
from random import random
from random import randrange

import math

#[layer] layer = [neuronSynapses] neuronSynapses = [synapsValues]
class NeuralNetwork:

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.synaps = []

    #initiate self.synaps , které je pole polí pro každou vrstvu, ve které jsou vahy synapsí postupně pro každý neuron 
    def createRandomNetwork(self, neuronsPerLayer):
        self.neuronsPerLayer = neuronsPerLayer
        self.layers = [self.inputs] + neuronsPerLayer + [self.outputs]

        for layer in range(len(self.layers) -1) :
            self.synaps.append(self._generateSynapsArray(self.layers[layer], self.layers[layer + 1]))
            
    def evaluateInputs(self, inputs): 
        self.layerValues = [inputs]

        for layer in range(len(self.synaps)):
            currentSynapsLayer = self.synaps[layer]
            self.layerValues.append([0] * len(currentSynapsLayer[0]))

            #for all synapse values in layer
            for neuronSynapses in range(len(currentSynapsLayer)):
                #make synapses value sum (synaps value * value from previous layer)
                for neuronSynapse in range(len(self.synaps[layer][neuronSynapses])):
                    self.layerValues[layer+1][neuronSynapse] += self.synaps[layer][neuronSynapses][neuronSynapse] * self.layerValues[layer][neuronSynapses]

            #multipty layer value by previous layer values and use activation function
            for layerValue in range(len(self.layerValues[layer + 1])):
                self.layerValues[layer + 1][layerValue] = self.sigmoid(self.layerValues[layer + 1][layerValue])

        return self.layerValues[len(self.layerValues) - 1]

    def clone(self):
        networkClone = NeuralNetwork(self.inputs, self.outputs)
        networkClone.neuronsPerLayer = self.neuronsPerLayer.copy()
        networkClone.layers = self.layers.copy()

        networkClone.synaps = self.synaps.copy()
        for i in range(len(self.synaps)):
            networkClone.synaps[i] = self.synaps[i].copy()

            for j in range(len(self.synaps[i])):
                networkClone.synaps[i][j] = self.synaps[i][j].copy()

        return networkClone    

    def mutate(self, koeficient, mutaleLayerOnlyOneLayer = False):
        if (not mutaleLayerOnlyOneLayer):
            for i in range(len(self.synaps)):
                for j in range(len(self.synaps[i])):
                    for k in range(len(self.synaps[i][j])):
                        self.synaps[i][j][k] = self.mutateValue(self.synaps[i][j][k], koeficient)

        else:
            #changeLayer replace i
            changeLayer = randrange(0, len(self.synaps))
            for j in range(len(self.synaps[changeLayer])):
                for k in range(len(self.synaps[changeLayer][j])):
                    self.synaps[changeLayer][j][k] = self.mutateValue(self.synaps[changeLayer][j][k], koeficient)


    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def mutateValue(self, value, koeficient):
        newValue = value + ((random() * koeficient * 2) - koeficient)

        if newValue < 0:
            return 0
        elif newValue > 1:
            return 1
        else:
            return newValue

    def evalToGetMax(self, inputs):
        result = self.evaluateInputs(inputs)
        maxValue = max(result)
        
        for i in range(len(result)):
            if result[i] == maxValue:
                return i


    def _generateSynapsArray(self, currentLayerNeurons, nextLayerNeurons):
        synaps = []
        for i in range(currentLayerNeurons):
            synaps.append(self._generateRandomSynapsValues(nextLayerNeurons))

        return synaps  

    def _generateRandomSynapsValues(self, synapsCount):
        synaps = []
        for i in range(synapsCount):
            synaps.append(random())

        return synaps
