from NeuralNetwork import NeuralNetwork
from CartpoleGym import CartPoleGym

class NeuralNetworkManager:
    
    def __init__(self):
        self.cartPoleGym = CartPoleGym()

    def createNeuralNetworkPool(self, inputs, outpust, layers, count):
        self.networkPool = []
        self.generation = 0

        for i in range(count):
            newNetwork = NeuralNetwork(inputs, outpust)
            newNetwork.createRandomNetwork(layers)
            network = {
                "network": newNetwork,
                "score": None
            }
            self.networkPool.append(network)

    def play(self):
        for i in range(len(self.networkPool)):
            returnValue = self.cartPoleGym.play(self.networkPool[i]["network"], 1000)
            #todo loop
            returnValue += self.cartPoleGym.play(self.networkPool[i]["network"], 1000)
            self.networkPool[i]["score"] = returnValue

        self._printScoresReport()
        

    def evolve(self, survivals, mutationCoeficient):
        self.generation += 1
        nextGen = []
        self.networkPool.sort(key = self._sortMethod, reverse=True)

        # take ancestor to next generation
        for i in range(len(survivals)):
            # take ancestor to next generation
            nextGen.append(self.networkPool.pop(0))

        # create childs
        for i in range(len(survivals)):
            for j in range(survivals[i]):
                child = nextGen[i]['network'].clone()
                child.mutate(mutationCoeficient)
                nextGen.append({
                    "network": child,
                    "score": None
                })

        self.networkPool = nextGen

    def startEvolution(self, maxGenerations, survivals, mutationCoeficient):
        print('evolution started')
        for i in range(maxGenerations):
            self.play()
            self.evolve(survivals, mutationCoeficient)

            #if (i % 10 == 0):
            #   print('layers')
            #   firstNetwork = self.networkPool[0]['network']
            #   for j in range(len(firstNetwork.synaps)):
            #       print('layer', j)
            #       for k in firstNetwork.synaps[j]:
            #            print(k)            


        self.cartPoleGym.close()

    def _printScoresReport(self):
        self.networkPool.sort(key = self._sortMethod, reverse=True)
        generationText = "generation " + str(self.generation)
        scoresText = " scores was:"
        average = 0

        for i in range(len(self.networkPool)):
            if(self.networkPool[i]["score"] == None):
                print('missing score')
                return

            average += float(self.networkPool[i]["score"])
            scoresText += " " + str(int(self.networkPool[i]["score"])) + ","

        average /= len(self.networkPool)
        averageText = " had average score is " + str(average) + " "
        print(generationText + averageText+ scoresText)


    def _sortMethod(self, e):
        if (e['score'] == None):
            return 0

        return e['score']


neuralNetworkManager = NeuralNetworkManager()
neuralNetworkManager.createNeuralNetworkPool(4 ,2 ,[8] , 170)
neuralNetworkManager.startEvolution(1100, [5 ,4 ,3 , 2, 0, 0, 0], 0.0005)
