from NeuralNetwork import NeuralNetwork
from CartpoleGym import CartPoleGym
from MountainCarGym import MountainCarGym

class NeuralNetworkManager:
    
    def __init__(self, game):
        if (game == 'CartPole'):
            self.gym = CartPoleGym()
        elif (game == 'MountainCar'):
            self.gym = MountainCarGym()
        else:
            raise Exception('incorect gym set')    


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

    def play(self, renderInterval = 1):
        for i in range(len(self.networkPool)):
            renderFirst = self.generation % renderInterval == 0 and i == 0

            returnValue = self.gym.play(self.networkPool[i]["network"], 1000, renderFirst)
            #todo loop
            returnValue += self.gym.play(self.networkPool[i]["network"], 1000)
            returnValue += self.gym.play(self.networkPool[i]["network"], 1000)
            returnValue += self.gym.play(self.networkPool[i]["network"], 1000)
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

    def startEvolution(self, maxGenerations, survivals, mutationCoeficient, renderInterval):
        print('evolution started')
        for i in range(maxGenerations):
            self.play(renderInterval)
            self.evolve(survivals, mutationCoeficient)

            #if (i % 10 == 0):
            #   print('layers')
            #   firstNetwork = self.networkPool[0]['network']
            #   for j in range(len(firstNetwork.synaps)):
            #       print('layer', j)
            #       for k in firstNetwork.synaps[j]:
            #            print(k)            


        self.gym.close()

    def _printScoresReport(self, logTopResults = 4, oneLineReport = True):
        self.networkPool.sort(key = self._sortMethod, reverse=True)
        generationText = "generation " + str(self.generation)
        scoresText = " scores was:"
        average = 0

        for i in range(len(self.networkPool)):
            score = self.networkPool[i]["score"]
            isFloatScore = score % 1 != 0
            if(score == None):
                print('missing score')
                return

            average += float(score)
            if i < logTopResults:
                scoresText += " " + str(float(score) if isFloatScore else int(score)) + ","

        average /= len(self.networkPool)
        averageText = " had average score is " + str(average) + " "

        if (not isFloatScore or oneLineReport):
            print(generationText + averageText + scoresText)
        else :
            print(generationText + averageText)
            print(scoresText)


    def _sortMethod(self, e):
        if (e['score'] == None):
            return 0

        return e['score']


#neuralNetworkManager = NeuralNetworkManager('MountainCar')
#neuralNetworkManager.createNeuralNetworkPool(2 ,3 ,[10, 6] , 250)
#neuralNetworkManager.startEvolution(1100, [5, 2, 1, 0, 0], 0.0005)

neuralNetworkManager = NeuralNetworkManager('CartPole')
neuralNetworkManager.createNeuralNetworkPool(4 ,2 , [8] , 170)
neuralNetworkManager.startEvolution(1100, [5 ,4 ,3 , 2, 0, 0, 0], 0.002, 10)
