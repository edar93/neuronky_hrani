from NeuralNetwork import NeuralNetwork
from CartpoleGym import CartPoleGym
from MountainCarGym import MountainCarGym

class NeuralNetworkManager:
    
    def __init__(self, configuration):

        self.configuration = configuration
        if (configuration['gym'] == 'CartPole'):
            self.gym = CartPoleGym()
        elif (configuration['gym'] == 'MountainCar'):
            self.gym = MountainCarGym()
        else:
            raise Exception('incorect gym set')    

    def start(self):
        configuration = self.configuration
        neuralNetworkManager.createNeuralNetworkPool(
            configuration['inputs'],
            configuration['outputs'],
            configuration['layers'],
            configuration['firstGenerationNetworkCount']
        )

        neuralNetworkManager.startEvolution(
            configuration['rounds'],
            configuration['childs'],
            configuration['mutationCoeficient'],
            configuration['renderInterval'],
            configuration['gamesInOneRound']
        )

    def createNeuralNetworkPool(self, inputs, outpust, layers, count):
        self.networkPool = []
        self.generation = 0

        for i in range(count):
            newNetwork = NeuralNetwork(inputs, outpust)
            newNetwork.createRandomNetwork(layers)
            network = {
                "network": newNetwork,
                "score": None,
                "generation": self.generation
            }
            self.networkPool.append(network)

    def play(self, renderInterval = 1, gamesInOneRound = 1):
        for i in range(len(self.networkPool)):
            renderFirst = self.generation % renderInterval == 0 and i == 0

            returnValue = self.gym.play(self.networkPool[i]["network"], 1000, renderFirst)
            for j in range(gamesInOneRound - 1):
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
                #small mulation
                child = nextGen[i]['network'].clone()
                child.mutate(mutationCoeficient[0])
                nextGen.append({
                    "network": child,
                    "score": None,
                    "generation": self.generation
                })
                #big mulation
                child = nextGen[i]['network'].clone()
                child.mutate(mutationCoeficient[1])
                nextGen.append({
                    "network": child,
                    "score": None,
                    "generation": self.generation
                })

        self.networkPool = nextGen

    def startEvolution(self, maxGenerations, survivals, mutationCoeficient, renderInterval, gamesInOneRound):
        print('evolution started')
        for i in range(maxGenerations):
            self.play(renderInterval, gamesInOneRound)
            self.evolve(survivals, mutationCoeficient)   


        self.gym.close()

    def _printScoresReport(self, logTopResults = 4, oneLineReport = True):
        self.networkPool.sort(key = self._sortMethod, reverse=True)
        champinGeneration = self.networkPool[0]["generation"]
        generationText = "generation " + str(self.generation) + "."
        scoresText = "Scores was:"
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
        averageText = " Average score is " + str(average) + ". "
        championTime = self.generation - champinGeneration + 1
        championText = " Champion survived " + str(championTime) + " generations."
        

        if (not isFloatScore or oneLineReport):
            print(generationText + championText + averageText + scoresText)
        else :
            print(generationText + championText + averageText)
            print(scoresText)


    def _sortMethod(self, e):
        if (e['score'] == None):
            return 0

        return e['score']


configurationMountainCar = dict(
    gym = 'MountainCar',
    inputs = 2,
    outputs = 3,
    layers = [10, 6],
    firstGenerationNetworkCount = 250,
    rounds = 10000,
    childs = [3 ,2 ,2 , 1, 0, 0, 0],
    mutationCoeficient = [0.012, 0.003],
    renderInterval = 10,
    gamesInOneRound = 4
)

configurationCartPole = dict(
    gym = 'CartPole',
    inputs = 4,
    outputs = 2,
    layers = [8, 6],
    firstGenerationNetworkCount = 1000,
    rounds = 10000,
    childs = [3 ,2 ,2 , 1, 0, 0, 0],
    mutationCoeficient = [0.012, 0.003],
    renderInterval = 10,
    gamesInOneRound = 4
)

#neuralNetworkManager = NeuralNetworkManager(configurationMountainCar)
neuralNetworkManager = NeuralNetworkManager(configurationCartPole)
neuralNetworkManager.start()
