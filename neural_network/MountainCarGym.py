import gym

class MountainCarGym:

    def __init__(self):
        self.env = gym.make('MountainCar-v0')

    def close(self):
        self.env.close()

    def play(self, network, maxRounds = 1000, render = False):
        env = self.env
        observation = env.reset()
        maxPosition = -1

        for round in range(maxRounds):
            if render:
                env.render()
            
            #action = env.action_space.sample()
            action = network.evalToGetMax(observation)
            observation, reward, done, info = env.step(action)
            maxPosition = maxPosition if maxPosition > observation[0] else observation[0]
            if done:
                break

        return maxPosition
        
