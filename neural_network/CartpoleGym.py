import gym

class CartPoleGym:

    def __init__(self):
        self.env = gym.make('CartPole-v1')

    def close(self):
        self.env.close()

    def play(self, network, maxRounds = 1000, logActions = False):
        #env = gym.make('CartPole-v1')
        env = self.env
        observation = env.reset()
        totalReward = 0
        actionsLog = 'actions on 1st in generation: '

        for round in range(maxRounds):
            env.render()
            #action = env.action_space.sample()
            action = network.evalToGetMax(observation)
            observation, reward, done, info = env.step(action)
            actionsLog += str(action) + ', '
            totalReward += reward
            
            if done:
                break
                

        if (logActions):
            print(actionsLog)

        #env.close()
        return totalReward
        
        

        
       
#cartPoleGym = CartPoleGym
#cartPoleGym.play(None, 1000)
#cartPoleGym.play(None, 1000)
#cartPoleGym.play(None, 1000)

    #print(env.action_space)
    #print(env.observation_space)
