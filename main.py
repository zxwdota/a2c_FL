import numpy as np
from actor_critic import Agent
from utils import plot_learning_curve
# from easy_env import ENV
from my_env import ENV

import threading

# https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/actor_critic/tensorflow2
env = ENV()
agent = Agent(alpha=1e-5, n_actions=env.n_action)
n_games = 50

def learning(agent,env,n_games,score_history):



    best_score = env.reward_range[0]

    load_checkpoint = False

    # agent.actor_critic.build(input_shape=(None, 31))
    # agent.load_models()
    # agent.actor_critic.summary()
    if load_checkpoint:
        agent.actor_critic.build(input_shape=(None, 31))
        agent.load_models()

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        time = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action, time)
            score += reward
            if not load_checkpoint:
                agent.learn(observation, reward, observation_, done)
            observation = observation_
            time += 1
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            # with open('best_agent.pkl', 'wb') as f:
            #     dill.dump(agent, f)  # pickle no || dill
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, f'score {score}', f'avg_score {avg_score}')



def init():
    a = Agent(alpha=1e-5, n_actions=env.n_action)
    b = Agent(alpha=1e-5, n_actions=env.n_action)
    c = Agent(alpha=1e-5, n_actions=env.n_action)
    return a,b,c

def avg(a,b,c):
    avg_weights = []
    for wa,wb,wc in zip(a.actor_critic.get_weights(), b.actor_critic.get_weights(),c.actor_critic.get_weights()):
        avg_weights.append((wa + wb + wc) / 3)
    return avg_weights


if __name__ == '__main__':
    a,b,c = init()
    a_score_history = []
    b_score_history = []
    c_score_history = []
    episode = 10
    filename = 'cartpole_demo.png'
    figure = 'plots/' + filename

    # for i in range(episode):
    #     learning(a, env, n_games,a_score_history)
    #     learning(b, env, n_games,b_score_history)
    #     learning(c, env, n_games,c_score_history)
    #     avg_weights = avg(a,b,c)
    #     a.actor_critic.set_weights(avg_weights)
    #     b.actor_critic.set_weights(avg_weights)
    #     c.actor_critic.set_weights(avg_weights)
    x = [i + 1 for i in range(episode*n_games)]
    # plot_learning_curve(x, a_score_history, 'a',figure)
    # plot_learning_curve(x, b_score_history, 'b',figure)
    # plot_learning_curve(x, c_score_history, 'c',figure)

    d_score_history = []
    d = Agent(alpha=1e-5, n_actions=env.n_action)
    nn_games = 500
    learning(d, env, nn_games, d_score_history)
    plot_learning_curve(x, d_score_history, 'd',figure)
