import numpy as np
from actor_critic import Agent
from utils import plot_learning_curve
from my_env import ENV
import dill



# https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/actor_critic/tensorflow2
env = ENV()
agent = Agent(alpha=1e-5, n_actions=env.n_action)
n_games = 500

def learning(agent,env,n_games):

    filename = 'cartpole_demo.png'
    figure = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
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
            with open('best_agent.pkl', 'wb') as f:
                dill.dump(agent, f)  # pickle no || dill
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, f'score {score}', f'avg_score {avg_score}')

    if not load_checkpoint:
        x = [i + 1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure)

if __name__ == '__main__':
    learning(agent, env, n_games)

