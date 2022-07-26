import argparse
import random
import numpy as np
from QAgent import QAgent
from DeepQAgent import DeepQAgent
from environment import SnakeGame
import pickle
import torch

def train(agent_name, episodes, epsilon, epsilon_min, epsilon_decay, discount, lr, max_steps, render, render_every, save, path_dir):
    if agent_name == "QAgent":
        agent = QAgent(12, 4)
    if agent_name == "DeepQAgent":
        agent =  DeepQAgent(87, 4)
    env = SnakeGame()
    for episode in range(episodes):
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        steps=0
        env.reset()
        a = random.choice([0,1,2,3])
        observation, r, done = env.step(a)
        s = agent.get_state(observation)
        while not env.game_over and steps < max_steps:
            a = agent.choose_action(s, epsilon)
            # Now I have s, a
            observation, r, done = env.step(a)
            steps += 1
            # Now I have s, a, r
            s_ = agent.get_state(observation)

            # Now I have s, a, r, s'
            if agent.name == "QAgent":
                agent.Q[s][a] = agent.Q[s][a] + lr * (r + discount * np.max(agent.Q[s_]) - agent.Q[s][a])
            else:
                agent.learn(s, a, r, s_, done)
            if render and episode % render_every == 0:
                env.render()
                env.clock.tick(24)
            s = s_
        print(f"EPISODE {episode}   SCORE {env.score}   EPSILON {epsilon:.4f}")
        if agent_name == "DeepQAgent":
            if episode % agent.replace_every == 0:
                print(f"UPDATING TARGET NETWORK")
                agent.update_target_net()
    if agent_name == "QAgent":
        with open(path_dir +"/Q.pkl", "wb") as f:
            pickle.dump(agent.Q, f)
    if agent_name == "DeepQAgent":
        torch.save(agent.policy_net.state_dict(), path_dir+"/policy_net_state_dict.torch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, help="Agent to train, choose betweeen QAgent and DeepQAgent", default="QAgent")
    parser.add_argument("--episodes", type=int, help="Number of training episodes", default=500)
    parser.add_argument("--epsilon", type=float, help="Starting value of epsilon", default=1.0)
    parser.add_argument("--epsilon_min", type=float, help="Minimum value of epsilon", default=0.01)
    parser.add_argument("--epsilon_decay", type=float, help="Epsilon decay rate", default=0.99)
    parser.add_argument("--discount", type=float, help="Discount factor", default=0.99)
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.01)
    parser.add_argument("--max_steps", type=int, help="Maximum steps per episode", default=2000)
    parser.add_argument("--render", type=int, help="Render the environment during training", default=1)
    parser.add_argument("--render_every", type=int, help="Interval of episodes between rendering", default=10)
    parser.add_argument("--save", type=int, help="Save model", default=0)
    parser.add_argument("--path", type=str, help="Path where the model is saved after training", default="./")
    args = parser.parse_args()

    train(
        agent_name=args.agent,
        episodes = args.episodes,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        discount=args.discount,
        lr=args.lr,
        max_steps=args.max_steps,
        render=args.render,
        render_every=args.render_every,
        save=args.save,
        path_dir=args.path
    )


if __name__ == "__main__":
    main()