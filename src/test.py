import argparse
import pickle
import torch
from environment import SnakeGame
from QAgent import QAgent
from DeepQAgent import DeepQAgent
import random

def test(agent_name, episodes, epsilon, render, frame_rate, max_steps, Q_table, policy_net_state_dict):
    if agent_name == "QAgent":
        agent = QAgent(12, 4)
    if agent_name == "DeepQAgent":
        agent =  DeepQAgent(87, 4)

    if agent_name == "QAgent":
        with open(Q_table, "rb") as f:
            agent.Q = pickle.load(f)
    if agent_name == "DeepQAgent":
        agent.policy_net.load_state_dict(torch.load(policy_net_state_dict))
    
    env = SnakeGame()
    for episode in range(1, episodes + 1):
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
            if render:
                env.render()
                env.clock.tick(frame_rate)
            s = s_
        print(f"EPISODE {episode}   SCORE {env.score}")





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, help="Agent to train, choose betweeen QAgent and DeepQAgent", default="QAgent")
    parser.add_argument("--episodes", type=int, help="Number of training episodes", default=10)
    parser.add_argument("--epsilon", type=float, help="Starting value of epsilon", default=0.01)
    parser.add_argument("--max_steps", type=int, help="Starting value of epsilon", default=2000)
 
    parser.add_argument("--render", type=int, help="Render the environment during training", default=1)
    parser.add_argument("--frame_rate", type=int, help="Interval of episodes between rendering", default=24)
    parser.add_argument("--Q_table", type=str, help="Path where the Q table is saved")
    parser.add_argument("--policy_net_state_dict", type=str, help="Path where the policy_network_state_dict is saved")
    args = parser.parse_args()

    test(
        agent_name=args.agent,
        episodes=args.episodes,
        epsilon=args.epsilon,
        max_steps=args.max_steps,
        render=args.render,
        frame_rate=args.frame_rate,
        Q_table=args.Q_table,
        policy_net_state_dict=args.policy_net_state_dict
    )


if __name__ == "__main__":
    main()