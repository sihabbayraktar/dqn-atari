import argparse

import gym
import torch
import torch.nn as nn

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize
from gym.wrappers import AtariPreprocessing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v0','Pong-v0'])
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v0': config.CartPole ,
    'Pong-v0': config.CartPole
}

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)
    #env = gym.make(args.env)
    env_config = ENV_CONFIGS[args.env]

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    # TODO: Create and initialize target Q-network.
    target_dqn = DQN(env_config=env_config).to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")
    steps = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_stack_size = 4
    for episode in range(env_config['n_episodes']):
        done = False


        obs = preprocess(env.reset(), env=args.env).unsqueeze(0)
        #print('obs', obs.shape)
        obs_stack = torch.cat(obs_stack_size * [obs]).unsqueeze(0).to(device)
        #print('obs_stack', obs_stack.shape)
        while not done:
            # TODO: Get action from DQN.
            #print('obs_stack.shape', obs_stack.shape)
            action = dqn.act(obs_stack, True).item()
            #print('action',action)
            # Act in the true environment.

            next_obs, reward, done, info = env.step(action)
            next_obs = preprocess(next_obs, env=args.env).unsqueeze(0)
            # Preprocess incoming observation.
            #print('obs_stack[:, 1:, ...]',obs_stack[:, 1:, ...].shape)
            #print('next_obs',next_obs.shape)
            #print('next_obs.unsqueeze(1)',(next_obs.unsqueeze(1)).shape)
            #print('done...', done)
            if not done:
                next_obs = preprocess(next_obs, env=args.env)#.unsqueeze(0)
                next_obs_stack = torch.cat((obs_stack[:, 1:, ...], next_obs.unsqueeze(1)), dim=1).to(device)
                action = torch.tensor(action, device=device).long()#.unsqueeze(0)
                reward = preprocess(reward, env=args.env)#.unsqueeze(0)
                #print('next_obs',next_obs)
            else:
                #print('done...')
                next_obs = preprocess(next_obs, env=args.env)#.unsqueeze(0)
                next_obs_stack = torch.cat((obs_stack[:, 1:, ...], next_obs.unsqueeze(1)), dim=1).to(device)
                action = torch.tensor(action, device=device).long()#.unsqueeze(0)
                reward = preprocess(-1, env=args.env)#.unsqueeze(0)
                #print('next_obs',next_obs)
            # TODO: Add the transition to the replay memory. Remember to convert
            #       everything to PyTorch tensors!
            #print('next_obs_stack..squeeze(0)',next_obs_stack.shape.squeeze(0))
            done_ = torch.tensor(done, device=device).int()#.unsqueeze(0)
            memory.push(obs_stack.squeeze(0), action, next_obs_stack.squeeze(0), reward, done_)
            #print('done', done_)
            # TODO: Run DQN.optimize() every env_config["train_frequency"] steps.
            steps = steps + 1
            #print('steps', steps)
            if steps % env_config["train_frequency"] == 0 :
                optimize(dqn = dqn, target_dqn=target_dqn, memory=memory, optimizer=optimizer)
            # TODO: Update the target network every env_config["target_update_frequency"] steps.
            if steps % env_config["target_update_frequency"] == 0:
                target_dqn.load_state_dict(dqn.state_dict())
            obs = next_obs
            obs_stack = next_obs_stack
        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
            
            print(f'Episode {episode}/{env_config["n_episodes"]}: {mean_return}')

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                torch.save(dqn, f'models/{args.env}_best.pt')
        
    # Close environment after training is completed.
    env.close()
