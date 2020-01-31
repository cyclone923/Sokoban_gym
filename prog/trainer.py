import gym
import gym_sokoban
import torch
from algorithm.ppo import PPO
from tool.memory import Memory
from tool.settings import get_env_setting
import numpy as np
import random
import os

class Trainer:
    def __init__(self, args):
        if args.seed is not None:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)

        self.env_name = args.environment
        self.env_setting = get_env_setting(self.env_name)
        self.solved_reward = self.env_setting["solved_reward"]
        self.update_timestep = self.env_setting["update_timestep"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.env = gym.make(args.environment)
        self.alg = PPO(args, self.device)
        self.log_interval = 5 # print avg reward in the interval
        self.max_episodes = 100000
        self.render = False

    def train(self):
        # logging variables
        running_reward = 0
        avg_length = 0
        time_step = 0
        memory = Memory()
        # self.alg.load_dict("./", self.env_name, self.alg_name, self.net_name)

        # training loop
        time_step = 0
        for i_episode in range(1, self.max_episodes + 1):
            self.alg.reset_memory()
            obs = self.env.reset(render_mode="logic")
            t = 0
            while True:
                t += 1
                # Running policy_old:
                action = self.alg.take_action(obs, memory)
                self.env.render()
                obs, reward, done, _ = self.env.step(action, observation_mode="logic")

                # Saving reward and is_terminal:
                memory.rewards.append(reward)
                memory.is_terminals.append(done)

                running_reward += reward
                if self.render:
                    self.env.render()
                if done:
                    break
            time_step += t

            # update if its time
            if time_step >= self.update_timestep and done == True:
                self.alg.update(memory)
                memory.clear_memory()
                time_step = 0

            avg_length += t

            # save every 500 episodes
            if i_episode % 500 == 0:
                directory = "./epoch_performance"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                self.alg.save_dict(directory, f'{self.env_name}_{i_episode}')

            # logging
            if i_episode % self.log_interval == 0:
                avg_length = int(avg_length / self.log_interval)
                running_reward = int((running_reward / self.log_interval))
                print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))

                # stop training if avg_reward > solved_reward or reaches the limit of training epoches
                if running_reward > (self.log_interval * self.solved_reward):
                    print("########## Solved! ##########")
                    directory = "./success/"
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    self.alg.save_dict(directory, f'{self.env_name}_{self.log_interval}')
                    break

                running_reward = 0
                avg_length = 0

