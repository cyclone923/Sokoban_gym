import gym
import gym_sokoban
import torch
import algorithm
from tool.memory import Memory
from tool.settings import get_env_setting

class Trainer:
    def __init__(self, args):
        self.env_name = args.environment
        self.env_setting = get_env_setting(self.env_name)
        self.solved_reward = self.env_setting["solved_reward"]
        self.update_timestep = self.env_setting["update_timestep"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.env = gym.make(args.environment)
        self.alg = algorithm.PPO(args, self.env, self.device)
        self.log_interval = 20 # print avg reward in the interval
        self.max_episodes = 100000
        self.render = False

        if args.seed:
            torch.manual_seed(args.seed)
            self.env.seed(args.seed)

    def train(self):
        # logging variables
        running_reward = 0
        avg_length = 0
        time_step = 0

        memory = Memory()

        # self.alg.load_dict("./", self.env_name, self.alg_name, self.net_name)

        # training loop
        for i_episode in range(1, self.max_episodes + 1):
            obs = self.env.reset(render_mode="logic")
            t = 0
            while True:
                t += 1
                # Running policy_old:
                action = self.alg.take_action(obs, memory)
                obs, reward, done, _ = self.env.step(action, observation_mode="logic")

                # Saving reward and is_terminal:
                memory.rewards.append(reward)
                memory.is_terminals.append(done)

                # update if its time
                if time_step >= self.update_timestep and done == True:
                    self.alg.update(memory)
                    memory.clear_memory()
                    time_step = 0

                running_reward += reward
                if self.render:
                    self.env.render()
                if done:
                    break

            avg_length += t

            # stop training if avg_reward > solved_reward or reaches the limit of training epoches
            if running_reward > (self.log_interval * self.solved_reward):
                print("########## Solved! ##########")
                directory = "./preTrained/"
                self.alg.save_dict(directory, self.env_name, self.alg_name, self.net_name)
                break

            # # save every 500 episodes
            # if i_episode % 500 == 0:
            #     directory = "./"
            #     self.alg.save_dict(directory, self.env_name, self.alg_name, self.net_name)

            # logging
            if i_episode % self.log_interval == 0:
                avg_length = int(avg_length / self.log_interval)
                running_reward = int((running_reward / self.log_interval))

                print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
                running_reward = 0
                avg_length = 0

    def test(self, n_episodes):
        memory = Memory()

        # self.alg.load_dict("./", self.alg_name, self.env_name, self.net_name)

        # testing loop
        for ep in range(1, n_episodes + 1):
            ep_reward = 0
            obs = self.env.reset()
            while True:
                action = self.alg.take_action(obs, memory)
                obs, reward, done, _ = self.env.step(action)
                ep_reward += reward
                if self.render:
                    self.env.render()
                if done:
                    break

            print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
            self.env.close()

