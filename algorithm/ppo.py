import torch
import torch.nn as nn
from .actor_critic import ActorCritic
import os

class PPO:
    def __init__(self, args, device):

        self.lr = args.lr
        self.betas = args.betas
        self.gamma = args.gamma
        self.eps_clip = args.eps_clip
        self.k_epochs = args.k_epochs
        self.device = device

        self.policy = ActorCritic(device).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.policy_old = ActorCritic(device).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def reset_memory(self):
        self.policy.reset_memory()
        self.policy_old.reset_memory()

    def take_action(self, state, memory):
        return self.policy_old.act(state, memory)

    def save_dict(self, directory, env_name):
        torch.save(self.policy.state_dict(), os.path.join(directory, '{}.pth'.format(env_name)))

    def load_dict(self, directory, env_name):
        self.policy_old.load_state_dict(torch.load(os.path.join(directory, '{}.pth'.format(env_name))))
        self.policy.load_state_dict(torch.load(os.path.join(directory, '{}.pth'.format(env_name))))

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.append(discounted_reward)
        rewards = list(reversed(rewards))

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(self.device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_logprobs = torch.cat(memory.logprobs).to(self.device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.k_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(memory)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.001 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())