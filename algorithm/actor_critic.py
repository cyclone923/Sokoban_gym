import torch
import torch.nn as nn
from algorithm.convlstm import ConvLSTM
from torch.distributions import Categorical

class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class ActorCritic(nn.Module):
    def __init__(self, device):
        super(ActorCritic, self).__init__()

        self.device = device
        self.n_latent_var = 32
        self.action_dim = 9
        self.action_distribution = Categorical
        self.convlstm = ConvLSTM(input_channels=7, hidden_channels=[16], kernel_size=3, device=device, step=1).to(self.device)
        self.mlp = nn.Sequential(
            nn.Linear(1600, self.n_latent_var),
            # nn.ReLU(),
            # nn.Linear(self.n_latent_var, self.n_latent_var),
            nn.ReLU(),
        ).to(self.device)

        self.actor = nn.Sequential(
            nn.Linear(self.n_latent_var, self.action_dim),
            nn.Softmax(dim=-1)
        ).to(self.device)

        self.critic = nn.Sequential(
            nn.Linear(self.n_latent_var, 1)
        ).to(self.device)

    def forward(self, x):
        hidden_conv, _ = self.convlstm(x)
        hidden_flatten = torch.flatten(hidden_conv, start_dim=1)
        hidden_final = self.mlp(hidden_flatten)
        return self.actor(hidden_final), self.critic(hidden_final)

    def reset_memory(self):
        self.convlstm.reset_memory()

    def act(self, obs, memory):
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        action_mean, _ = self.__call__(obs)

        dist = self.action_distribution(action_mean)
        action = dist.sample()

        memory.obs.append(obs)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.cpu().numpy()[0]

    def evaluate(self, memory):
        self.reset_memory()
        all_action_mean, all_state_value = [], []
        for obs, done in zip(memory.obs, memory.is_terminals):
            action_mean, state_value = self.__call__(obs)
            all_action_mean.append(action_mean)
            all_state_value.append(state_value)
            if done:
                self.reset_memory()

        action_mean = torch.cat(all_action_mean)

        dist = self.action_distribution(action_mean)

        all_action = torch.cat(memory.actions).to(self.device).detach()
        action_logprobs = dist.log_prob(all_action)
        dist_entropy = dist.entropy()

        state_value = torch.cat(all_state_value)
        return action_logprobs, state_value.squeeze(), dist_entropy