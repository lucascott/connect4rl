from collections import deque

import torch
from torch import nn, optim

from connect4rl.rl.replay import Transition, ReplayMemory


class DQN(nn.Sequential):
    def __init__(self, inputs, outputs):
        super(DQN, self).__init__(
            nn.Linear(inputs, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, outputs),
        )


class DqnAgent:

    BATCH_SIZE = 32
    GAMMA = 0.99
    memory = ReplayMemory(10000)

    # Huber loss
    criterion = nn.SmoothL1Loss()

    loss_vals = deque(maxlen=10)

    def __init__(self, rows, cols):
        self.model = DQN(inputs=rows * cols, outputs=cols)
        self.target_model = DQN(inputs=rows * cols, outputs=cols)

        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.RMSprop(self.model.parameters())

    def predict(self, state):
        return self.model(state).argmax()

    def optimize(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool
        )
        non_final_next_states = torch.cat(
            [
                state.flatten().unsqueeze(0)
                for state in batch.next_state
                if state is not None
            ]
        )
        state_batch = torch.cat([state.flatten().unsqueeze(0) for state in batch.state])
        action_batch = torch.cat(batch.action).unsqueeze(-1)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE)
        next_state_values[non_final_mask] = (
            self.target_model(non_final_next_states).max(dim=1).values
        )

        # Compute the expected Q values
        expected_state_action_values = reward_batch + self.GAMMA * next_state_values

        loss = self.criterion(state_action_values, expected_state_action_values.detach())

        self.loss_vals.append(float(loss))
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def push(self, state, action, next_state, reward):
        transition = Transition(state, action, next_state, reward)
        self.memory.push(transition)
