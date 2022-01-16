from torch import nn


class DQN(nn.Sequential):
    def __init__(self, inputs, outputs):
        super(DQN, self).__init__(
            nn.Linear(inputs, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, outputs),
        )
