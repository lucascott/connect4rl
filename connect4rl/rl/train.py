import logging
import random
from collections import Counter

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from connect4rl.exceptions import BoardError
from connect4rl.rl.agent import DQN
from connect4rl.rl.env import Connect4Rl
from connect4rl.rl.replay import ReplayMemory

rows, cols = 6, 7
env = Connect4Rl(rows, cols)

model = DQN(inputs=rows * cols, outputs=cols)
target_model = DQN(inputs=rows * cols, outputs=cols)

optimizer = optim.RMSprop(model.parameters())
memory = ReplayMemory(10000)

BATCH_SIZE = 32
GAMMA = 0.99
logging.basicConfig(level=logging.DEBUG)

_logger = logging.getLogger(__name__)


def process_state(state: np.ndarray):
    return torch.from_numpy(state.copy()).type(torch.FloatTensor)


def run():

    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    target_update = 5
    epsilon = 0

    counter = Counter()
    for match in tqdm(range(1000), unit=" matches", ncols=100):
        _logger.debug(f"Match {match}")
        env.reset()
        state = process_state(env.board.status())

        while True:
            # choose epsilon greedy action
            rand_sample = random.random()
            # TODO epsilon greedy
            if rand_sample > epsilon:
                with torch.no_grad():
                    action = model(torch.flatten(state)).argmax()
            else:
                action = torch.tensor(random.randrange(cols), dtype=torch.long)

            try:
                next_state, reward, done, _ = env.step(action.numpy())
            except BoardError as err:
                _logger.debug(err)
                next_state, reward, done = None, 0, True

            if not done:
                next_state = process_state(next_state)

            # Store the transition in memory
            memory.push(
                state,
                action.detach(),
                next_state,
                torch.tensor(reward, dtype=torch.long),
            )

            # Move to the next state
            state = next_state

            if done:
                env.board.show()
                counter[reward] += 1
                if reward > 0:
                    _logger.debug("Agent won")
                elif reward < 0:
                    _logger.debug("Opponent won")
                break

        # Update the target network, copying all weights and biases in DQN
        if match % target_update == 0:
            target_model.load_state_dict(model.state_dict())
        _logger.debug("----------")
    _logger.info(counter)


if __name__ == "__main__":
    run()
