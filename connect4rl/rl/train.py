import logging
import random
import statistics
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm

from connect4rl.exceptions import BoardError
from connect4rl.rl.agent import DqnAgent
from connect4rl.rl.env import Connect4Rl

rows, cols = 6, 7
env = Connect4Rl(rows, cols)

agent = DqnAgent(rows, cols)


logging.basicConfig(level=logging.INFO)

_logger = logging.getLogger(__name__)


def process_state(state: np.ndarray):
    return torch.from_numpy(state.copy()).type(torch.FloatTensor)


def run():
    target_update = 5
    epsilon = 0.2
    max_episodes = 10000
    counter = Counter()
    for match in tqdm(range(max_episodes), unit=" matches", ncols=100):
        _logger.debug(f"Match {match}")
        env.reset()
        state = process_state(env.board.status())

        while True:
            # choose epsilon greedy action
            rand_sample = random.random()
            # TODO epsilon greedy
            if rand_sample > epsilon:
                with torch.no_grad():
                    action = agent.predict(torch.flatten(state))
            else:
                action = torch.tensor(random.randrange(cols), dtype=torch.long)

            try:
                next_state, reward, done, _ = env.step(int(action))
            except BoardError as err:
                _logger.debug(err)
                next_state, reward, done = None, 0, True

            if not done:
                next_state = process_state(next_state)

            # Store the transition in memory
            trans = (
                torch.unsqueeze(state, dim=0),
                torch.unsqueeze(action.detach(), dim=0),
                torch.unsqueeze(state, dim=0),
                torch.tensor([reward], dtype=torch.int),
            )
            agent.push(*trans)

            # Move to the next state
            state = next_state

            agent.optimize()

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
            agent.update_target()

        if agent.loss_vals and match % 10 == 0:
            _logger.info("%.4f", statistics.mean(agent.loss_vals))
        # epsilon -= 0.001
        _logger.debug("----------")
    _logger.info(counter)


if __name__ == "__main__":
    run()
