import random
from collections import deque, namedtuple

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ExperienceBuffer:
    def __init__(self, capacity):
        self.storage = deque(maxlen=capacity)
        self.capacity = capacity

    def remember(self, *args):
        self.storage.append(Transition(*args))

    def sample(self, batch):
        return random.sample(self.storage, batch)

    def __len__(self):
        return len(self.storage)
