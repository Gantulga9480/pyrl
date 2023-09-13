import numpy as np


class ReplayBufferBase(object):

    def __init__(self, max_size, min_size) -> None:
        self.max_size = max_size
        self.min_size = min_size

    @property
    def trainable(self):
        return False

    def push(self, *args):
        raise NotImplementedError

    def extend(self, *args):
        raise NotImplementedError

    def sample(self, *args):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError


class ReplayBuffer(ReplayBufferBase):

    def __init__(self, max_size, min_size, state_space_shape, action_space_shape=1) -> None:
        super().__init__(max_size, min_size)
        self.state_buffer = np.zeros((max_size, state_space_shape), dtype=np.float32)
        self.next_state_buffer = np.zeros((max_size, state_space_shape), dtype=np.float32)
        self.action_buffer = np.zeros((max_size, action_space_shape), dtype=np.int32)
        self.reward_buffer = np.zeros((max_size, 1), dtype=np.float32)
        self.done_buffer = np.zeros((max_size, 1), dtype=np.int32)
        self.buffer_idx = 0
        self.is_full = False

    @property
    def trainable(self):
        return self.buffer_idx >= self.min_size or self.is_full

    def push(self, state, action, next_state, reward, episode_over):
        """Data format [state, action, next_state, reward, episode_over]"""
        self.state_buffer[self.buffer_idx] = state
        self.action_buffer[self.buffer_idx] = action
        self.next_state_buffer[self.buffer_idx] = next_state
        self.reward_buffer[self.buffer_idx] = reward
        self.done_buffer[self.buffer_idx] = episode_over
        self.buffer_idx += 1
        if self.buffer_idx == self.max_size:
            self.is_full = True
            self.buffer_idx = 0

    def sample(self, sample_size):
        buffer_size_len = self.max_size if self.is_full else self.buffer_idx
        idx = np.random.choice(np.arange(buffer_size_len), sample_size, replace=False)
        s = np.array(self.state_buffer[idx])
        ns = np.array(self.next_state_buffer[idx])
        a = np.concatenate(self.action_buffer[idx])
        r = np.concatenate(self.reward_buffer[idx])
        d = np.concatenate(self.done_buffer[idx])
        return s, a, ns, r, d


class DoubleReplayBuffer(ReplayBufferBase):

    def __init__(self, max_size, min_size) -> None:
        super().__init__(max_size, min_size)
