from collections import deque
import random


class ReplayBufferBase(object):

    def __init__(self, max_size, min_size) -> None:
        self.max_size = max_size
        self.min_size = min_size
        self.buffer = None

    @property
    def trainable(self):
        pass

    def push(self, *args):
        raise NotImplementedError

    def extend(self, *args):
        raise NotImplementedError

    def sample(self, *args):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError


class ReplayBuffer(ReplayBufferBase):

    def __init__(self, max_size, min_size) -> None:
        super().__init__(max_size, min_size)
        self.buffer = deque(maxlen=max_size)

    @property
    def trainable(self):
        return self.buffer.__len__() >= self.min_size

    def push(self, data):
        """Data format [state, action, next_state, reward, episode_over]"""
        self.buffer.append(data)

    def extend(self, datas):
        """Data format [[states], [actions], [next_states], [rewards], [episode_overs]]"""
        data = list(zip(datas[0], datas[1], datas[2], datas[3], datas[4]))
        self.buffer.extend(data)

    def sample(self, sample_size):
        return random.sample(self.buffer, sample_size)

    def clear(self):
        self.buffer.clear()


class DoubleReplayBuffer(ReplayBufferBase):

    def __init__(self, max_size, min_size) -> None:
        super().__init__(max_size, min_size)
        self.buffer_new = deque(maxlen=max_size)
        self.buffer_old = deque(maxlen=max_size * 4)

    @property
    def trainable(self):
        fn = self.buffer_new.__len__() >= self.min_size
        fo = self.buffer_old.__len__() >= self.min_size
        return fn and fo

    def push(self, data):
        if self.buffer_new.__len__() == self.max_size:
            self.buffer_old.append(self.buffer_new.popleft())
        self.buffer_new.append(data)

    def extend(self, datas):
        raise NotImplementedError

    def sample(self, sample_size, factor):
        n_size = round(sample_size * factor)
        o_size = sample_size - n_size
        sn = random.sample(self.buffer_new, n_size)
        so = random.sample(self.buffer_old, o_size)
        sn.extend(so)
        return sn

    def clear(self):
        self.buffer_new.clear()
        self.buffer_old.clear()
