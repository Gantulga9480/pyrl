import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import random
import os
from .agent import Agent


class DQN(Agent):

    def __init__(self, lr: float, y: float, batchs: int, epochs: int, gpu: bool = True) -> None:
        super().__init__(lr, y)
        self.batchs = batchs
        self.epochs = epochs
        self.gpu = gpu
        self.target_model = None
        if self.gpu:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Currently, memory growth needs to be the same across GPUs
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        tf.config.list_logical_devices('GPU')
                except RuntimeError as e:
                    # Memory growth must be set before GPUs have been initialized
                    print(e)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    def create_model(self, sizes: list) -> Sequential:
        self.model = Sequential()
        self.target_model = Sequential()
        self.model.add(Input(shape=(sizes[0],)))
        self.target_model.add(Input(shape=(sizes[0],)))
        for dim in sizes[1:-1]:
            self.model.add(Dense(dim, activation='relu'))
            self.target_model.add(Dense(dim, activation='relu'))
        self.model.add(Dense(sizes[-1], activation='linear'))
        self.target_model.add(Dense(sizes[-1], activation='linear'))
        self.model.compile(loss="mse",
                           optimizer=Adam(learning_rate=self.lr),
                           metrics=["accuracy"])
        self.model.summary()

    def update_target(self):
        if self.model:
            self.target_model.set_weights(self.model.get_weights())

    def save_model(self, path) -> None:
        if not path.endswith('.h5'):
            path += '.h5'
        if self.model:
            self.model.save(path)

    def load_model(self, path) -> None:
        if not path.endswith('.h5'):
            path += '.h5'
        try:
            self.model = load_model(path)
            self.target_model = load_model(path)
        except IOError:
            print('Model file not found!')
            exit()
        self.model.summary()

    def policy(self, state, greedy=False):
        self.step_count += 1
        if not greedy and np.random.random() < self.e:
            return random.randint(0, 4)
        else:
            action_values = self.model.predict(
                np.expand_dims(state, axis=0)
            )[0]
            return np.argmax(action_values)

    def learn(self, samples):
        current_states = np.array([item[0] for item in samples])
        new_current_state = np.array([item[2] for item in samples])
        current_qs_list = []
        future_qs_list = []
        current_qs_list = self.model.predict(current_states)
        future_qs_list = self.target_model.predict(new_current_state)

        X = []
        Y = []
        for index, (state, action, _, reward, done) in enumerate(samples):
            if not done:
                new_q = reward + self.y * np.max(future_qs_list[index])
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(state)
            Y.append(current_qs)
        self.model.fit(np.array(X), np.array(Y),
                       epochs=self.epochs,
                       batch_size=self.batchs,
                       shuffle=False,
                       verbose=1)


class ReplayBuffer:

    def __init__(self, max_size, min_size) -> None:
        self.max_size = max_size
        self.min_size = min_size
        self.buffer = deque(maxlen=max_size)

    @property
    def trainable(self):
        return self.buffer.__len__() >= self.min_size

    def push(self, data):
        self.buffer.append(data)

    def sample(self, sample_size):
        return random.sample(self.buffer, sample_size)


class DoubleReplayBuffer:

    def __init__(self, max_size, min_size) -> None:
        self.max_size = max_size
        self.min_size = min_size
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

    def sample(self, sample_size, factor):
        n_size = round(sample_size * factor)
        o_size = sample_size - n_size
        sn = random.sample(self.buffer_new, n_size)
        so = random.sample(self.buffer_old, o_size)
        so.extend(so)
        return sn
