import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
from .agent import Agent
from .utils import ReplayBuffer, DoubleReplayBuffer


class DQN(Agent):

    def __init__(self, action_space: list, lr: float, y: float, e_decay: float = 0.99999) -> None:
        super().__init__(action_space, lr, y, e_decay)
        self.target_model = None
        self.buffer = None
        self.batchs = 0
        self.epochs = 0
        self.gpu = False
        self.train_freq = 10
        self.update_freq = 10
        self.train_count = 0

    def create_buffer(self, max_size: int, min_size: int = 0, opt: int = 0):
        """opt - [0] for single buffer, [1] for double buffer"""
        if min_size == 0:
            min_size = self.batchs
        if opt == 0:
            self.buffer = ReplayBuffer(max_size=max_size, min_size=min_size)
        elif opt == 1:
            self.buffer = DoubleReplayBuffer(max_size=max_size, min_size=min_size)
        else:
            self.buffer = ReplayBuffer(max_size=max_size, min_size=min_size)

    def create_model(self, sizes: list, batchs: int = 128, epochs: int = 1, gpu: bool = False, train_freq: int = 10, update_freq: int = 10) -> Sequential:
        """sizes - [state_size, hidden_state, action_size]"""
        self.batchs = batchs
        self.epochs = epochs
        self.gpu = gpu
        self.train_freq = train_freq
        self.update_freq = update_freq
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

    def save_model(self, path) -> None:
        if not path.endswith('.h5'):
            path += '.h5'
        if self.model:
            self.model.save(path)

    def load_model(self, path) -> None:
        # TODO configure batch and epoch size
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
        """greedy - False (default) for training, True for inference"""
        self.step_count += 1
        if not greedy and np.random.random() < self.e:
            return np.random.choice(self.action_space)
        else:
            action_values = self.model.predict(np.expand_dims(state, axis=0))[0]
            return np.argmax(action_values)

    def learn(self, state, action, next_state, reward, episode_over):
        self.reward_history['step_reward'].append(reward)
        if episode_over:
            self.reward_history['episode_reward'].append(sum(self.reward_history['step_reward']))
            self.reward_history['step_reward'].clear()
        self.buffer.push([state, action, next_state, reward, episode_over])
        if self.buffer.trainable and self.train:
            if self.step_count % self.train_freq == 0:
                self.update_model(self.buffer.sample(self.batchs))
            elif self.train_count % self.update_freq == 0:
                self.update_target()
            self.decay_epsilon(self.e_decay)

    def update_target(self):
        if self.model:
            self.target_model.set_weights(self.model.get_weights())

    def update_model(self, samples):
        if not self.batchs or not self.epochs:
            print("Model not configured!, set batch size and epoch count")
            return
        self.train_count += 1
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
        self.model.fit(np.array(X), np.array(Y), epochs=self.epochs, batch_size=self.batchs, shuffle=False, verbose=0)
