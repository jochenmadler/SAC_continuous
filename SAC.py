from ReplayBuffer import SACReplayBuffer

import os
import math
import glob
import random
import warnings
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.regularizers import L2
import tensorflow_probability as tfp

class SAC():
    def __init__(self, config) -> None:
        self.config = config # can be dictionary or wandb.config object
        
        # use GPU when available, otherwise CPU
        self.device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
        
        # if wandb: use current timestamp and create checkpoint directory
        if self.config is not None and type(self.config) != dict:
            self.timestamp = datetime.now().strftime(format="%Y-%m-%d_%H-%M-%S")
            self.checkpoint_dir, self.checkpoint_ctr = None, 1
            if self.wandb_config is not None:
                wandb_dirname = self.wandb.run.dir.split('-')[1:-1][0]
                self.timestamp = datetime.strptime(wandb_dirname, '%Y%m%d_%H%M%S').strftime('%Y-%m-%d_%H-%M-%S')
                self.checkpoint_dir = os.path.join(self.wandb.run.dir, 'checkpoints')
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                self.wandb.run.name = self.timestamp # rename wandb run to timestamp
        
        # default configuration
        if self.config is None:
            self.state_size = None # integer, e.g. 32
            self.action_size = None # tuple (t, k) where t*k actions are sampled from t k-dimensional distributions
            self.max_steps = 1000
            self.max_episode_steps = 24
            self.min_steps = 24
            self.warmup = False
            self.buffer_size = 1000
            self.minibatch_size = 256
            self.update_interval = 1
            self.validation_interval = 0
            self.preprocess_state = False
            self.actor_weights_scaling = 0.1
            self.activation_function = 'tanh'
            self.weights_initializer = 'glorot_uniform'
            self.pol_nr_layers = 2
            self.pol_hidden_size = 16
            self.val_nr_layers = 2
            self.val_hidden_size = 16
            self.gamma = 0.99
            self.lr = 1e-3
            self.alpha_init = 0.001
            self.alpha_lr = 0.
            self.alpha_to_zero_steps = 1000
            self.alpha_decay_rate = 0.
            self.polyak = 0.995
            self.huber_delta = 2.
            self.gradient_clip_norm = 2.
            self.reg_coef = 0.
            self.std_initial_value = 0.4
            self.seed = 42
        else:
            # configuration from wandb.config or config dictionary
            self.state_size = self.config.state_size if type(self.config) != dict else self.config['state_size']
            self.action_size = self.config.action_size if type(self.config) != dict else self.config['action_size']
            self.max_steps = self.config.max_steps if type(self.config) != dict else self.config['max_steps']
            self.max_episode_steps = self.config.max_episode_steps if type(self.config) != dict else self.config['max_episode_steps']
            self.min_steps = self.config.min_steps if type(self.config) != dict else self.config['min_steps']
            self.warmup = self.config.warmup if type(self.config) != dict else self.config['warmup']
            self.buffer_size = self.config.buffer_size if type(self.config) != dict else self.config['buffer_size']
            self.minibatch_size = self.config.minibatch_size if type(self.config) != dict else self.config['minibatch_size']
            self.update_interval = self.config.update_interval if type(self.config) != dict else self.config['update_interval']
            self.validation_interval = self.config.validation_interval if type(self.config) != dict else self.config['validation_interval']
            self.preprocess_state = self.config.preprocess_state if type(self.config) != dict else self.config['preprocess_state']
            self.actor_weights_scaling = float(self.config.actor_weights_scaling) if type(self.config) != dict else float(self.config['actor_weights_scaling'])
            self.activation_function = self.config.activation_function if type(self.config) != dict else self.config['activation_function']
            self.weights_initializer = self.config.weights_initializer if type(self.config) != dict else self.config['weights_initializer']
            self.pol_nr_layers = self.config.pol_nr_layers if type(self.config) != dict else self.config['pol_nr_layers']
            self.pol_hidden_size = self.config.pol_hidden_size if type(self.config) != dict else self.config['pol_hidden_size']
            self.val_nr_layers = self.config.val_nr_layers if type(self.config) != dict else self.config['val_nr_layers']
            self.val_hidden_size = self.config.val_hidden_size if type(self.config) != dict else self.config['val_hidden_size']
            self.gamma = self.config.gamma if type(self.config) != dict else self.config['gamma']
            self.lr = float(self.config.lr) if type(self.config) != dict else float(self.config['lr'])
            self.alpha_init = float(self.config.alpha_init) if type(self.config) != dict else float(self.config['alpha_init'])
            self.alpha_lr = float(self.config.alpha_lr) if type(self.config) != dict else float(self.config['alpha_lr'])
            self.alpha_to_zero_steps = self.config.alpha_to_zero_steps if type(self.config) != dict else self.config['alpha_to_zero_steps']
            self.alpha_decay_rate = float(self.config.alpha_decay_rate) if type(self.config) != dict else float(self.config['alpha_decay_rate'])
            self.polyak = self.config.polyak if type(self.config) != dict else self.config['polyak']
            self.huber_delta = self.config.huber_delta if type(self.config) != dict else self.config['huber_delta']
            self.gradient_clip_norm = self.config.gradient_clip_norm if type(self.config) != dict else self.config['gradient_clip_norm']
            self.reg_coef = float(self.config.reg_coef) if type(self.config) != dict else float(self.config['reg_coef'])
            self.std_initial_value = self.config.std_initial_value if type(self.config) != dict else self.config['std_initial_value']
            self.seed = self.config.seed if type(self.config) != dict else self.config['seed']

        # assert that state and action sizes are defined
        assert self.state_size is not None and self.action_size is not None, 'Please define state and action size (self.state_size, self.action_size).'

        # set seed(s) for reproducibility and training/testing mode
        self.set_seed(self.seed)
        self.training = True
        
        # if warmup is set to True, use minimum steps for warmup
        self.warmup_steps = self.min_steps if self.warmup else 0
        
        # set up preprocessing configuration
        self.preprocessor = None # define your own preprocessing here
    
        # set up replay buffer
        self.buffer = SACReplayBuffer(self, self.state_size, self.action_size[0], self.action_size[1], self.buffer_size)
    
        # create alpha: temperature variable / entropy term
        self.target_entropy = -tf.constant(math.prod(self.action_size), dtype=tf.float32)
        self.log_alpha = tf.Variable(tf.constant(tf.math.log(self.alpha_init)), dtype=tf.float32)
        self.alpha = tfp.util.DeferredTensor(pretransformed_input=self.log_alpha, transform_fn=tf.exp)
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha_lr)
        
        # set minimum std. and calculate constant for softplus transformation
        self.eps = 1e-5
        self.min_std = 1e-3
        self.std_constant = tfp.math.softplus_inverse(self.std_initial_value - self.min_std) if self.std_initial_value is not None else 0.0
    
        # create actor (policy) and critic (Q) networks
        self.actor = self.create_actor_model(hidden_size=self.pol_hidden_size, nr_layers=self.pol_nr_layers, layer_suffix='actor')
        self.q1 = self.create_critic_model(hidden_size=self.val_hidden_size, nr_layers=self.val_nr_layers, layer_suffix='q1')
        self.q2 = self.create_critic_model(hidden_size=self.val_hidden_size, nr_layers=self.val_nr_layers, layer_suffix='q2')    
        self.q1_target = self.create_critic_model(hidden_size=self.val_hidden_size, nr_layers=self.val_nr_layers, layer_suffix='q1_target')
        self.q2_target = self.create_critic_model(hidden_size=self.val_hidden_size, nr_layers=self.val_nr_layers, layer_suffix='q2_target')
        self.q1_target.set_weights(self.q1.get_weights())
        self.q2_target.set_weights(self.q2.get_weights())
    
        # create optimizer for NN models
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=0.9)
        self.q1_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.q2_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        
    
    # set seed for reproducibility
    def set_seed(self, seed) -> None:
        # numpy and tensorflow
        np.random.seed(seed)
        tf.random.set_seed(seed)
        tf.experimental.numpy.random.seed(seed)
        # CuDNN backend
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        # python environment
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)   
    
    
    def create_actor_model(self, hidden_size, nr_layers, layer_suffix):
        # input layer
        input = layers.Input(shape=(self.state_size,), name=f'state_input_{layer_suffix}', dtype=tf.float32)
        x = input

        # dense layers
        for i in range(nr_layers):
            x = layers.Dense(hidden_size, activation=self.activation_function, kernel_initializer=self.weights_initializer, kernel_regularizer=L2(self.reg_coef), name=f'dense_{i}_{layer_suffix}')(x)

        # output layers
        mean_layer = layers.Dense(math.prod(self.action_size), activation=None, name=f'mean_output_{layer_suffix}', kernel_initializer='glorot_normal', kernel_regularizer=L2(self.reg_coef))(x)
        logstd_layer = layers.Dense(math.prod(self.action_size), activation=None, name=f'logstd_output_{layer_suffix}', kernel_initializer='glorot_normal', kernel_regularizer=L2(self.reg_coef))(x)
        model = models.Model(inputs=[input], outputs=[mean_layer, logstd_layer])
            
        # scale weights of all output layers
        all_output_layers = [layer for layer in model.layers if '_output' in layer.name]
        for layer in all_output_layers:
            weights, biases = layer.get_weights()
            scaled_weights = weights * self.actor_weights_scaling
            layer.set_weights([scaled_weights, biases])
            
        # instantiate models
        dummy_observation = tf.ones((1, self.state_size), dtype=tf.float32)
        model(dummy_observation)

        return model
    
    
    def create_critic_model(self, hidden_size, nr_layers, layer_suffix):
        # input layer
        action_input = layers.Input(shape=(math.prod(self.action_size),), name=f'action_input_{layer_suffix}')
        state_input = layers.Input(shape=(self.state_size,), name=f'state_input_{layer_suffix}')
        x = layers.Concatenate(axis=-1, name=f'state_action_input_{layer_suffix}')([state_input, action_input])
        inputs = [state_input, action_input]

        # common Dense layers
        for i in range(nr_layers):
            x = layers.Dense(hidden_size, activation=self.activation_function, kernel_initializer=self.weights_initializer, kernel_regularizer=L2(self.reg_coef), name=f'dense_{i}_{layer_suffix}')(x)

        # output layer
        output_layer = layers.Dense(1, activation=None, kernel_initializer=self.weights_initializer, kernel_regularizer=L2(self.reg_coef), name=f'q_value_output_{layer_suffix}')(x)
        model = models.Model(inputs=inputs, outputs=output_layer)

        # instantiate model
        dummy_state = tf.ones((1, self.state_size), dtype=tf.float32)
        dummy_action = tf.ones((1, math.prod(self.action_size)), dtype=tf.float32)
        model([dummy_state, dummy_action])

        return model
        
        
    @tf.function
    def random_action(self):
        mean = tf.zeros((self.action_size[0], self.action_size[1]), dtype=tf.float32)
        std = tf.zeros((self.action_size[0], self.action_size[1]), dtype=tf.float32)
        action = tf.random.uniform((self.action_size[0], self.action_size[1]), minval=-1, maxval=1, dtype=tf.float32)
        return mean, std, action, action


    # private helper function for select_actions
    def _select_action(self, observation, training):
        # preprocess (scale) raw observation_buffer
        if self.preprocess_state:
            observation = tf.cast(self.preprocessor(observation), dtype=tf.float32)
        else:
            observation = tf.cast(observation, dtype=tf.float32)
        nr_obses = tf.shape(observation)[0]

        # perform actor model forward pass
        mean, logstd = self.actor(observation)
        
        # recover action shapes from actor output
        mean = tf.reshape(mean, shape=(nr_obses, self.action_size[0], self.action_size[1]))
        logstd = tf.reshape(logstd, shape=(nr_obses, self.action_size[0], self.action_size[1]))
        
        # clip logstd for numerical stability
        logstd = tf.clip_by_value(logstd, -20.0, 2.5) # alternative: tf.exp() with (-6.5, 1.0)
        std = tf.nn.softplus(logstd + self.std_constant) + self.min_std # alternative: tf.exp()
        
        # create multivariate normal distribution with dim (action_size[0], action_size[1])
        dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std)
        
        # depending on agent mode, select action and calculate logprob
        if training:
            raw_action = dist.sample()
            logpis = dist.log_prob(raw_action) # logprob of policy (pi) PDF, can be >1
            action = tf.tanh(raw_action)
            diff = tf.reduce_sum(tf.math.log(1 - tf.math.pow(action,2) + self.eps), axis=2) # squashing correction
            logpis -= diff
            entropy = dist.entropy()
        else:
            raw_action = dist.mean()
            action = tf.tanh(raw_action)
            logpis = tf.zeros((nr_obses, self.action_size[0])) # placeholder, not used
            entropy = tf.zeros((nr_obses, self.action_size[0])) # placeholder, not used
            pass
        
        return mean, std, action, logpis, entropy, raw_action
    
    
    @tf.function
    def select_action(self, observation, training=True):
        mean, std, action, logpis, entropy, raw_action = self._select_action(observation, training)
        if observation.shape[0] == 1:
            mean, std, action, raw_action = [tf.squeeze(x, axis=0) for x in [mean, std, action, raw_action]]
        return mean, std, action, logpis, entropy, raw_action

    def get_current_value(self, observation, action):
        # preprocess (scale) raw observation_buffer
        if self.preprocess_state:
            observation = tf.cast(self.preprocessor(observation), dtype=tf.float32)
        else:
            observation = tf.cast(observation, dtype=tf.float32)
            
        # get Q-value estimates for current state-action pair
        current_q1 = self.q1([observation, action])
        current_q2 = self.q2([observation, action])
        
        return current_q1, current_q2
        
    
    @tf.function
    def update_value(self, observation_buffer, action_buffer, return_buffer, next_observation_buffer):
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                action_buffer = tf.reshape(action_buffer, shape=(tf.shape(action_buffer)[0],-1))
                current_q1, current_q2 = self.get_current_value(observation_buffer, action_buffer)
                _, _, next_actions, next_logpis, _, _ = self._select_action(next_observation_buffer, training=self.training)
                next_actions = tf.reshape(next_actions, shape=(tf.shape(next_actions)[0],-1))
                 
                # preprocess (scale) raw observation_buffer
                if self.preprocess_state:
                    next_observation_buffer = tf.cast(self.preprocessor(next_observation_buffer), dtype=tf.float32)
                else:
                    next_observation_buffer = tf.cast(next_observation_buffer, dtype=tf.float32)
                
                # calculate target Q-values
                target_q1 = tf.stop_gradient(self.q1_target([next_observation_buffer, next_actions]))
                target_q2 = tf.stop_gradient(self.q2_target([next_observation_buffer, next_actions]))
                target_min_q = tf.minimum(target_q1, target_q2) # double Q-trick
                
                # calculate combined (huber) loss for both networks
                target_q = tf.stop_gradient(
                    return_buffer + self.gamma * (tf.squeeze(target_min_q) - self.alpha * tf.reduce_mean(next_logpis, axis=1)))
                q1_loss = tf.reduce_mean(self.huber_loss(target_q - tf.squeeze(current_q1), delta=self.huber_delta))
                q2_loss = tf.reduce_mean(self.huber_loss(target_q - tf.squeeze(current_q2), delta=self.huber_delta))
                combined_loss = q1_loss + q2_loss
            
            # update current Q-value networks with combined loss
            combined_grads = tape.gradient(combined_loss, self.q1.trainable_variables + self.q2.trainable_variables)
            combined_grads, _ = tf.clip_by_global_norm(combined_grads, self.gradient_clip_norm)
            self.q1_optimizer.apply_gradients(zip(combined_grads[:len(self.q1.trainable_variables)], self.q1.trainable_variables))
            self.q2_optimizer.apply_gradients(zip(combined_grads[len(self.q1.trainable_variables):], self.q2.trainable_variables))
            
            # update target Q-value networks (polyak averaging)
            for target_weights, weights in zip(self.q1_target.trainable_variables, self.q1.trainable_variables):
                target_weights.assign(self.polyak * target_weights + (1. - self.polyak) * weights)
            for target_weights, weights in zip(self.q2_target.trainable_variables, self.q2.trainable_variables):
                target_weights.assign(self.polyak * target_weights + (1. - self.polyak) * weights)
            
        # release the resources held by the gradient tape
        del tape
        
        return q1_loss, q2_loss
    
    
    # Huber loss: MSE for -delta < x < delta, MAE otherwise
    def huber_loss(self, x, delta):
        delta = tf.ones_like(x) * delta
        less_than_max = 0.5 * tf.square(x) # MSE
        greater_than_max = delta * (tf.abs(x) - 0.5 * delta) # MAE
        return tf.where(tf.abs(x) <= delta, less_than_max, greater_than_max)
        
    
    @tf.function
    def update_policy(self, observation_buffer):
        
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                # obtain actions and their logprobs from observations
                _, _, actions, logpis, entropies, _ = self._select_action(observation_buffer, training=self.training)
                actions = tf.reshape(actions, shape=(tf.shape(actions)[0],-1))
                
                # obtain Q-value estimates for the state-actions pairs
                current_q1, current_q2 = self.get_current_value(observation_buffer, actions)
                current_min_q = tf.minimum(current_q1, current_q2) # double Q-trick
                
                # calculate policy and alpha loss
                policy_loss = -tf.reduce_mean(current_min_q - self.alpha * logpis)
                alpha_loss = -tf.reduce_mean((self.alpha * tf.stop_gradient(self.target_entropy - entropies)))
            
            # update actor network(s) with clipped gradients
            grads = tape.gradient(policy_loss, self.actor.trainable_variables)
            grads, grads_norm = tf.clip_by_global_norm(grads, self.gradient_clip_norm)
            self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
                
            # update alpha variable (optional)
            if tf.greater(self.alpha_optimizer.learning_rate, 0.0):
                alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
                alpha_grads, _ = tf.clip_by_global_norm(alpha_grads, self.gradient_clip_norm)
                self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
            
        # release the resources held by the gradient tape
        del tape
            
        return policy_loss, tf.reduce_min(logpis), tf.reduce_max(logpis), tf.reduce_mean(logpis), alpha_loss, tf.reduce_mean(current_min_q), tf.reduce_mean(entropies), grads_norm

    
    # agent update
    def train(self):
        # get random minibatch from replay buffer
        observation_buffer, action_buffer, return_buffer, next_observation_buffer = self.buffer.sample(self.minibatch_size)

        # update value network
        q1_loss, q2_loss = self.update_value(observation_buffer, action_buffer, return_buffer, next_observation_buffer)
        
        # update policy network
        policy_loss, min_logpi, max_logpi, mean_logpi, alpha_loss, min_q, entropy, actor_grads_norm = self.update_policy(observation_buffer)
    
        return q1_loss, q2_loss, min_q, policy_loss, min_logpi, max_logpi, mean_logpi, alpha_loss, entropy, actor_grads_norm
    
    
    # manage alpha temperature parameter, depending on scheduling method
    def manage_alpha_value(self, steps) -> None:
        # decay
        self.log_alpha.assign(self.log_alpha - self.alpha_decay_rate)
        # set to zero after certain number of steps
        if steps == self.alpha_to_zero_steps and self.alpha_to_zero_steps > 0:
            self.log_alpha.assign(-1/self.eps)
            self.alpha_optimizer.learning_rate.assign(0.0)
            self.alpha_decay_rate = 0.0