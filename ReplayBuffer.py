import tensorflow as tf

class BaseBuffer(object):
    def __init__(self, agent, state_size, action_horizon, action_assets, max_size) -> None:
        self.agent = agent
        self.observations = tf.Variable(tf.zeros((max_size, state_size), dtype=tf.float32))
        self.actions = tf.Variable(tf.zeros((max_size, action_horizon, action_assets), dtype=tf.float32))
        self.rewards = tf.Variable(tf.zeros((max_size,)), dtype=tf.float32)
        
        self.max_size = max_size
        self.size = tf.Variable(0, dtype=tf.int32)
        self.next_idx = tf.Variable(0, dtype=tf.int32)


class SACReplayBuffer(BaseBuffer):
    def __init__(self, agent, state_size, action_horizon, action_assets, max_size) -> None:
        super().__init__(agent, state_size, action_horizon, action_assets, max_size)
        self.next_observations = tf.Variable(tf.zeros((max_size, state_size), dtype=tf.float32))
        self.shuffle = True
    
    
    @tf.function
    def store(self, observation, action, reward, next_observation) -> None:
        self.observations.scatter_nd_update([[self.next_idx]], [tf.squeeze(observation)])
        self.actions.scatter_nd_update([[self.next_idx]], [action])
        self.rewards.scatter_nd_update([[self.next_idx]], [reward])
        self.next_observations.scatter_nd_update([[self.next_idx]], [tf.squeeze(next_observation)])
        
        self.size.assign(tf.math.minimum(self.size + 1, self.max_size))
        self.next_idx.assign((self.next_idx + 1) % self.max_size)
        
        
    @tf.function
    def sample(self, minibatch_size):
        idxs = tf.random.uniform((minibatch_size,), maxval=self.size, dtype=tf.int32)
        
        observations = tf.gather(self.observations, idxs)
        actions = tf.gather(self.actions, idxs)
        rewards = tf.gather(self.rewards, idxs)
        next_observations = tf.gather(self.next_observations, idxs)
        
        # normalize rewards; avoiding division by zero
        mean_rewards = tf.reduce_mean(self.rewards[:self.size])
        mean_stds = tf.math.reduce_std(self.rewards[:self.size])
        normalized_rewards = (rewards - mean_rewards) / (mean_stds + self.agent.eps)
        
        return observations, actions, normalized_rewards, next_observations