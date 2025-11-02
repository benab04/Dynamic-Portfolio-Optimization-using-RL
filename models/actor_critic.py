import numpy as np
from sklearn.neural_network import MLPRegressor
from scipy.special import softmax

class ActorCritic:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = 0.95
        
        # Actor and Critic networks
        self.actor = MLPRegressor(
            hidden_layer_sizes=(24, 24),
            learning_rate_init=learning_rate,
            max_iter=1,
            warm_start=True
        )
        self.critic = MLPRegressor(
            hidden_layer_sizes=(24, 24),
            learning_rate_init=learning_rate,
            max_iter=1,
            warm_start=True
        )
        
        # Initialize models with dummy fits
        self.actor.fit(np.zeros((1, state_size)), np.zeros((1, action_size)))
        self.critic.fit(np.zeros((1, state_size)), np.zeros((1, 1)))

    def act(self, state):
        """Choose action based on actor network"""
        action_logits = self.actor.predict(state.reshape(1, -1))[0]
        return softmax(action_logits)

    def train(self, state, action, reward, next_state, done):
        """Train both actor and critic networks"""
        # Get value predictions
        value = self.critic.predict(state.reshape(1, -1))[0]
        next_value = self.critic.predict(next_state.reshape(1, -1))[0] if not done else 0

        # Calculate advantage
        advantage = reward + self.gamma * next_value - value

        # Train critic
        target = reward + self.gamma * next_value
        self.critic.fit(state.reshape(1, -1), np.array([target]).reshape(-1, 1))

        # Train actor using policy gradient
        action_logits = self.actor.predict(state.reshape(1, -1))[0]
        target_probs = softmax(action_logits)
        target_probs[np.argmax(action)] += advantage
        
        self.actor.fit(state.reshape(1, -1), target_probs.reshape(1, -1))