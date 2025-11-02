import numpy as np
from sklearn.neural_network import MLPRegressor

class SARSAAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Initialize model with larger network and better parameters
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            learning_rate_init=learning_rate,
            max_iter=1,
            warm_start=True,
            activation='relu',
            solver='adam',
            early_stopping=True
        )
        
        # Initialize model with dummy data
        dummy_x = np.zeros((2, state_size))  # Multiple samples for better initialization
        dummy_y = np.zeros(2)  # Single output for Q-value
        self.model.fit(dummy_x, dummy_y)
        
        # Keep track of the best portfolio for exploitation
        self.best_portfolio = None
        self.best_value = float('-inf')

    def _preprocess_state(self, state):
        """Preprocess state to handle NaN and infinity values"""
        # Flatten and ensure no NaN or infinite values
        state_flat = state.flatten()
        return np.nan_to_num(state_flat, nan=0.0, posinf=1.0, neginf=-1.0)

    def act(self, state):
        """Choose action using epsilon-greedy policy with portfolio optimization"""
        state = self._preprocess_state(state)
        
        if np.random.rand() <= self.epsilon:
            # During exploration, try random portfolios
            portfolio_weights = np.random.dirichlet(np.ones(self.action_size))
            return portfolio_weights
        
        # During exploitation, try to improve upon the best known portfolio
        best_portfolio = None
        best_q_value = float('-inf')
        
        # Generate multiple candidate portfolios
        n_candidates = 20
        candidates = []
        
        # Include the best known portfolio if we have one
        if self.best_portfolio is not None:
            candidates.append(self.best_portfolio)
        
        # Add random portfolios
        for _ in range(n_candidates - len(candidates)):
            portfolio = np.random.dirichlet(np.ones(self.action_size))
            candidates.append(portfolio)
        
        # Evaluate all candidates
        for portfolio in candidates:
            state_action = np.concatenate([state, portfolio])
            q_value = self.model.predict(state_action.reshape(1, -1))[0]
            
            if q_value > best_q_value:
                best_q_value = q_value
                best_portfolio = portfolio
        
        return best_portfolio

    def train(self, state, action, reward, next_state, next_action, done):
        """Train the agent using SARSA update rule"""
        # Preprocess states
        state = self._preprocess_state(state)
        next_state = self._preprocess_state(next_state)
        
        # Combine state and action into a single input vector
        state_action = np.concatenate([state, action])
        next_state_action = np.concatenate([next_state, next_action])
        
        # Calculate target Q-value
        if done:
            target = reward
        else:
            next_q = self.model.predict(next_state_action.reshape(1, -1))[0]
            target = reward + self.gamma * next_q
        
        # Clip target to prevent extreme values
        target = np.clip(target, -1, 1)
        
        # Train the model to predict the Q-value
        try:
            self.model.fit(state_action.reshape(1, -1), np.array([target]))
        except Exception as e:
            print(f"Training error: {e}")
            print(f"State action shape: {state_action.shape}")
            print(f"Target shape: {np.array([target]).shape}")
            print(f"Target value: {target}")
            return
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update best portfolio if this one performed better
        if reward > self.best_value:
            self.best_value = reward
            self.best_portfolio = action.copy()