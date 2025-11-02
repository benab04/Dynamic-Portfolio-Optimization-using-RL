import numpy as np
import pandas as pd

class PortfolioEnvironment:
    def __init__(self, data, initial_balance=10000, transaction_cost=0.001):
        """
        Initialize the portfolio environment
        
        Args:
            data (pd.DataFrame): Historical price data for assets
            initial_balance (float): Initial amount of money
            transaction_cost (float): Transaction cost as a fraction
        """
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state"""
        self.current_step = 10  # Start after initial window to have enough history
        self.balance = self.initial_balance
        self.portfolio = np.ones(len(self.data.columns)) / len(self.data.columns)  # Equal weight initial portfolio
        self.done = False
        self.portfolio_value_history = [self.initial_balance]
        
        return self._get_state()
    
    def step(self, action):
        """
        Take an action in the environment
        
        Args:
            action (np.array): Portfolio weights for each asset
        
        Returns:
            tuple: (next_state, reward, done, info)
        """
        if self.done:
            return self._get_state(), 0, True, {
                'portfolio_value': self.portfolio_value_history[-1],
                'transaction_costs': 0
            }

        old_portfolio_value = self._get_portfolio_value()
        
        # Ensure action is valid
        action = np.array(action).clip(0, 1)  # Clip weights between 0 and 1
        action_sum = np.sum(action)
        if action_sum > 0:
            action = action / action_sum  # Normalize to sum to 1
        else:
            action = np.ones_like(action) / len(action)  # Equal weights if invalid
        
        # Calculate transaction costs
        portfolio_change = np.abs(self.portfolio - action)
        transaction_costs = np.sum(portfolio_change) * self.transaction_cost * old_portfolio_value
        
        # Update portfolio
        self.portfolio = action
        self.current_step += 1
        
        # Calculate new portfolio value
        new_portfolio_value = self._get_portfolio_value() - transaction_costs
        self.portfolio_value_history.append(new_portfolio_value)
        
        # Calculate reward (daily returns with penalty for large changes)
        returns = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
        transaction_penalty = np.sum(portfolio_change) * self.transaction_cost
        reward = returns - transaction_penalty
        
        # Clip reward to prevent extreme values
        reward = np.clip(reward, -1, 1)
        
        # Check if episode is done
        self.done = self.current_step >= len(self.data) - 1
        
        return self._get_state(), reward, self.done, {
            'portfolio_value': new_portfolio_value,
            'transaction_costs': transaction_costs
        }
    
    def _get_portfolio_value(self):
        """Calculate current portfolio value"""
        if self.current_step >= len(self.data):
            return self.portfolio_value_history[-1]
        
        current_prices = self.data.iloc[self.current_step]
        return np.sum(self.portfolio * current_prices) * self.balance
    
    def _get_state(self):
        """Get current state of the environment"""
        # Use returns and normalized prices of last 10 days as state
        prices = self.data.iloc[self.current_step-10:self.current_step]
        returns = prices.pct_change().fillna(0).values
        
        # Add current portfolio weights to state
        portfolio_state = np.tile(self.portfolio, (10, 1))
        
        # Combine returns and portfolio state
        state = np.concatenate([returns, portfolio_state], axis=1)
        
        # Ensure no NaN or infinite values
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        return state