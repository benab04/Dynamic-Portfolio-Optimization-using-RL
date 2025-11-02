import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PORTFOLIO ENVIRONMENT
# ============================================================================
class PortfolioEnvironment:
    """
    Portfolio Environment for Reinforcement Learning
    
    STATE SPACE:
    - Portfolio holdings (shares of each stock)
    - Current prices
    - Price momentum (5-day return for each asset)
    - Available cash
    
    ACTION SPACE:
    - For each stock: BUY, SELL, or HOLD
    - Action is represented as an integer that maps to a combination
    - Example for 4 stocks: action 0 = [HOLD, HOLD, HOLD, HOLD]
    -                       action 1 = [BUY, HOLD, HOLD, HOLD]
    -                       action 2 = [SELL, HOLD, HOLD, HOLD]
    - Total actions = 3^n_assets (e.g., 3^4 = 81 for 4 stocks)
    
    REWARD:
    - Change in portfolio value (holdings + cash) minus transaction costs
    """
    
    def __init__(self, data, initial_capital=10000, transaction_cost=0.001, 
                 trade_fraction=0.25):
        """
        Args:
            data: Stock price data
            initial_capital: Starting cash
            transaction_cost: Cost per trade (fraction of trade value)
            trade_fraction: Fraction of available amount to trade (0.25 = 25%)
        """
        self.data = data
        self.returns = data.pct_change().fillna(0)
        self.n_assets = len(data.columns)
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.trade_fraction = trade_fraction
        
        # Action space: 3 actions (BUY=0, SELL=1, HOLD=2) per stock
        self.action_types = ['BUY', 'SELL', 'HOLD']
        self.n_actions = 3 ** self.n_assets
        
        # State discretization parameters
        self.n_holding_bins = 4  # bins: none, low, medium, high
        self.n_cash_bins = 3     # bins: low, medium, high cash
        self.n_trend_bins = 3    # bins: negative, neutral, positive
        
        # Calculate state space size
        self.n_states = (self.n_holding_bins ** self.n_assets) * self.n_cash_bins * (self.n_trend_bins ** self.n_assets)
        
        self.reset()
    
    def _action_to_trades(self, action):
        """Convert action integer to list of trade decisions"""
        trades = []
        for _ in range(self.n_assets):
            trades.append(action % 3)
            action //= 3
        return trades  # 0=BUY, 1=SELL, 2=HOLD
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 20  # Start after warmup
        self.cash = self.initial_capital
        self.holdings = np.zeros(self.n_assets)  # Number of shares held
        self.portfolio_value_history = [self.initial_capital]
        return self._get_state()
    
    def _get_portfolio_value(self):
        """Calculate total portfolio value (holdings + cash)"""
        if self.current_step >= len(self.data):
            return self.portfolio_value_history[-1]
        
        current_prices = self.data.iloc[self.current_step].values
        holdings_value = np.sum(self.holdings * current_prices)
        return holdings_value + self.cash
    
    def _get_state(self):
        """Get current state representation"""
        current_prices = self.data.iloc[self.current_step].values
        
        # Discretize holdings (as fraction of total value)
        total_value = self._get_portfolio_value()
        holding_values = self.holdings * current_prices
        holding_fractions = holding_values / total_value if total_value > 0 else np.zeros(self.n_assets)
        holding_bins = np.digitize(holding_fractions, bins=[0.01, 0.25, 0.5])  # none, low, med, high
        
        # Discretize cash level
        cash_fraction = self.cash / total_value if total_value > 0 else 0
        cash_bin = np.digitize(cash_fraction, bins=[0.2, 0.5])  # low, med, high
        
        # Calculate momentum
        lookback = 5
        start_idx = max(0, self.current_step - lookback)
        momentum = []
        for col in self.data.columns:
            prices = self.data[col].iloc[start_idx:self.current_step+1]
            if len(prices) > 1:
                trend = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
            else:
                trend = 0
            momentum.append(trend)
        
        momentum = np.array(momentum)
        trend_bins = np.digitize(momentum, bins=[-0.02, 0.02])  # neg, neutral, pos
        
        # Combine into state
        state = tuple(holding_bins) + (cash_bin,) + tuple(trend_bins)
        
        # Convert to single integer
        state_int = 0
        multiplier = 1
        for s in reversed(state):
            state_int += s * multiplier
            multiplier *= 5  # Max bin value
        
        return state_int % self.n_states
    
    def step(self, action):
        """Execute trading action"""
        trades = self._action_to_trades(action)  # [0,1,2,...] for each stock
        
        old_value = self._get_portfolio_value()
        current_prices = self.data.iloc[self.current_step].values
        
        total_transaction_cost = 0
        
        # Execute trades for each stock
        for i, trade in enumerate(trades):
            price = current_prices[i]
            
            if trade == 0:  # BUY
                # Buy with a fraction of available cash
                max_buy_value = self.cash * self.trade_fraction
                max_shares = max_buy_value / price
                shares_to_buy = int(max_shares)
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    transaction_fee = cost * self.transaction_cost
                    total_cost = cost + transaction_fee
                    
                    if self.cash >= total_cost:
                        self.holdings[i] += shares_to_buy
                        self.cash -= total_cost
                        total_transaction_cost += transaction_fee
            
            elif trade == 1:  # SELL
                # Sell a fraction of holdings
                shares_to_sell = int(self.holdings[i] * self.trade_fraction)
                
                if shares_to_sell > 0:
                    revenue = shares_to_sell * price
                    transaction_fee = revenue * self.transaction_cost
                    
                    self.holdings[i] -= shares_to_sell
                    self.cash += (revenue - transaction_fee)
                    total_transaction_cost += transaction_fee
            
            # trade == 2 is HOLD, do nothing
        
        # Move to next time step
        self.current_step += 1
        
        # Calculate new portfolio value and reward
        new_value = self._get_portfolio_value()
        
        # Reward = change in portfolio value (accounts for price movements)
        value_change = new_value - old_value
        reward = (value_change / old_value) if old_value > 0 else 0
        
        # Penalize transaction costs
        reward -= (total_transaction_cost / old_value) if old_value > 0 else 0
        
        self.portfolio_value_history.append(new_value)
        
        # Check if done
        done = self.current_step >= len(self.data) - 1
        
        next_state = self._get_state()
        
        info = {
            'portfolio_value': new_value,
            'cash': self.cash,
            'holdings': self.holdings.copy(),
            'transaction_cost': total_transaction_cost
        }
        
        return next_state, reward, done, info

# ============================================================================
# POLICY ITERATION ALGORITHM
# ============================================================================
class PolicyIterationAgent:
    """
    Policy Iteration Agent using Dynamic Programming
    
    Policy Iteration alternates between:
    1. Policy Evaluation: Compute V(s) for current policy
    2. Policy Improvement: Update policy to be greedy w.r.t. V(s)
    
    This converges to optimal policy in fewer iterations than value iteration
    """
    
    def __init__(self, n_states, n_actions, discount_factor=0.95, theta=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = discount_factor
        self.theta = theta  # Convergence threshold for policy evaluation
        
        # Initialize random policy
        self.policy = np.random.randint(0, n_actions, size=n_states)
        
        # Initialize value function
        self.V = np.zeros(n_states)
        
        # Store transition data: transitions[s][a] = [(next_s, reward, count)]
        self.transitions = [[{} for _ in range(n_actions)] for _ in range(n_states)]
        self.state_visits = np.zeros(n_states)
        
    def get_action(self, state, explore=False, epsilon=0.1):
        """Get action from current policy"""
        if explore and np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        return self.policy[state]
    
    def update_transition(self, state, action, next_state, reward):
        """Update transition model from experience"""
        if next_state not in self.transitions[state][action]:
            self.transitions[state][action][next_state] = {'reward': 0, 'count': 0}
        
        trans = self.transitions[state][action][next_state]
        # Running average of reward
        trans['count'] += 1
        trans['reward'] += (reward - trans['reward']) / trans['count']
        
        self.state_visits[state] += 1
    
    def policy_evaluation(self):
        """
        Policy Evaluation: Compute state values for current policy
        Iteratively update V(s) until convergence
        """
        iteration = 0
        while True:
            delta = 0
            
            for s in range(self.n_states):
                if self.state_visits[s] == 0:
                    continue  # Skip unvisited states
                
                v = self.V[s]
                action = self.policy[s]
                
                # Calculate expected value
                new_v = 0
                total_count = sum(trans['count'] for trans in self.transitions[s][action].values())
                
                if total_count > 0:
                    for next_s, trans in self.transitions[s][action].items():
                        prob = trans['count'] / total_count
                        reward = trans['reward']
                        new_v += prob * (reward + self.gamma * self.V[next_s])
                
                self.V[s] = new_v
                delta = max(delta, abs(v - new_v))
            
            iteration += 1
            if delta < self.theta or iteration > 100:
                break
        
        return iteration
    
    def policy_improvement(self):
        """
        Policy Improvement: Update policy to be greedy w.r.t. current V
        Returns True if policy changed
        """
        policy_stable = True
        
        for s in range(self.n_states):
            if self.state_visits[s] == 0:
                continue
            
            old_action = self.policy[s]
            
            # Find best action
            action_values = np.zeros(self.n_actions)
            
            for a in range(self.n_actions):
                total_count = sum(trans['count'] for trans in self.transitions[s][a].values())
                
                if total_count > 0:
                    for next_s, trans in self.transitions[s][a].items():
                        prob = trans['count'] / total_count
                        reward = trans['reward']
                        action_values[a] += prob * (reward + self.gamma * self.V[next_s])
            
            # Greedy action selection
            best_action = np.argmax(action_values)
            self.policy[s] = best_action
            
            if old_action != best_action:
                policy_stable = False
        
        return policy_stable
    
    def train_iteration(self):
        """Perform one iteration of policy iteration"""
        eval_iters = self.policy_evaluation()
        policy_stable = self.policy_improvement()
        return policy_stable, eval_iters

# ============================================================================
# SARSA ALGORITHM
# ============================================================================
class SARSAAgent:
    """
    SARSA (State-Action-Reward-State-Action) Agent
    
    SARSA is an on-policy TD control algorithm:
    - Updates Q-values based on the action actually taken (including exploration)
    - Update rule: Q(s,a) = Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
    """
    
    def __init__(self, n_states, n_actions, learning_rate=0.1, 
                 discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table
        self.q_table = np.zeros((n_states, n_actions))
        
    def get_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, next_action):
        """SARSA update rule"""
        current_q = self.q_table[state, action]
        next_q = self.q_table[next_state, next_action]
        
        # SARSA update
        td_target = reward + self.gamma * next_q
        td_error = td_target - current_q
        self.q_table[state, action] = current_q + self.lr * td_error
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ============================================================================
# DATA FETCHING
# ============================================================================
@st.cache_data
def fetch_stock_data(tickers, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance"""
    # Download data for all tickers at once
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    # Extract Close prices
    if isinstance(data.columns, pd.MultiIndex):
        try:
            price_data = data['Adj Close']
        except KeyError:
            price_data = data['Close']
    else:
        try:
            price_data = data[['Adj Close']]
        except KeyError:
            price_data = data[['Close']]
        price_data.columns = tickers
    
    # Remove any rows with NaN values
    price_data = price_data.dropna()
    
    return price_data

# ============================================================================
# TRAINING FUNCTION
# ============================================================================
# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_policy_iteration(env, agent, n_episodes, progress_bar, status_text):
    """Train Policy Iteration agent"""
    episode_rewards = []
    episode_values = []
    best_value = 0
    best_history = []
    policy_iterations = []
    
    # Phase 1: Exploration to learn transition model
    exploration_episodes = max(10, n_episodes // 3)
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        # More exploration in early episodes
        explore = episode < exploration_episodes
        epsilon = 0.3 if explore else 0.05
        
        while not done:
            action = agent.get_action(state, explore=explore, epsilon=epsilon)
            next_state, reward, done, info = env.step(action)
            
            # Update transition model
            agent.update_transition(state, action, next_state, reward)
            
            total_reward += reward
            state = next_state
        
        # Perform policy iteration every few episodes (after enough exploration)
        if episode >= exploration_episodes and episode % 3 == 0:
            policy_stable, eval_iters = agent.train_iteration()
            policy_iterations.append(eval_iters)
        
        # Get final portfolio value
        final_value = env.portfolio_value_history[-1]
        
        episode_rewards.append(total_reward)
        episode_values.append(final_value)
        
        # Track best performance
        if final_value > best_value:
            best_value = final_value
            best_history = env.portfolio_value_history.copy()
        
        # Update progress
        progress = (episode + 1) / n_episodes
        progress_bar.progress(progress)
        
        if (episode + 1) % 5 == 0:
            avg_reward = np.mean(episode_rewards[-5:])
            avg_value = np.mean(episode_values[-5:])
            phase = "Exploring" if explore else "Optimizing"
            status_text.text(
                f"Episode {episode+1}/{n_episodes} ({phase}) | "
                f"Avg Reward: {avg_reward:.4f} | "
                f"Best Value: ${best_value:,.2f}"
            )
    
    return episode_rewards, episode_values, best_history

def train_sarsa(env, agent, n_episodes, progress_bar, status_text):
    """Train SARSA agent with progress tracking"""
    episode_rewards = []
    episode_values = []
    best_value = 0
    best_history = []
    
    for episode in range(n_episodes):
        state = env.reset()
        action = agent.get_action(state)
        total_reward = 0
        done = False
        
        while not done:
            next_state, reward, done, info = env.step(action)
            next_action = agent.get_action(next_state)
            agent.update(state, action, reward, next_state, next_action)
            
            total_reward += reward
            state = next_state
            action = next_action
        
        agent.decay_epsilon()
        
        # Get final portfolio value
        final_value = env.portfolio_value_history[-1]
        
        episode_rewards.append(total_reward)
        episode_values.append(final_value)
        
        # Track best performance
        if final_value > best_value:
            best_value = final_value
            best_history = env.portfolio_value_history.copy()
        
        # Update progress
        progress = (episode + 1) / n_episodes
        progress_bar.progress(progress)
        
        if (episode + 1) % 5 == 0:
            avg_reward = np.mean(episode_rewards[-5:])
            avg_value = np.mean(episode_values[-5:])
            status_text.text(
                f"Episode {episode+1}/{n_episodes} | "
                f"Avg Reward: {avg_reward:.4f} | "
                f"Best Value: ${best_value:,.2f} | "
                f"ε: {agent.epsilon:.3f}"
            )
    
    return episode_rewards, episode_values, best_history

# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_training_progress(episode_rewards, episode_values):
    """Plot training metrics"""
    fig = go.Figure()
    
    # Add rewards trace
    fig.add_trace(go.Scatter(
        x=list(range(len(episode_rewards))),
        y=episode_rewards,
        mode='lines',
        name='Episode Reward',
        line=dict(color='cyan', width=2)
    ))
    
    fig.update_layout(
        title='Training Rewards per Episode',
        xaxis_title='Episode',
        yaxis_title='Total Reward',
        template='plotly_dark',
        hovermode='x unified'
    )
    
    return fig

def calculate_baseline_strategies(stock_data, initial_capital, portfolio_history_length):
    """Calculate baseline strategies: equal weight and individual stocks"""
    baselines = {}
    
    # Equal weight strategy (buy and hold)
    equal_weight_values = [initial_capital]
    equal_weights = 1 / len(stock_data.columns)
    
    for i in range(1, min(portfolio_history_length, len(stock_data))):
        if i >= len(stock_data):
            break
        returns = stock_data.iloc[i] / stock_data.iloc[i-1] - 1
        portfolio_return = returns.mean()  # Equal weight
        equal_weight_values.append(equal_weight_values[-1] * (1 + portfolio_return))
    
    baselines['Equal Weight'] = equal_weight_values
    
    # Individual stock strategies (invest 100% in each stock)
    for stock in stock_data.columns:
        stock_values = [initial_capital]
        
        for i in range(1, min(portfolio_history_length, len(stock_data))):
            if i >= len(stock_data):
                break
            stock_return = stock_data[stock].iloc[i] / stock_data[stock].iloc[i-1] - 1
            stock_values.append(stock_values[-1] * (1 + stock_return))
        
        baselines[f'{stock} Only'] = stock_values
    
    return baselines

def plot_portfolio_performance(portfolio_history, initial_capital, stock_data):
    """Plot portfolio value over time with baseline comparisons"""
    fig = go.Figure()
    
    # Calculate baseline strategies
    baselines = calculate_baseline_strategies(stock_data, initial_capital, len(portfolio_history))
    
    # RL Agent
    fig.add_trace(go.Scatter(
        x=list(range(len(portfolio_history))),
        y=portfolio_history,
        mode='lines',
        name='RL Agent',
        line=dict(color='lime', width=4)
    ))
    
    # Equal weight baseline
    fig.add_trace(go.Scatter(
        x=list(range(len(baselines['Equal Weight']))),
        y=baselines['Equal Weight'],
        mode='lines',
        name='Equal Weight (Baseline)',
        line=dict(color='cyan', width=2, dash='dash')
    ))
    
    # Individual stock baselines
    colors = ['orange', 'pink', 'yellow', 'purple', 'red', 'blue']
    for idx, (stock_name, stock_values) in enumerate(baselines.items()):
        if stock_name == 'Equal Weight':
            continue
        
        color = colors[idx % len(colors)]
        fig.add_trace(go.Scatter(
            x=list(range(len(stock_values))),
            y=stock_values,
            mode='lines',
            name=stock_name,
            line=dict(color=color, width=1.5, dash='dot'),
            opacity=0.7
        ))
    
    # Initial capital line
    fig.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color="red",
        annotation_text="Initial Capital",
        line_width=1
    )
    
    fig.update_layout(
        title='Portfolio Performance: RL Agent vs Baseline Strategies',
        xaxis_title='Time Steps',
        yaxis_title='Portfolio Value ($)',
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

# ============================================================================
# STREAMLIT APP
# ============================================================================
def main():
    st.set_page_config(
        page_title="SARSA Portfolio Optimization",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Portfolio Optimization using SARSA")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Algorithm selection
    st.sidebar.subheader("Algorithm Selection")
    algorithm = st.sidebar.selectbox(
        "Choose Algorithm",
        ["Policy Iteration (Recommended)", "SARSA"],
        help="Policy Iteration uses dynamic programming with policy evaluation and improvement"
    )
    
    # Stock selection
    st.sidebar.subheader("Stock Selection")
    default_tickers = ['GOOGL', 'AAPL', 'MSFT', 'META']
    tickers_input = st.sidebar.text_input(
        "Stock Tickers (comma-separated)",
        value=', '.join(default_tickers)
    )
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    # Date range
    st.sidebar.subheader("Date Range")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    date_start = st.sidebar.date_input(
        "Start Date",
        value=start_date,
        max_value=datetime.now()
    )
    date_end = st.sidebar.date_input(
        "End Date",
        value=end_date,
        max_value=datetime.now()
    )
    
    # Environment parameters
    st.sidebar.subheader("Environment Parameters")
    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)",
        min_value=1000,
        max_value=1000000,
        value=100000,
        step=10000
    )
    
    transaction_cost = st.sidebar.slider(
        "Transaction Cost (%)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05
    ) / 100
    
    trade_fraction = st.sidebar.slider(
        "Trade Fraction (%)",
        min_value=10,
        max_value=100,
        value=25,
        step=5,
        help="Percentage of available amount to trade per action"
    ) / 100
    
    # SARSA/Policy Iteration parameters
    st.sidebar.subheader("Algorithm Parameters")
    
    if "Policy Iteration" in algorithm:
        discount_factor = st.sidebar.slider(
            "Discount Factor (γ)",
            min_value=0.5,
            max_value=0.99,
            value=0.95,
            step=0.01
        )
        
        convergence_threshold = st.sidebar.slider(
            "Convergence Threshold (θ)",
            min_value=0.001,
            max_value=0.1,
            value=0.01,
            step=0.001,
            format="%.3f"
        )
    else:  # SARSA
        learning_rate = st.sidebar.slider(
            "Learning Rate (α)",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01
        )
        
        discount_factor = st.sidebar.slider(
            "Discount Factor (γ)",
            min_value=0.5,
            max_value=0.99,
            value=0.95,
            step=0.01
        )
    
    n_episodes = st.sidebar.slider(
        "Training Episodes",
        min_value=10,
        max_value=200,
        value=50,
        step=10
    )
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Selected Stocks")
        st.write(", ".join(tickers))
    
    with col2:
        st.subheader("📅 Date Range")
        st.write(f"{date_start} to {date_end}")
    
    # Information boxes
    st.markdown("---")
    
    with st.expander("ℹ️ About the Algorithms"):
        st.markdown("""
        ### Policy Iteration (Recommended)
        
        **Policy Iteration** is a dynamic programming algorithm that alternates between:
        
        1. **Policy Evaluation**: Calculate the value V(s) of each state under the current policy
           - Iteratively updates state values until convergence
           - Uses the Bellman expectation equation
        
        2. **Policy Improvement**: Update the policy to be greedy with respect to V(s)
           - For each state, select the action that maximizes expected return
           - Guaranteed to improve or maintain policy quality
        
        **Advantages**:
        - Converges to optimal policy in fewer iterations
        - More stable learning than TD methods
        - Works well with learned transition models
        
        **Process**:
        - Phase 1: Exploration to learn state transitions and rewards
        - Phase 2: Repeated policy evaluation and improvement until convergence
        
        ---
        
        ### SARSA
        
        **SARSA (State-Action-Reward-State-Action)** is an on-policy TD learning algorithm.
        
        **Update Rule**: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        
        **Characteristics**:
        - On-policy: Learns from actions actually taken
        - More conservative than Q-learning
        - Simpler but may require more episodes to converge
        """)
    
    with st.expander("🎯 State and Action Space"):
        st.markdown(f"""
        **States**: 
        - Current holdings (shares of each stock)
        - Available cash level
        - Price momentum (5-day trend for each asset)
        
        **Actions**: 
        - For each stock: **BUY**, **SELL**, or **HOLD**
        - Total actions = 3^(number of stocks) = **{3**len(tickers)} actions**
        
        **Trade Fraction**: {trade_fraction*100:.0f}%
        - BUY: Uses {trade_fraction*100:.0f}% of available cash
        - SELL: Sells {trade_fraction*100:.0f}% of holdings
        - HOLD: No trade, no transaction costs
        """)
    
    # Training button
    st.markdown("---")
    
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        try:
            # Fetch data
            with st.spinner("Fetching stock data..."):
                stock_data = fetch_stock_data(tickers, date_start, date_end)
                st.success(f"✅ Fetched {len(stock_data)} days of data")
            
            # Create environment
            env = PortfolioEnvironment(
                data=stock_data,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                trade_fraction=trade_fraction
            )
            
            # Display environment info
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Assets", env.n_assets)
            col2.metric("Actions", env.n_actions)
            col3.metric("States", env.n_states)
            col4.metric("Data Points", len(stock_data))
            
            # Create agent
            if "Policy Iteration" in algorithm:
                agent = PolicyIterationAgent(
                    n_states=env.n_states,
                    n_actions=env.n_actions,
                    discount_factor=discount_factor,
                    theta=convergence_threshold
                )
                agent_type = "Policy Iteration"
            else:
                agent = SARSAAgent(
                    n_states=env.n_states,
                    n_actions=env.n_actions,
                    learning_rate=learning_rate,
                    discount_factor=discount_factor
                )
                agent_type = "SARSA"
            
            # Training progress
            st.markdown("---")
            st.subheader(f"🔄 Training {agent_type}")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Train based on algorithm
            if "Policy Iteration" in algorithm:
                episode_rewards, episode_values, best_history = train_policy_iteration(
                    env, agent, n_episodes, progress_bar, status_text
                )
            else:
                episode_rewards, episode_values, best_history = train_sarsa(
                    env, agent, n_episodes, progress_bar, status_text
                )
            
            status_text.text("✅ Training completed!")
            
            # Results
            st.markdown("---")
            st.subheader("📈 Results")
            
            # Metrics
            final_capital = best_history[-1]
            total_return = (final_capital - initial_capital) / initial_capital * 100
            max_capital = max(best_history)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Final Capital", f"${final_capital:,.2f}")
            col2.metric("Total Return", f"{total_return:.2f}%")
            col3.metric("Max Capital", f"${max_capital:,.2f}")
            
            # Plots
            tab1, tab2, tab3 = st.tabs(["Portfolio Performance", "Training Progress", "Strategy Comparison"])
            
            with tab1:
                fig_portfolio = plot_portfolio_performance(best_history, initial_capital, stock_data)
                st.plotly_chart(fig_portfolio, use_container_width=True)
                
                # Show algorithm used
                st.info(f"📊 Best performance achieved using **{agent_type}**")
            
            with tab2:
                fig_training = plot_training_progress(episode_rewards, episode_values)
                st.plotly_chart(fig_training, use_container_width=True)
            
            with tab3:
                # Calculate baseline returns
                baselines = calculate_baseline_strategies(stock_data, initial_capital, len(best_history))
                
                # Create comparison table
                st.subheader("📊 Strategy Comparison")
                
                comparison_data = []
                
                # SARSA Agent
                sarsa_return = (best_history[-1] - initial_capital) / initial_capital * 100
                comparison_data.append({
                    'Strategy': 'RL Agent (' + agent_type + ')',
                    'Final Value': f"${best_history[-1]:,.2f}",
                    'Return (%)': f"{sarsa_return:.2f}%",
                    'Return': sarsa_return
                })
                
                # Baselines
                for strategy_name, values in baselines.items():
                    final_value = values[-1]
                    strategy_return = (final_value - initial_capital) / initial_capital * 100
                    comparison_data.append({
                        'Strategy': strategy_name,
                        'Final Value': f"${final_value:,.2f}",
                        'Return (%)': f"{strategy_return:.2f}%",
                        'Return': strategy_return
                    })
                
                # Sort by return
                comparison_data.sort(key=lambda x: x['Return'], reverse=True)
                
                # Display table
                comparison_df = pd.DataFrame(comparison_data)[['Strategy', 'Final Value', 'Return (%)']]
                
                # Highlight RL Agent row
                def highlight_agent(row):
                    if 'RL Agent' in row['Strategy']:
                        return ['background-color: #1a5f1a'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    comparison_df.style.apply(highlight_agent, axis=1),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Show winner
                winner = comparison_data[0]
                if 'RL Agent' in winner['Strategy']:
                    st.success(f"🎉 {winner['Strategy']} outperformed all baseline strategies with {winner['Return (%)']}")
                else:
                    st.info(f"ℹ️ Best performing strategy: {winner['Strategy']} with {winner['Return (%)']}")
                    agent_rank = next(i for i, d in enumerate(comparison_data) if 'RL Agent' in d['Strategy']) + 1
                    st.warning(f"{agent_type} ranked #{agent_rank} out of {len(comparison_data)} strategies")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()