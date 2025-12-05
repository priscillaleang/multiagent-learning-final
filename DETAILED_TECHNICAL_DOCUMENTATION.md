================================================================================
DETAILED TECHNICAL DOCUMENTATION
MIT 6.S890 - NFSP vs PSRO Implementation
================================================================================

This document provides complete technical specifications for the NFSP and PSRO
implementation project. It includes architecture, pseudocode, hyperparameters,
and experimental protocols from the course materials.

================================================================================
PART 1: THEORETICAL FOUNDATIONS
================================================================================

## 1.1 Classical Algorithm Foundations

### Fictitious Play (NFSP's Foundation) - Lecture 5

Fictitious Play is a game-theoretic learning model where players choose best
responses to their opponents' **historical average behavior**.

**Key Properties:**
- At time t, player i computes: σ̄₋ᵢᵗ = (1/t) Σₛ₌₁ᵗ σ₋ᵢˢ (opponent's average strategy)
- Player i then plays: βᵢᵗ⁺¹ = BR(σ̄₋ᵢᵗ) (best response to average)
- In two-player zero-sum games, average strategies converge to Nash equilibrium

**Convergence Guarantee:**
Exploitability(σ̄ᵀ) = O(1/√T)

### Double Oracle (PSRO's Foundation) - Lecture 12

The Double Oracle algorithm maintains a restricted game over a population of
pure strategies and iteratively expands by computing best responses to the
current meta-Nash equilibrium.

**Key Properties:**
- Maintain population of policies: Π = {π₁, π₂, ..., πₖ}
- Solve restricted Nash equilibrium: σ* = Nash(M_Π) where M is the payoff matrix
- Add best response: Π ← Π ∪ {BR(σ*)}
- Guaranteed to converge to Nash equilibrium of full game

**Convergence Guarantee:** Finite convergence in games with finite pure strategy sets.

## 1.2 Deepification Strategy - Lecture 12

Both algorithms are "deepified" by using deep reinforcement learning to approximate
best responses:

| Classical Algorithm | Deep Version | Best Response Approximation |
|--------------------|--------------|-----------------------------|
| Fictitious Play | NFSP | DQN + Supervised Learning |
| Double Oracle | PSRO | Deep RL (DQN/PPO) |
| CFR | Deep CFR | Deep Learning for Regret |

**Core Insight from Course:** Use deep RL to approximate the expensive best response
computation while maintaining the theoretical structure that guarantees convergence.

================================================================================
PART 2: NFSP ALGORITHM - COMPLETE SPECIFICATION
================================================================================

## 2.1 Algorithm Overview

NFSP combines Fictitious Self-Play with neural network function approximation.
Each agent maintains:

1. **Q-Network Q(s,a|θ_Q):** Learns approximate best response via DQN
2. **Average Policy Network Π(s,a|θ_Π):** Learns historical average strategy via SL
3. **RL Memory M_RL:** Circular buffer for off-policy RL transitions
4. **SL Memory M_SL:** Reservoir buffer for behavior cloning data

## 2.2 Neural Network Architecture

### Q-Network (Best Response)
```
Input: Information state encoding s ∈ ℝᵈ
Architecture:
  - Input Layer: d neurons (d = state dimension)
  - Hidden Layer 1: 128 neurons, ReLU activation
  - Hidden Layer 2: 128 neurons, ReLU activation
  - Output Layer: |A| neurons (one per action)
Output: Q(s,a) for all actions a ∈ A
```

### Average Policy Network
```
Input: Information state encoding s ∈ ℝᵈ
Architecture:
  - Input Layer: d neurons
  - Hidden Layer 1: 128 neurons, ReLU activation
  - Hidden Layer 2: 128 neurons, ReLU activation
  - Output Layer: |A| neurons with Softmax
Output: Π(a|s) probability distribution over actions
```

## 2.3 Detailed Algorithm Pseudocode

```python
class NFSPAgent:
    def __init__(self, state_dim, action_dim, params):
        # Networks
        self.Q_network = MLP(state_dim, action_dim, hidden=[128, 128])
        self.Q_target = copy(self.Q_network)
        self.Pi_network = MLP(state_dim, action_dim, hidden=[128, 128], softmax=True)
        
        # Replay Buffers
        self.M_RL = CircularBuffer(capacity=params.rl_buffer_size)  # 2×10⁵
        self.M_SL = ReservoirBuffer(capacity=params.sl_buffer_size)  # 2×10⁶
        
        # Hyperparameters
        self.eta = params.anticipatory_param  # 0.1
        self.epsilon = params.exploration_rate  # starts at 0.06, decays
        self.lr_rl = params.learning_rate_rl    # 0.1
        self.lr_sl = params.learning_rate_sl    # 0.005
        
    def select_action(self, state):
        """Select action using anticipatory dynamics mixture"""
        if random() < self.eta:
            # With probability η: use ε-greedy best response
            if random() < self.epsilon:
                return random_action()
            else:
                return argmax(self.Q_network(state))
        else:
            # With probability (1-η): use average policy
            return sample(self.Pi_network(state))
    
    def store_transition(self, s, a, r, s_next, used_best_response):
        """Store experience in appropriate buffers"""
        # Always store in RL buffer
        self.M_RL.add((s, a, r, s_next))
        
        # Only store in SL buffer when using best response policy
        if used_best_response:
            self.M_SL.add((s, a))  # Reservoir sampling
    
    def train_step(self):
        """Perform one training update for both networks"""
        # === DQN Update for Q-network ===
        batch_rl = self.M_RL.sample(batch_size=128)
        s, a, r, s_next = batch_rl
        
        # Double DQN target computation
        a_next = argmax(self.Q_network(s_next))
        y = r + γ * self.Q_target(s_next)[a_next]
        
        # Q-network loss
        loss_rl = MSE(self.Q_network(s)[a], y)
        self.Q_network.update(loss_rl, lr=self.lr_rl)
        
        # === Supervised Learning Update for Π-network ===
        batch_sl = self.M_SL.sample(batch_size=128)
        s_sl, a_sl = batch_sl
        
        # Cross-entropy loss for behavior cloning
        loss_sl = CrossEntropy(self.Pi_network(s_sl), a_sl)
        self.Pi_network.update(loss_sl, lr=self.lr_sl)
        
        # Periodically update target network
        if self.step_count % 300 == 0:
            self.Q_target = copy(self.Q_network)
```

## 2.4 Key Hyperparameters

| Parameter | Symbol | Recommended Value | Notes |
|-----------|--------|-------------------|-------|
| Anticipatory parameter | η | 0.1 | Controls BR vs. average policy mixture |
| RL buffer size | |M_RL| | 2×10⁵ | Circular buffer |
| SL buffer size | |M_SL| | 2×10⁶ | Reservoir sampling |
| RL learning rate | α_RL | 0.1 | For Q-network |
| SL learning rate | α_SL | 0.005 | For Π-network |
| Batch size | B | 128 | For both networks |
| Target update frequency | - | 300 | Steps between target network updates |
| Initial exploration | ε₀ | 0.06 | ε-greedy exploration |
| Exploration decay | - | 0.9999 | ε *= decay per episode |
| Discount factor | γ | 1.0 | For two-player zero-sum games |

================================================================================
PART 3: PSRO ALGORITHM - COMPLETE SPECIFICATION
================================================================================

## 3.1 Algorithm Overview

PSRO (Policy Space Response Oracles) generalizes the Double Oracle algorithm by:
1. Maintaining a **population of policies** for each player
2. Using a **Meta-Strategy Solver (MSS)** to compute the target mixture
3. Training **approximate best responses** via deep RL
4. Building an **empirical payoff matrix** through simulation

## 3.2 Detailed Algorithm Pseudocode

```python
class PSROFramework:
    def __init__(self, game, params):
        self.game = game
        self.num_players = game.num_players
        
        # Policy populations (one per player)
        self.populations = [
            [RandomPolicy()] for _ in range(self.num_players)
        ]
        
        # Empirical game payoff matrix
        self.payoff_matrix = {}
        
        # Meta-strategy solver choice
        self.meta_solver = params.meta_solver  # 'nash', 'uniform', 'alpha_rank'
        
        # Best response training parameters
        self.br_training_steps = params.br_training_steps  # e.g., 50000
    
    def compute_meta_strategy(self):
        """Compute mixture over current population using MSS"""
        # Build payoff matrix for current populations
        M = self.build_payoff_matrix()
        
        if self.meta_solver == 'nash':
            # Solve for Nash equilibrium of empirical game
            sigma = solve_nash_equilibrium(M)
        elif self.meta_solver == 'uniform':
            # Equal weight on all policies
            sigma = [uniform(len(pop)) for pop in self.populations]
        
        return sigma
    
    def train_best_response(self, player_idx, opponent_mixture):
        """Train a new policy as best response to opponent mixture"""
        # Initialize new policy network
        new_policy = PolicyNetwork(self.game.state_dim, self.game.action_dim)
        
        # Create opponent that samples from mixture
        opponent_sampler = MixtureSampler(
            self.populations[1 - player_idx], 
            opponent_mixture
        )
        
        # Train using DQN
        for step in range(self.br_training_steps):
            # Collect experience against mixture opponent
            trajectory = collect_trajectory(new_policy, opponent_sampler, self.game)
            
            # Update policy using RL
            dqn_update(new_policy, trajectory)
        
        return new_policy
    
    def expand_populations(self, meta_strategies):
        """Add best response policies to each player's population"""
        for player in range(self.num_players):
            # Compute best response to opponent's meta-strategy
            opponent_mixture = meta_strategies[1 - player]
            new_policy = self.train_best_response(player, opponent_mixture)
            
            # Add to population
            self.populations[player].append(new_policy)
    
    def update_payoff_matrix(self):
        """Estimate payoffs for new policy combinations"""
        for p1_idx in range(len(self.populations[0])):
            for p2_idx in range(len(self.populations[1])):
                if (p1_idx, p2_idx) not in self.payoff_matrix:
                    # Estimate expected payoff through simulation
                    payoff = estimate_payoff(
                        self.populations[0][p1_idx],
                        self.populations[1][p2_idx],
                        self.game,
                        num_episodes=1000
                    )
                    self.payoff_matrix[(p1_idx, p2_idx)] = payoff
```

## 3.3 Key Hyperparameters

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| BR training steps | 5×10⁴ | Steps to train each best response |
| Payoff estimation episodes | 1000 | Episodes to estimate each payoff entry |
| Meta-solver | Nash | For 2-player zero-sum games |
| RL algorithm | DQN | Standard choice |
| Learning rate (BR) | 1×10⁻³ | For DQN best response training |
| Network architecture | [128, 128] | Two hidden layers |

================================================================================
PART 4: EXPLOITABILITY COMPUTATION
================================================================================

## 4.1 Definition

For a strategy profile σ in a two-player zero-sum game, exploitability measures
the distance from Nash equilibrium:

ε(σ) = 0.5 * [max_σ'₁ u₁(σ'₁, σ₂) - u₁(σ₁, σ₂) + max_σ'₂ u₂(σ₁, σ'₂) - u₂(σ₁, σ₂)]

Equivalently:
ε(σ) = 0.5 * Σᵢ [u_i(BR_i(σ₋ᵢ), σ₋ᵢ) - u_i(σ)]

## 4.2 Exact Computation (Small Games)

For games where the full game tree can be enumerated (Kuhn Poker, Leduc Poker):

```python
def compute_exact_exploitability(strategy_profile, game):
    """Compute exact exploitability using full tree traversal"""
    total_exploitability = 0
    
    for player in range(game.num_players):
        # Compute best response value via backward induction
        br_value = compute_best_response_value(
            player, 
            strategy_profile, 
            game
        )
        
        # Compute current expected value
        current_value = compute_expected_value(
            player, 
            strategy_profile, 
            game
        )
        
        total_exploitability += br_value - current_value
    
    return total_exploitability / 2
```

## 4.3 Approximate Exploitability (Large Games)

For larger games, use learned approximate best response:

```python
def compute_approximate_exploitability(strategy_profile, game, params):
    """Approximate exploitability via learned best responses"""
    total_exp = 0
    
    for player in range(game.num_players):
        # Train approximate best response network
        br_network = train_best_response_network(
            player,
            strategy_profile,
            game,
            training_steps=params.br_training_steps
        )
        
        # Estimate values through simulation
        br_value = estimate_value(br_network, strategy_profile, game)
        current_value = estimate_value(strategy_profile, strategy_profile, game)
        
        total_exp += br_value - current_value
    
    return total_exp / 2
```

================================================================================
PART 5: STATE ENCODINGS FOR BENCHMARK GAMES
================================================================================

## Kuhn Poker (30 dimensions)

- Cards: 6-dim one-hot (J, Q, K × 2 cards each)
- Betting history: 24-dim tensor (2 players × 2 rounds × 3 raises × 2 actions)

Encoding: Concatenate one-hot card + betting history

## Leduc Poker (50 dimensions)

- Private card: 6-dim one-hot
- Public card: 6-dim one-hot (or zeros if not revealed)
- Betting history: Variable length, pad to 38-dim

Encoding: Concatenate all three components

## Liar's Dice (20 dimensions)

- Own dice: One-hot encoding of dice configuration (5 dim)
- Opponent claim history: Sequence of (face, count) claims (15 dim)

Encoding: Concatenate dice state + claim history

## Dark Hex (18 dimensions)

- Own pieces: Board-sized one-hot encoding (8 dim)
- Revealed opponent pieces: Board-sized encoding (8 dim)
- Move history: Game phase encoding (2 dim)

Encoding: Concatenate all components

================================================================================
PART 6: EXPERIMENTAL EXECUTION PLAN
================================================================================

## 6.1 Phase 1: Environment Setup (Week 1)

### Task 1.1: OpenSpiel Installation
```bash
pip install open_spiel
pip install tensorflow
pip install numpy pandas matplotlib seaborn
```

### Task 1.2: Verify Game Implementations
```python
import pyspiel

# Test games
games = [
    'kuhn_poker',
    'leduc_poker', 
    'liars_dice',
    'dark_hex(board_size=2)'
]

for game_name in games:
    game = pyspiel.load_game(game_name)
    print(f"{game_name}: {game.num_players()} players, "
          f"{game.num_distinct_actions()} actions")
```

## 6.2 Phase 2: NFSP Implementation (Week 2)

### Task 2.1: Implement Core Components
- CircularBuffer class for M_RL
- ReservoirBuffer class for M_SL with proper sampling
- MLP networks with appropriate architectures

### Task 2.2: Implement Training Loop
- Action selection with η mixture
- DQN updates with target network
- Supervised learning updates
- Exploration decay schedule

### Task 2.3: Verification on Kuhn Poker
- Target: Exploitability < 0.1 after 10⁶ episodes
- Compare to known results from Heinrich & Silver (2016)

## 6.3 Phase 3: PSRO Implementation (Week 3)

### Task 3.1: Implement Core Components
- Policy population management
- Payoff matrix estimation
- Nash equilibrium solver for empirical game

### Task 3.2: Implement Best Response Oracle
- DQN-based best response training
- Mixture opponent sampling
- Proper opponent policy loading

### Task 3.3: Verification on Kuhn Poker
- Target: Exploitability decreasing with iterations
- Population should discover key strategies

## 6.4 Phase 4: Main Experiments (Weeks 4-5)

### Experiment Matrix

| Game | NFSP Seeds | PSRO Seeds | Metrics |
|------|------------|------------|---------|
| Kuhn Poker | 5 | 5 | Exploitability, time, samples |
| Leduc Poker | 5 | 5 | Exploitability, time, samples |
| Liar's Dice | 5 | 5 | Exploitability, time, samples |
| Dark Hex 2×2 | 5 | 5 | Exploitability, time, samples |

### Logging Protocol
```python
def run_experiment(algorithm, game, seed, params):
    """Run single experiment with comprehensive logging"""
    set_seed(seed)
    
    logs = {
        'exploitability': [],
        'wall_time': [],
        'samples': [],
        'episode': []
    }
    
    start_time = time.time()
    
    for episode in range(params.total_episodes):
        # Training step
        train_step(algorithm, game)
        
        # Periodic evaluation
        if episode % params.eval_freq == 0:
            exp = compute_exploitability(algorithm, game)
            logs['exploitability'].append(exp)
            logs['wall_time'].append(time.time() - start_time)
            logs['samples'].append(algorithm.total_samples)
            logs['episode'].append(episode)
    
    save_logs(logs, f'{algorithm}_{game}_{seed}.csv')
```

## 6.5 Phase 5: Analysis (Weeks 6-7)

### Analysis 1: Sample Efficiency
- Plot exploitability vs. environment samples
- Compute samples to reach ε ∈ {0.1, 0.05, 0.01}

### Analysis 2: Computational Cost
- Wall-clock time comparison
- Memory usage profiling
- Neural network forward pass counts

### Analysis 3: Game Characteristics
- Correlate game complexity (info states, tree depth) with performance
- Identify regimes where each algorithm excels

### Statistical Tests
```python
from scipy import stats

def compare_algorithms(nfsp_results, psro_results, metric='final_exploitability'):
    """Statistical comparison of algorithm performance"""
    nfsp_values = [r[metric] for r in nfsp_results]
    psro_values = [r[metric] for r in psro_results]
    
    # Paired t-test
    t_stat, p_value = stats.ttest_ind(nfsp_values, psro_values)
    
    # Effect size (Cohen's d)
    cohens_d = (np.mean(nfsp_values) - np.mean(psro_values)) / \
               np.sqrt((np.var(nfsp_values) + np.var(psro_values)) / 2)
    
    return {'t_stat': t_stat, 'p_value': p_value, 'cohens_d': cohens_d}
```

================================================================================
PART 7: EXPECTED RESULTS AND HYPOTHESES
================================================================================

## 7.1 Primary Hypotheses

**H1 (Sample Efficiency):** NFSP will show better sample efficiency in games
with larger state spaces due to its continuous learning approach, while PSRO
will be more sample efficient in smaller games where exact best responses
provide more information per iteration.

**H2 (Final Exploitability):** Both algorithms will converge to similar final
exploitability levels, as both are based on game-theoretically sound foundations
(fictitious play and double oracle respectively).

**H3 (Computational Cost):** PSRO will have higher per-iteration computational
cost due to full best response training, but may require fewer total iterations.
NFSP will have lower per-step cost but require more total steps.

**H4 (Game Complexity Scaling):** NFSP will scale better with game tree size
due to its sample-based nature, while PSRO will struggle as the payoff matrix grows.

## 7.2 Success Criteria

| Metric | Kuhn Poker | Leduc Poker | Liar's Dice | Dark Hex |
|--------|------------|-------------|-------------|----------|
| Target ε | 0.01 | 0.05 | 0.1 | 0.1 |
| Max episodes | 10⁶ | 10⁷ | 10⁷ | 10⁷ |
| Seeds | 5 | 5 | 5 | 5 |

================================================================================

End of Technical Documentation
