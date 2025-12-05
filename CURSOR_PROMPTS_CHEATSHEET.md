================================================================================
CURSOR PROMPTS CHEATSHEET FOR RAPID IMPLEMENTATION
================================================================================

Use these prompts in Cursor IDE to generate code implementations quickly.
Copy-paste each prompt into a new Cursor tab and let it generate code.

================================================================================
PHASE 1: CORE UTILITIES
================================================================================

---
PROMPT 1.1: MLP Neural Network (TensorFlow)
---

You are implementing a deep reinforcement learning framework.

Task: Implement the MLP class for neural networks using TensorFlow/Keras.

File: src/networks/networks.py

Requirements:
1. Create class MLP(tf.keras.Model)
2. __init__ takes: input_dim, output_dim, hidden_dims=[128,128], activation='relu', use_softmax=False
3. Create dense layers with ReLU activation for hidden layers
4. Final layer has use_softmax=True for policies, False for Q-networks
5. Use He normal initializer for weights
6. Implement call(self, x, training=False) method
7. Support batch processing

Code template:
```python
import tensorflow as tf
from typing import List

class MLP(tf.keras.Model):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 128], 
                 activation='relu', use_softmax=False):
        super().__init__()
        self.layers_list = []
        # Create hidden layers
        # Create output layer
        self.use_softmax = use_softmax
    
    def call(self, x, training=False):
        # Apply layers sequentially
        pass
```

Generate complete implementation.

---
PROMPT 1.2: Circular and Reservoir Buffers
---

Implement two buffer classes for experience replay in RL.

File: src/networks/buffers.py

Requirements:
1. CircularBuffer class:
   - Fixed-size buffer using collections.deque
   - Methods: add(transition), sample(batch_size), clear(), size(), full()
   - Return numpy arrays of shape (batch_size, *feature_dims)
   - O(1) operations

2. ReservoirBuffer class:
   - Implements uniform reservoir sampling
   - Methods: add(item), sample(batch_size), size()
   - Maintains uniform distribution over all items ever added
   - O(1) per add, O(batch_size) per sample

Both classes should:
- Handle transitions as tuples (state, action, reward, next_state)
- Return numpy arrays
- Support max_size limits

Example usage:
```python
rl_buffer = CircularBuffer(200000)
rl_buffer.add((state, action, reward, next_state))
batch = rl_buffer.sample(128)  # Returns tuple of 4 arrays
```

Generate complete implementation with tests.

---
PROMPT 1.3: OpenSpiel GameWrapper
---

Create a unified interface for OpenSpiel games.

File: src/games/game_wrapper.py

Requirements:
1. GameWrapper class that wraps pyspiel.load_game()
2. **ADD VALIDATION in __init__:**
   - Verify game.get_type().provides_information_state_tensor == True
   - Verify game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL
   - Verify game.num_players() == 2
   - Warn if not zero-sum
   - Raise ValueError with helpful message if checks fail
3. Methods: reset(), step(), current_player(), legal_actions()
4. Handle chance nodes automatically
5. Use state_encoder for encoding

Auto-import state encoders:
```python
from src.games import state_encoder
obs = state_encoder.encode_state(state, self.game)
```

Skeleton Implementation
```
class GameWrapper:
    def __init__(self, game_name):
        self.game = pyspiel.load_game(game_name)
        
        # === VERIFY EXTENSIVE-FORM REQUIREMENTS ===
        game_type = self.game.get_type()
        
        # Check 1: Must provide information state
        if not game_type.provides_information_state_tensor:
            raise ValueError(f"{game_name} does not provide information_state_tensor. "
                           "NFSP/PSRO require extensive-form games with imperfect information.")
        
        # Check 2: Must be sequential (not simultaneous moves)
        if game_type.dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
            raise ValueError(f"{game_name} has simultaneous moves. "
                           "Use sequential extensive-form games instead.")
        
        # Check 3: Should be two-player zero-sum (for Nash convergence)
        if game_type.utility != pyspiel.GameType.Utility.ZERO_SUM:
            print(f"Warning: {game_name} is not zero-sum. "
                  "Convergence guarantees may not hold.")
        
        if self.game.num_players() != 2:
            raise ValueError(f"{game_name} has {self.game.num_players()} players. "
                           "This implementation supports 2-player games only.")
        
        # === INITIALIZE ===
        self.state_dim = self.game.information_state_tensor_size()
        self.action_dim = self.game.num_distinct_actions()
        self.num_players = self.game.num_players()

```



Generate complete wrapper class with documentation.

---
PROMPT 1.4: State Encoders for Each Game
---

Create state encoding functions for 4 games.

File: src/games/state_encoder.py

Implement encode_state(state, game) -> np.ndarray for:

1. kuhn_poker (30 dims):
   - 6-dim one-hot for cards (J, Q, K for each player)
   - 24-dim for betting history
   
2. leduc_poker (50 dims):
   - 6-dim one-hot private card
   - 6-dim one-hot public card (zeros if unrevealed)
   - 38-dim betting history (padded)

3. liars_dice (20 dims):
   - 5-dim own dice config
   - 15-dim opponent claim history

4. dark_hex (18 dims):
   - 8-dim own board state
   - 8-dim opponent board
   - 2-dim game phase

All should:
- Return np.float32 arrays
- Values normalized to [-1, 1] or [0, 1]
- Be deterministic
- Handle edge cases (unrevealed cards, etc.)

Generate encoder functions with input validation.

---
PROMPT 1.5: Exploitability Computation
---

Implement exploitability metrics for game theory evaluation.

File: src/utils/exploitability.py

Implement two functions:

1. compute_exact_exploitability(strategy_profile, game):
   - Use for Kuhn Poker (small game)
   - Backward induction to find best response values
   - strategy_profile = (policy1, policy2)
   - Return scalar exploitability

2. compute_approximate_exploitability(strategies, game, num_evals=1000):
   - Monte Carlo estimation
   - strategies = callable that returns action probabilities
   - Evaluate via self-play
   - Return exploitability estimate

Formula: ε = 0.5 * Σ[u_i(BR_i, σ_-i) - u_i(σ)]

Both should handle:
- Two-player zero-sum games
- Stochastic and deterministic strategies
- Error handling

Generate with docstrings and examples.

================================================================================
PHASE 2: NFSP ALGORITHM
================================================================================

---
PROMPT 2.1: NFSPAgent Class
---

Implement Neural Fictitious Self-Play agent.

File: src/algorithms/nfsp.py

Class: NFSPAgent(state_dim, action_dim, player_id, params)

Methods required:
1. __init__: Initialize Q-network, Pi-network, buffers, hyperparameters
2. select_action(state, training=True) -> (action, used_br):
   - With prob η: ε-greedy on Q-network
   - With prob 1-η: sample from Pi-network
3. store_transition(s, a, r, s', used_br):
   - Always add to M_RL
   - Add to M_SL only if used_br=True
4. train_step():
   - DQN update on M_RL (Double DQN with target network)
   - Supervised learning on M_SL (behavior cloning)
5. get_strategy(): Return callable policy
6. decay_exploration(): epsilon *= decay_rate

Hyperparameters (from params dict):
- eta=0.1, epsilon=0.06, lr_rl=0.1, lr_sl=0.005
- gamma=1.0, batch_size=128, target_update_freq=300

Generate complete class with docstrings.

---
PROMPT 2.2: NFSP Training Loop
---

Create training script for NFSP.

File: experiments/runners/train_nfsp.py

Main function: train_nfsp(game_name, num_episodes, seed, output_dir)

Structure:
1. Set random seed
2. Load game via GameWrapper
3. Create two NFSPAgent instances
4. Training loop (num_episodes):
   - Reset game state
   - While not done:
     - Current player selects action
     - Execute step
     - Store transition
     - Train step
   - Decay exploration
   - Every eval_freq: compute exploitability, log results

Output: CSV file to output_dir with columns:
[episode, exploitability, wall_time, samples, memory_mb]

Command line args:
--game, --num_episodes, --seed, --output_dir, --eval_freq

Include:
- tqdm progress bar
- File logging
- Checkpoint saving

Generate script with argument parsing and main function.

---
PROMPT 2.3: NFSP Unit Tests
---

Create integration tests for NFSP on Kuhn Poker.

File: tests/integration/test_nfsp_kuhn.py

Tests:
1. test_initialization: Create agents, verify dimensions
2. test_action_selection: Run 100 steps, verify legal actions
3. test_training_small: 1000 episodes, exploitability decreases
4. test_training_full: 10000 episodes, ε < 0.1 (slow test)

Use pytest, mark slow tests with @pytest.mark.slow

Generate test file.

================================================================================
PHASE 3: PSRO ALGORITHM
================================================================================

---
PROMPT 3.1: PSRO Framework (Part 1)
---

Implement Policy Space Response Oracles.

File: src/algorithms/psro.py

Classes:

1. PolicyPopulation:
   - Maintain list of policies
   - add_policy(policy)
   - get_mixed_strategy(mixture) -> callable
   - Properties: size, policies

2. MetaStrategySolver:
   - Base class with solve(payoff_matrix) method
   - NashSolver: scipy.optimize.linprog for zero-sum
   - UniformSolver: equal probability
   - Implement as separate classes

3. PSROFramework:
   - __init__(game, meta_solver='nash', params)
   - Maintain: populations, payoff_matrix
   - Key methods: (stubs first)
     - train_best_response(player_idx, opp_mixture)
     - compute_meta_strategy()
     - update_payoff_matrix()
     - expand_populations()

Start with stubs, full implementation in next prompt.

Generate class structure and docstrings.

---
PROMPT 3.2: PSRO Best Response Training
---

Implement best response oracle for PSRO.

File: src/algorithms/psro.py - Add to PSROFramework

Method: train_best_response(self, player_idx, opponent_mixture)

Requirements:
1. Create new PolicyNetwork
2. Create MixtureOpponent that samples from opponent policies
3. DQN training loop (br_training_steps):
   - Play episode against MixtureOpponent
   - Collect trajectory
   - DQN update

MixtureOpponent behavior:
- Sample policy from opponent_mixture at episode start
- Play full episode with that policy
- Sample new policy each episode

Return: trained PolicyNetwork

Generate implementation.

---
PROMPT 3.3: PSRO Meta-Strategy Solver
---

Implement Nash equilibrium solver for empirical game.

File: src/algorithms/psro.py - Add to PSROFramework

Method: compute_meta_strategy(self)

For 2-player zero-sum, solve linear program:
max_σ1 min_σ2: σ1ᵀ M σ2

Using scipy.optimize.linprog or cvxpy.

Return: (sigma1, sigma2) as numpy arrays

Handle:
- All-zero payoffs (uniform)
- Small populations
- Numerical stability

Generate solver implementation.

---
PROMPT 3.4: PSRO Training Loop
---

Create training script for PSRO.

File: experiments/runners/train_psro.py

Main function: train_psro(game_name, num_iterations, seed, output_dir)

Structure:
1. Set random seed
2. Load game
3. Create PSROFramework
4. For each iteration:
   - Compute meta-strategies
   - Expand populations
   - Update payoff matrix
   - Compute exploitability
   - Log and checkpoint

Output: CSV with columns:
[iteration, exploitability, wall_time, samples, pop_size_p1, pop_size_p2]

Args: --game, --num_iterations, --seed, --output_dir

Generate script with main function.

---
PROMPT 3.5: PSRO Unit Tests
---

Create integration tests for PSRO on Kuhn Poker.

File: tests/integration/test_psro_kuhn.py

Tests:
1. test_initialization: Create PSRO, verify populations
2. test_br_training: Train one BR, verify validity
3. test_meta_strategy: Compute strategy on simple matrix
4. test_full_run: 5 iterations, verify growth and convergence

Generate test file using pytest.

================================================================================
PHASE 4: CONFIGURATION & ORCHESTRATION
================================================================================

---
PROMPT 4.1: Create Config Files
---

Create YAML configuration files for 4 games.

Files: experiments/configs/kuhn.yaml, leduc.yaml, liarsdice.yaml, darkhex.yaml

Each file should contain:
- game: name
- nfsp: hyperparameters (num_episodes, learning rates, etc.)
- psro: hyperparameters (num_iterations, br_training_steps, etc.)
- logging: eval_freq, checkpoint_freq, log_to_stdout
- hardware: batch sizes

Base structure for kuhn.yaml:
```yaml
game:
  name: kuhn_poker
nfsp:
  num_episodes: 1000000
  eval_freq: 1000
  eta: 0.1
  epsilon_0: 0.06
  epsilon_decay: 0.9999
  lr_rl: 0.1
  lr_sl: 0.005
  gamma: 1.0
```

Scale leduc, liarsdice, darkhex appropriately by game complexity.

Generate 4 config files.

---
PROMPT 4.2: GCP Job Submission Script
---

Create Google Cloud job submission script.

File: experiments/gcp/submit_job.py

Functions:
1. create_job_config(algorithm, game, seed, output_bucket):
   - Create job configuration dict
   - Setup environment variables
   - Point to correct training script

2. submit_to_gcp(job_config):
   - Submit to GCP (Vertex AI or Compute Engine)
   - Return job_id

3. main():
   - Parse args: --algorithm, --game, --seeds (0-4), --output_bucket
   - Support --local flag to run locally
   - Loop through seeds, submit 5 jobs

Usage:
```bash
python experiments/gcp/submit_job.py \
  --algorithm nfsp \
  --game kuhn_poker \
  --seeds 0-4 \
  --output_bucket gs://my-bucket/results/
```

Generate script with GCS integration.

================================================================================
PHASE 5: ANALYSIS & PLOTTING
================================================================================

---
PROMPT 5.1: Results Aggregation
---

Create results aggregation script.

File: analysis/stats/aggregate_results.py

Function: aggregate_experiments(results_dir, output_dir)

Processing:
1. Read all CSV files from results_dir
2. For each game x algorithm:
   - Combine seed results (0-4)
   - Compute mean, std of exploitability
   - Find samples/time to reach ε in {0.1, 0.05, 0.01}
3. Output: summary_results.csv with columns:
   [game, algorithm, final_exp, final_exp_std,
    samples_0.1, samples_0.05, samples_0.01,
    time_0.1, time_0.05, time_0.01]

Generate script.

---
PROMPT 5.2: Plotting Functions
---

Create plotting functions for results visualization.

File: analysis/plots/plot_exploitability.py

Functions:
1. plot_exploitability_curves(results_dir, output_dir):
   - 2x2 subplot (one per game)
   - Each plot: NFSP vs PSRO
   - X-axis: samples (log)
   - Y-axis: exploitability (log)
   - Error bars from seed std
   - Save as PDF

2. plot_wall_time(results_dir, output_dir):
   - Same as above but X-axis = wall_time

3. plot_sample_efficiency_heatmap(results_dir, output_dir):
   - Table: games x algorithms
   - Cells: samples to ε=0.1
   - Color: lower is better
   - Highlight winner
   - Save as PNG

Generate with matplotlib/seaborn.

================================================================================
QUICK REFERENCE: Execution Order
================================================================================

Day 1 (Phase 0):
  - mkdir & git init
  - Create requirements.txt, setup.py, Makefile, pytest.ini

Days 1-2 (Phase 1):
  Use Prompts: 1.1 (networks), 1.2 (buffers), 1.3 (wrapper), 1.4 (encoders), 1.5 (exploitability)

Days 3-5 (Phase 2):
  Use Prompts: 2.1 (NFSP agent), 2.2 (training loop), 2.3 (tests)
  Milestone: NFSP works on Kuhn Poker with exploitability < 0.1

Days 6-8 (Phase 3):
  Use Prompts: 3.1 (framework), 3.2 (BR trainer), 3.3 (solver), 3.4 (loop), 3.5 (tests)
  Milestone: PSRO works on Kuhn Poker with population growth

Day 9 (Phase 4):
  Use Prompts: 4.1 (configs), 4.2 (GCP submission)
  Milestone: Ready for cloud deployment

Day 10 (Phase 5):
  Use Prompts: 5.1 (aggregation), 5.2 (plotting)
  Milestone: Results visualization pipeline ready

================================================================================
