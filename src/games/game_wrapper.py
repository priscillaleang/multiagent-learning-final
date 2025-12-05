"""
Unified interface for OpenSpiel games.

Provides a consistent API for NFSP and PSRO algorithms to interact with
OpenSpiel games, with automatic chance node handling and state encoding.
"""

import pyspiel
import numpy as np
from typing import Tuple, List, Optional


class GameWrapper:
    """
    Wrapper for OpenSpiel games that provides a unified interface for RL algorithms.
    
    Handles:
    - Automatic chance node resolution
    - State encoding via state_encoder module
    - Game validation (extensive-form, 2-player, sequential)
    - Episode statistics tracking
    """
    
    def __init__(self, game_name: str):
        """
        Initialize game wrapper with validation.
        
        Args:
            game_name: Name of the OpenSpiel game (e.g., 'kuhn_poker', 'leduc_poker')
        
        Raises:
            ValueError: If game doesn't meet requirements for NFSP/PSRO
        """
        # Load game
        self.game_name = game_name
        self.game = pyspiel.load_game(game_name)
        
        # === VERIFY EXTENSIVE-FORM REQUIREMENTS ===
        game_type = self.game.get_type()
        
        # Check 1: Must provide information state
        if not game_type.provides_information_state_tensor:
            raise ValueError(
                f"{game_name} does not provide information_state_tensor. "
                "NFSP/PSRO require extensive-form games with imperfect information."
            )
        
        # Check 2: Must be sequential (not simultaneous moves)
        if game_type.dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
            raise ValueError(
                f"{game_name} has simultaneous moves. "
                "Use sequential extensive-form games instead."
            )
        
        # Check 3: Should be two-player zero-sum (for Nash convergence)
        if game_type.utility != pyspiel.GameType.Utility.ZERO_SUM:
            print(
                f"Warning: {game_name} is not zero-sum. "
                "Convergence guarantees may not hold."
            )
        
        if self.game.num_players() != 2:
            raise ValueError(
                f"{game_name} has {self.game.num_players()} players. "
                "This implementation supports 2-player games only."
            )
        
        # === INITIALIZE ===
        self.state_dim = self.game.information_state_tensor_size()
        self.action_dim = self.game.num_distinct_actions()
        self.num_players = self.game.num_players()
        
        # Current game state
        self.state = None
        
        # Episode statistics
        self.episode_rewards = [0.0, 0.0]
        self.episode_length = 0
        
        # Try to import state_encoder, fallback to direct encoding if not available
        try:
            from src.games import state_encoder
            self.state_encoder = state_encoder
            self._use_custom_encoder = True
        except ImportError:
            # Fallback: use OpenSpiel's built-in information_state_tensor
            self.state_encoder = None
            self._use_custom_encoder = False
    
    def reset(self) -> np.ndarray:
        """
        Reset game to initial state and return encoded observation.
        
        Returns:
            Encoded information state as numpy array
        """
        self.state = self.game.new_initial_state()
        
        # Handle chance nodes automatically
        self._resolve_chance_nodes()
        
        # Reset episode statistics
        self.episode_rewards = [0.0, 0.0]
        self.episode_length = 0
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[Optional[np.ndarray], List[float], bool]:
        """
        Execute an action and return next observation, rewards, and done flag.
        
        Args:
            action: Action index to execute
        
        Returns:
            Tuple of (observation, rewards, done):
            - observation: Encoded information state (None if terminal)
            - rewards: List of rewards for each player [player_0_reward, player_1_reward]
            - done: Whether the episode is finished
        """
        if self.state is None:
            raise RuntimeError("Game not initialized. Call reset() first.")
        
        if self.state.is_terminal():
            raise RuntimeError("Cannot step from terminal state. Call reset() first.")
        
        # Apply action
        self.state.apply_action(action)
        self.episode_length += 1
        
        # Handle chance nodes automatically
        self._resolve_chance_nodes()
        
        # Check if terminal
        done = self.state.is_terminal()
        
        if done:
            # Get final rewards
            returns = self.state.returns()
            self.episode_rewards = [returns[0], returns[1]]
            return None, self.episode_rewards, True
        else:
            # Non-terminal: return observation and zero rewards
            obs = self._get_observation()
            return obs, [0.0, 0.0], False
    
    def current_player(self) -> int:
        """
        Get the current player index.
        
        Returns:
            Player index (0 or 1), or -1 if terminal/chance node
        """
        if self.state is None:
            raise RuntimeError("Game not initialized. Call reset() first.")
        
        if self.state.is_terminal() or self.state.is_chance_node():
            return -1
        
        return self.state.current_player()
    
    def legal_actions(self) -> List[int]:
        """
        Get list of legal actions for the current player.
        
        Returns:
            List of legal action indices
        """
        if self.state is None:
            raise RuntimeError("Game not initialized. Call reset() first.")
        
        if self.state.is_terminal():
            return []
        
        if self.state.is_chance_node():
            # Return chance outcomes as "actions" (though they're sampled automatically)
            return [outcome[0] for outcome in self.state.chance_outcomes()]
        
        return self.state.legal_actions()
    
    def _resolve_chance_nodes(self) -> None:
        """
        Automatically resolve chance nodes by sampling from their distributions.
        
        Continues until we reach a decision node or terminal state.
        """
        while self.state.is_chance_node():
            outcomes = self.state.chance_outcomes()
            # Extract actions and probabilities
            actions = [outcome[0] for outcome in outcomes]
            probs = [outcome[1] for outcome in outcomes]
            
            # Sample action according to probability distribution
            action = np.random.choice(actions, p=probs)
            self.state.apply_action(action)
    
    def _get_observation(self) -> np.ndarray:
        """
        Get encoded observation for the current player.
        
        Returns:
            Encoded information state as numpy array
        """
        if self.state.is_terminal():
            # Return zeros for terminal state (shouldn't be used, but handle gracefully)
            return np.zeros(self.state_dim, dtype=np.float32)
        
        current_player = self.state.current_player()
        
        # Try to use custom encoder if available
        if self._use_custom_encoder and self.state_encoder is not None:
            try:
                return self.state_encoder.encode_state(self.state, self.game)
            except (AttributeError, NotImplementedError):
                # Fallback if encoder doesn't support this game
                pass
        
        # Fallback: use OpenSpiel's built-in information state tensor
        tensor = self.state.information_state_tensor(current_player)
        return np.array(tensor, dtype=np.float32)
    
    def get_episode_stats(self) -> dict:
        """
        Get statistics for the current/last episode.
        
        Returns:
            Dictionary with episode statistics:
            - rewards: List of final rewards [player_0, player_1]
            - length: Episode length (number of actions)
        """
        return {
            'rewards': self.episode_rewards.copy(),
            'length': self.episode_length
        }
    
    def clone_state(self) -> pyspiel.State:
        """
        Clone the current game state (useful for tree search, evaluation, etc.).
        
        Returns:
            Cloned state object
        """
        if self.state is None:
            raise RuntimeError("Game not initialized. Call reset() first.")
        
        return self.state.clone()
    
    def __repr__(self) -> str:
        """String representation of the wrapper."""
        return (
            f"GameWrapper(game='{self.game_name}', "
            f"state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"players={self.num_players})"
        )

