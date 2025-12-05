"""
State encoding utilities for OpenSpiel games.

This module provides game-specific state encoders that convert OpenSpiel
information states into fixed-size numpy arrays suitable for neural networks.

TODO: Implement game-specific encoders for:
- kuhn_poker (30 dims)
- leduc_poker (50 dims)
- liars_dice (20 dims)
- dark_hex (18 dims)
"""

import numpy as np
import pyspiel
from typing import Optional


def encode_state(state: pyspiel.State, game: pyspiel.Game) -> np.ndarray:
    """
    Encode an OpenSpiel state to a fixed-size numpy array.
    
    This is a fallback implementation that uses OpenSpiel's built-in
    information_state_tensor. Game-specific encoders should override this
    for better performance and custom features.
    
    Args:
        state: OpenSpiel state object
        game: OpenSpiel game object
    
    Returns:
        Encoded state as numpy array of shape (state_dim,)
    """
    if state.is_terminal():
        # Return zeros for terminal states
        return np.zeros(game.information_state_tensor_size(), dtype=np.float32)
    
    current_player = state.current_player()
    tensor = state.information_state_tensor(current_player)
    return np.array(tensor, dtype=np.float32)


# Game-specific encoders will be implemented here
# Example structure:
#
# def encode_kuhn_poker(state: pyspiel.State, game: pyspiel.Game) -> np.ndarray:
#     """Encode Kuhn Poker state (30 dimensions)."""
#     pass
#
# def encode_leduc_poker(state: pyspiel.State, game: pyspiel.Game) -> np.ndarray:
#     """Encode Leduc Poker state (50 dimensions)."""
#     pass

