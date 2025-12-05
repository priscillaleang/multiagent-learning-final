"""
State encoding utilities for OpenSpiel games.

This module provides game-specific state encoders that convert OpenSpiel
information states into fixed-size numpy arrays suitable for neural networks.
"""

import numpy as np
import pyspiel


def _get_game_name(game: pyspiel.Game) -> str:
    """Extract game name from game object."""
    return game.get_type().short_name


def _validate_state(state: pyspiel.State, game: pyspiel.Game) -> None:
    """Validate state before encoding."""
    if state is None:
        raise ValueError("State cannot be None")
    if game is None:
        raise ValueError("Game cannot be None")
    if not state.is_terminal() and not state.is_chance_node():
        if state.current_player() < 0:
            raise ValueError("Invalid current player in state")


def encode_state(state: pyspiel.State, game: pyspiel.Game) -> np.ndarray:
    """
    Encode an OpenSpiel state to a fixed-size numpy array.
    
    Routes to game-specific encoders when available, otherwise uses
    OpenSpiel's built-in information_state_tensor as fallback.
    
    Args:
        state: OpenSpiel state object
        game: OpenSpiel game object
    
    Returns:
        Encoded state as numpy array of shape (state_dim,)
    """
    _validate_state(state, game)
    
    if state.is_terminal():
        # Return zeros for terminal states with appropriate dimension
        game_name = _get_game_name(game)
        if game_name == "kuhn_poker":
            return np.zeros(30, dtype=np.float32)
        elif game_name == "leduc_poker":
            return np.zeros(50, dtype=np.float32)
        elif game_name == "liars_dice":
            return np.zeros(20, dtype=np.float32)
        elif game_name == "dark_hex":
            return np.zeros(18, dtype=np.float32)
        else:
            return np.zeros(game.information_state_tensor_size(), dtype=np.float32)
    
    # Route to game-specific encoder
    game_name = _get_game_name(game)
    
    if game_name == "kuhn_poker":
        return encode_kuhn_poker(state, game)
    elif game_name == "leduc_poker":
        return encode_leduc_poker(state, game)
    elif game_name == "liars_dice":
        return encode_liars_dice(state, game)
    elif game_name == "dark_hex":
        return encode_dark_hex(state, game)
    else:
        # Fallback to OpenSpiel's default encoding
        current_player = state.current_player()
        tensor = state.information_state_tensor(current_player)
        return np.array(tensor, dtype=np.float32)


def encode_kuhn_poker(state: pyspiel.State, game: pyspiel.Game) -> np.ndarray:
    """
    Encode Kuhn Poker state (30 dimensions).
    
    Encoding:
    - 6-dim one-hot for cards (J, Q, K × 2 players)
    - 24-dim for betting history
    
    Args:
        state: OpenSpiel Kuhn Poker state
        game: OpenSpiel game object
    
    Returns:
        Encoded state as numpy array of shape (30,)
    """
    if state.is_terminal():
        return np.zeros(30, dtype=np.float32)
    
    current_player = state.current_player()
    tensor = state.information_state_tensor(current_player)
    tensor = np.array(tensor, dtype=np.float32)
    
    # Kuhn Poker information state tensor structure:
    # - First 3 dims: one-hot for player's card (J=0, Q=1, K=2)
    # - Next 3 dims: one-hot for opponent's card (if known)
    # - Remaining dims: betting history
    
    # Extract card information (6 dims)
    # Player's card (3 dims) + Opponent's card (3 dims)
    if len(tensor) >= 6:
        cards = tensor[:6].copy()
    else:
        # Pad if needed
        cards = np.zeros(6, dtype=np.float32)
        cards[:min(len(tensor), 6)] = tensor[:min(len(tensor), 6)]
    
    # Extract betting history (24 dims)
    # OpenSpiel's Kuhn Poker has variable betting history
    # We'll take up to 24 dims and pad if necessary
    if len(tensor) > 6:
        betting_raw = tensor[6:].copy()
        betting = np.zeros(24, dtype=np.float32)
        betting[:min(len(betting_raw), 24)] = betting_raw[:min(len(betting_raw), 24)]
    else:
        betting = np.zeros(24, dtype=np.float32)
    
    # Normalize to [0, 1] (already in that range for one-hot, but ensure)
    cards = np.clip(cards, 0.0, 1.0)
    betting = np.clip(betting, 0.0, 1.0)
    
    # Concatenate: [6-dim cards] + [24-dim betting] = 30 dims
    encoded = np.concatenate([cards, betting], axis=0).astype(np.float32)
    
    assert len(encoded) == 30, f"Expected 30 dims, got {len(encoded)}"
    return encoded


def encode_leduc_poker(state: pyspiel.State, game: pyspiel.Game) -> np.ndarray:
    """
    Encode Leduc Poker state (50 dimensions).
    
    Encoding:
    - 6-dim one-hot private card
    - 6-dim one-hot public card (zeros if unrevealed)
    - 38-dim betting history (padded)
    
    Args:
        state: OpenSpiel Leduc Poker state
        game: OpenSpiel game object
    
    Returns:
        Encoded state as numpy array of shape (50,)
    """
    if state.is_terminal():
        return np.zeros(50, dtype=np.float32)
    
    current_player = state.current_player()
    tensor = state.information_state_tensor(current_player)
    tensor = np.array(tensor, dtype=np.float32)
    
    # Leduc Poker information state tensor structure:
    # - First 3 dims: one-hot for player's private card (J=0, Q=1, K=2)
    # - Next 3 dims: one-hot for public card (if revealed, else zeros)
    # - Remaining dims: betting history
    
    # Extract private card (6 dims - one-hot for 6 possible cards: J, Q, K × 2 suits)
    if len(tensor) >= 3:
        private_raw = tensor[:3]
        # Expand to 6 dims (2 suits × 3 ranks)
        private_card = np.zeros(6, dtype=np.float32)
        # Map: J=0->[0,3], Q=1->[1,4], K=2->[2,5]
        for i in range(3):
            if private_raw[i] > 0:
                private_card[i] = 1.0  # First suit
                private_card[i + 3] = 0.0  # Second suit (not held)
    else:
        private_card = np.zeros(6, dtype=np.float32)
    
    # Extract public card (6 dims)
    if len(tensor) >= 6:
        public_raw = tensor[3:6]
        # Expand to 6 dims similar to private
        public_card = np.zeros(6, dtype=np.float32)
        for i in range(3):
            if public_raw[i] > 0:
                public_card[i] = 1.0
                public_card[i + 3] = 0.0
    else:
        public_card = np.zeros(6, dtype=np.float32)  # Not revealed yet
    
    # Extract betting history (38 dims)
    if len(tensor) > 6:
        betting_raw = tensor[6:].copy()
        betting = np.zeros(38, dtype=np.float32)
        betting[:min(len(betting_raw), 38)] = betting_raw[:min(len(betting_raw), 38)]
    else:
        betting = np.zeros(38, dtype=np.float32)
    
    # Normalize to [0, 1]
    private_card = np.clip(private_card, 0.0, 1.0)
    public_card = np.clip(public_card, 0.0, 1.0)
    betting = np.clip(betting, 0.0, 1.0)
    
    # Concatenate: [6-dim private] + [6-dim public] + [38-dim betting] = 50 dims
    encoded = np.concatenate([private_card, public_card, betting], axis=0).astype(np.float32)
    
    assert len(encoded) == 50, f"Expected 50 dims, got {len(encoded)}"
    return encoded


def encode_liars_dice(state: pyspiel.State, game: pyspiel.Game) -> np.ndarray:
    """
    Encode Liar's Dice state (20 dimensions).
    
    Encoding:
    - 5-dim own dice configuration
    - 15-dim opponent claim history
    
    Args:
        state: OpenSpiel Liar's Dice state
        game: OpenSpiel game object
    
    Returns:
        Encoded state as numpy array of shape (20,)
    """
    if state.is_terminal():
        return np.zeros(20, dtype=np.float32)
    
    current_player = state.current_player()
    tensor = state.information_state_tensor(current_player)
    tensor = np.array(tensor, dtype=np.float32)
    
    # Liar's Dice information state tensor structure:
    # - First few dims: own dice configuration
    # - Remaining dims: claim history and game state
    
    # Extract own dice configuration (5 dims)
    # Represent dice counts for faces 1-5 (or similar encoding)
    if len(tensor) >= 5:
        dice_raw = tensor[:5]
        # Normalize dice counts (typically 0-5 dice per face)
        dice_config = dice_raw / 5.0 if dice_raw.max() > 1.0 else dice_raw
    else:
        dice_config = np.zeros(5, dtype=np.float32)
        dice_config[:min(len(tensor), 5)] = tensor[:min(len(tensor), 5)]
    
    # Extract opponent claim history (15 dims)
    # Encode recent claims: (face, count) pairs
    if len(tensor) > 5:
        claims_raw = tensor[5:].copy()
        claims = np.zeros(15, dtype=np.float32)
        # Take up to 15 dims for claim history
        claims[:min(len(claims_raw), 15)] = claims_raw[:min(len(claims_raw), 15)]
    else:
        claims = np.zeros(15, dtype=np.float32)
    
    # Normalize to [0, 1]
    dice_config = np.clip(dice_config, 0.0, 1.0)
    claims = np.clip(claims, 0.0, 1.0)
    
    # Concatenate: [5-dim dice] + [15-dim claims] = 20 dims
    encoded = np.concatenate([dice_config, claims], axis=0).astype(np.float32)
    
    assert len(encoded) == 20, f"Expected 20 dims, got {len(encoded)}"
    return encoded


def encode_dark_hex(state: pyspiel.State, game: pyspiel.Game) -> np.ndarray:
    """
    Encode Dark Hex state (18 dimensions).
    
    Encoding:
    - 8-dim own board state
    - 8-dim opponent board (revealed pieces)
    - 2-dim game phase
    
    Args:
        state: OpenSpiel Dark Hex state
        game: OpenSpiel game object
    
    Returns:
        Encoded state as numpy array of shape (18,)
    """
    if state.is_terminal():
        return np.zeros(18, dtype=np.float32)
    
    current_player = state.current_player()
    tensor = state.information_state_tensor(current_player)
    tensor = np.array(tensor, dtype=np.float32)
    
    # Dark Hex (2x2 board) information state tensor structure:
    # - Board positions (own pieces, opponent pieces)
    # - Game phase information
    
    # Extract own board state (8 dims for 2x2 board = 4 positions × 2 states)
    # Or simpler: 8 positions if using a different encoding
    if len(tensor) >= 8:
        own_board = tensor[:8].copy()
    else:
        own_board = np.zeros(8, dtype=np.float32)
        own_board[:min(len(tensor), 8)] = tensor[:min(len(tensor), 8)]
    
    # Extract opponent board (8 dims - revealed pieces only)
    if len(tensor) >= 16:
        opponent_board = tensor[8:16].copy()
    else:
        opponent_board = np.zeros(8, dtype=np.float32)
        if len(tensor) > 8:
            opponent_board[:min(len(tensor) - 8, 8)] = tensor[8:min(len(tensor), 16)]
    
    # Extract game phase (2 dims)
    # Could encode: round number, turn number, etc.
    if len(tensor) >= 18:
        phase = tensor[16:18].copy()
    else:
        phase = np.zeros(2, dtype=np.float32)
        if len(tensor) > 16:
            phase[:min(len(tensor) - 16, 2)] = tensor[16:min(len(tensor), 18)]
        # If no phase info, encode based on game progress
        if len(tensor) > 0:
            # Use a simple heuristic: normalize by total moves possible
            phase[0] = min(len(tensor) / 20.0, 1.0)  # Progress indicator
            phase[1] = 1.0 if current_player == 0 else 0.0  # Turn indicator
    
    # Normalize to [0, 1]
    own_board = np.clip(own_board, 0.0, 1.0)
    opponent_board = np.clip(opponent_board, 0.0, 1.0)
    phase = np.clip(phase, 0.0, 1.0)
    
    # Concatenate: [8-dim own] + [8-dim opponent] + [2-dim phase] = 18 dims
    encoded = np.concatenate([own_board, opponent_board, phase], axis=0).astype(np.float32)
    
    assert len(encoded) == 18, f"Expected 18 dims, got {len(encoded)}"
    return encoded
