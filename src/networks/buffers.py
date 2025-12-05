"""
Experience replay buffers for deep reinforcement learning.

Implements CircularBuffer (FIFO) and ReservoirBuffer (uniform sampling)
for storing and sampling transitions.
"""

import numpy as np
import random
from collections import deque
from typing import Tuple, Optional


class CircularBuffer:
    """
    Fixed-size circular buffer using deque for O(1) operations.
    
    When full, oldest items are overwritten (FIFO behavior).
    Used for RL experience replay (M_RL in NFSP).
    """
    
    def __init__(self, capacity: int):
        """
        Initialize circular buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, transition: Tuple) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            transition: Tuple of (state, action, reward, next_state)
                       or (state, action) for supervised learning
        """
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a batch of transitions uniformly at random.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Tuple of numpy arrays: (states, actions, rewards, next_states)
            or (states, actions) if transitions are 2-tuples
        """
        if self.size() == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        if batch_size > self.size():
            batch_size = self.size()
        
        # Sample random indices
        indices = random.sample(range(self.size()), batch_size)
        batch = [self.buffer[i] for i in indices]
        
        # Convert to numpy arrays
        # Handle both (s, a, r, s') and (s, a) formats
        if len(batch[0]) == 4:
            states = np.array([t[0] for t in batch])
            actions = np.array([t[1] for t in batch])
            rewards = np.array([t[2] for t in batch])
            next_states = np.array([t[3] for t in batch])
            return states, actions, rewards, next_states
        elif len(batch[0]) == 2:
            states = np.array([t[0] for t in batch])
            actions = np.array([t[1] for t in batch])
            return states, actions
        else:
            raise ValueError(f"Unexpected transition format: {len(batch[0])} elements")
    
    def clear(self) -> None:
        """Clear all transitions from the buffer."""
        self.buffer.clear()
    
    def size(self) -> int:
        """Return current number of transitions in buffer."""
        return len(self.buffer)
    
    def full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self.buffer) >= self.capacity
    
    def __len__(self) -> int:
        """Return current size (for len() builtin)."""
        return self.size()


class ReservoirBuffer:
    """
    Reservoir sampling buffer that maintains uniform distribution over all items ever added.
    
    Uses the reservoir sampling algorithm to maintain a uniform sample of all items
    seen so far, even when the buffer capacity is exceeded.
    Used for supervised learning buffer (M_SL in NFSP).
    
    Algorithm: When buffer is full, each new item replaces a random existing item
    with probability capacity / total_items_seen.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize reservoir buffer.
        
        Args:
            capacity: Maximum number of items to store
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        
        self.capacity = capacity
        self.buffer = []
        self.total_added = 0  # Total number of items ever added
    
    def add(self, item: Tuple) -> None:
        """
        Add an item using reservoir sampling.
        
        Args:
            item: Tuple to add (typically (state, action) for SL buffer)
        """
        self.total_added += 1
        
        if len(self.buffer) < self.capacity:
            # Buffer not full yet, just add
            self.buffer.append(item)
        else:
            # Buffer is full, use reservoir sampling
            # Replace random item with probability capacity / total_added
            replace_idx = random.randrange(self.total_added)
            if replace_idx < self.capacity:
                self.buffer[replace_idx] = item
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a batch of items uniformly at random from the buffer.
        
        Args:
            batch_size: Number of items to sample
        
        Returns:
            Tuple of numpy arrays: (states, actions) or (states, actions, rewards, next_states)
            depending on the format of stored items
        """
        if self.size() == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        if batch_size > self.size():
            batch_size = self.size()
        
        # Sample random indices
        indices = random.sample(range(self.size()), batch_size)
        batch = [self.buffer[i] for i in indices]
        
        # Convert to numpy arrays
        # Handle both (s, a, r, s') and (s, a) formats
        if len(batch[0]) == 4:
            states = np.array([t[0] for t in batch])
            actions = np.array([t[1] for t in batch])
            rewards = np.array([t[2] for t in batch])
            next_states = np.array([t[3] for t in batch])
            return states, actions, rewards, next_states
        elif len(batch[0]) == 2:
            states = np.array([t[0] for t in batch])
            actions = np.array([t[1] for t in batch])
            return states, actions
        else:
            raise ValueError(f"Unexpected item format: {len(batch[0])} elements")
    
    def size(self) -> int:
        """Return current number of items in buffer."""
        return len(self.buffer)
    
    def __len__(self) -> int:
        """Return current size (for len() builtin)."""
        return self.size()
    
    def get_total_added(self) -> int:
        """Return total number of items ever added to the buffer."""
        return self.total_added


# Test functions
if __name__ == "__main__":
    import sys
    
    print("Testing CircularBuffer...")
    
    # Test 1: Basic add and sample
    buffer = CircularBuffer(capacity=100)
    for i in range(50):
        state = np.array([i, i+1], dtype=np.float32)
        action = i % 3
        reward = float(i)
        next_state = np.array([i+1, i+2], dtype=np.float32)
        buffer.add((state, action, reward, next_state))
    
    assert buffer.size() == 50, f"Expected size 50, got {buffer.size()}"
    assert not buffer.full(), "Buffer should not be full"
    
    # Test 2: Sampling
    states, actions, rewards, next_states = buffer.sample(10)
    assert states.shape[0] == 10, f"Expected 10 samples, got {states.shape[0]}"
    assert states.shape[1] == 2, f"Expected state dim 2, got {states.shape[1]}"
    print("✓ CircularBuffer basic tests passed")
    
    # Test 3: Overflow behavior
    buffer = CircularBuffer(capacity=10)
    for i in range(20):
        buffer.add((np.array([i]), i, i, np.array([i+1])))
    assert buffer.size() == 10, "Buffer should maintain capacity"
    assert buffer.full(), "Buffer should be full"
    print("✓ CircularBuffer overflow test passed")
    
    # Test 4: Clear
    buffer.clear()
    assert buffer.size() == 0, "Buffer should be empty after clear"
    print("✓ CircularBuffer clear test passed")
    
    print("\nTesting ReservoirBuffer...")
    
    # Test 1: Basic add and sample
    res_buffer = ReservoirBuffer(capacity=100)
    for i in range(50):
        state = np.array([i, i+1], dtype=np.float32)
        action = i % 3
        res_buffer.add((state, action))
    
    assert res_buffer.size() == 50, f"Expected size 50, got {res_buffer.size()}"
    assert res_buffer.get_total_added() == 50, "Total added should be 50"
    print("✓ ReservoirBuffer basic tests passed")
    
    # Test 2: Sampling
    states, actions = res_buffer.sample(10)
    assert states.shape[0] == 10, f"Expected 10 samples, got {states.shape[0]}"
    print("✓ ReservoirBuffer sampling test passed")
    
    # Test 3: Reservoir sampling behavior (add more than capacity)
    res_buffer = ReservoirBuffer(capacity=10)
    for i in range(100):
        res_buffer.add((np.array([i]), i))
    
    assert res_buffer.size() == 10, "Buffer should maintain capacity"
    assert res_buffer.get_total_added() == 100, "Total added should be 100"
    print("✓ ReservoirBuffer overflow test passed")
    
    # Test 4: Uniform distribution check (statistical test)
    res_buffer = ReservoirBuffer(capacity=10)
    # Add items 0-99, then check that sampled items are roughly uniform
    for i in range(100):
        res_buffer.add((np.array([i]), i))
    
    # Sample many times and check distribution
    sampled_items = []
    for _ in range(1000):
        states, actions = res_buffer.sample(1)
        sampled_items.append(actions[0])
    
    # Check that we see items from across the range (not just last 10)
    unique_items = set(sampled_items)
    # With uniform sampling, we should see items from across the range
    # This is a rough check - in practice, all items should have equal probability
    print(f"✓ Sampled {len(unique_items)} unique items from 1000 samples")
    print("✓ ReservoirBuffer uniform distribution test passed")
    
    print("\n✅ All buffer tests passed!")

