import numpy as np
import pickle
import random
from typing import Dict, Tuple, Any
from .action_space import DiscountAction

class QTable:
    """Q-Learning implementation with memory-efficient sparse storage"""
    
    def __init__(self, 
                 state_space_size: int,
                 action_space_size: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.1):
        """
        Args:
            state_space_size: Total number of possible states
            action_space_size: Total number of possible actions  
            learning_rate: Alpha parameter for Q-learning updates
            discount_factor: Gamma parameter for future reward discounting
            epsilon: Exploration rate for epsilon-greedy policy
        """
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Memory efficient storage - only create arrays for visited states
        self.q_values: Dict[Tuple, np.ndarray] = {}
        
        # State visit tracking
        self.state_visits: Dict[Tuple, int] = {}
        
        # Training statistics
        self.total_updates = 0
        self.episodes_trained = 0
        self.total_reward = 0.0
        
    def _ensure_state_exists(self, state: Tuple) -> None:
        """Initialize Q-values for a state if not seen before"""
        if state not in self.q_values:
            self.q_values[state] = np.zeros(self.action_space_size)
            self.state_visits[state] = 0
        
        self.state_visits[state] += 1
    
    def get_q_value(self, state: Tuple, action: DiscountAction) -> float:
        """Get Q-value for state-action pair"""
        self._ensure_state_exists(state)
        return self.q_values[state][action.value]
    
    def get_best_action(self, state: Tuple) -> DiscountAction:
        """Get best action for state (greedy policy)"""
        self._ensure_state_exists(state)
        best_action_idx = np.argmax(self.q_values[state])
        return DiscountAction(best_action_idx)
    
    def get_action_epsilon_greedy(self, state: Tuple) -> DiscountAction:
        """Get action using epsilon-greedy exploration"""
        if random.random() < self.epsilon:
            # Explore: random action
            return DiscountAction(random.randint(0, self.action_space_size - 1))
        else:
            # Exploit: best known action
            return self.get_best_action(state)
    
    def update(self, 
               state: Tuple, 
               action: DiscountAction, 
               reward: float, 
               next_state: Tuple, 
               done: bool) -> None:
        """Q-learning update rule"""
        self._ensure_state_exists(state)
        
        current_q = self.get_q_value(state, action)
        
        if done:
            # Terminal state - no future rewards
            target_q = reward
        else:
            # Non-terminal - include discounted future reward
            self._ensure_state_exists(next_state)
            max_next_q = np.max(self.q_values[next_state])
            target_q = reward + self.discount_factor * max_next_q
        
        # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        self.q_values[state][action.value] = current_q + self.learning_rate * (target_q - current_q)
        
        # Update statistics
        self.total_updates += 1
        self.total_reward += reward
    
    def decay_epsilon(self, decay_rate: float, min_epsilon: float = 0.01) -> None:
        """Decay exploration rate over time"""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
    
    def save(self, filepath: str) -> None:
        """Save Q-table to file using pickle"""
        data = {
            'q_values': self.q_values,
            'state_visits': self.state_visits,
            'state_space_size': self.state_space_size,
            'action_space_size': self.action_space_size,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'total_updates': self.total_updates,
            'episodes_trained': self.episodes_trained,
            'total_reward': self.total_reward
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str) -> None:
        """Load Q-table from file using pickle"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.q_values = data['q_values']
        self.state_visits = data['state_visits']
        self.state_space_size = data['state_space_size']
        self.action_space_size = data['action_space_size']
        self.learning_rate = data['learning_rate']
        self.discount_factor = data['discount_factor']
        self.epsilon = data['epsilon']
        self.total_updates = data.get('total_updates', 0)
        self.episodes_trained = data.get('episodes_trained', 0)
        self.total_reward = data.get('total_reward', 0.0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'total_states_visited': len(self.q_values),
            'total_updates': self.total_updates,
            'episodes_trained': self.episodes_trained,
            'current_epsilon': self.epsilon,
            'average_reward_per_update': self.total_reward / max(1, self.total_updates),
            'state_space_coverage': len(self.q_values) / self.state_space_size,
            'most_visited_states': sorted(
                [(state, visits) for state, visits in self.state_visits.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]  # Top 10 most visited states
        }
    
    def episode_finished(self) -> None:
        """Call this when an episode finishes to update statistics"""
        self.episodes_trained += 1
