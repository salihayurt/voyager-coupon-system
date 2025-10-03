from typing import List, Dict, Tuple, Any
from tqdm import tqdm
import numpy as np

from .q_table import QTable
from .state_encoder import StateEncoder
from .reward_function import RewardCalculator
from .action_space import ActionSpace, DiscountAction
from core.domain.user import User

class QLearningTrainer:
    """Q-Learning trainer that orchestrates all components"""
    
    def __init__(self, 
                 q_table: QTable, 
                 state_encoder: StateEncoder, 
                 reward_calculator: RewardCalculator):
        """
        Args:
            q_table: Q-Table for learning
            state_encoder: Converts User to state tuples
            reward_calculator: Computes rewards for actions
        """
        self.q_table = q_table
        self.state_encoder = state_encoder
        self.reward_calculator = reward_calculator
    
    def train_episode(self, 
                     user: User,
                     predicted_conversion_rates: Dict[int, float],
                     estimated_profits: Dict[int, float]) -> Dict[str, Any]:
        """
        Train on a single user episode
        
        Args:
            user: User to train on
            predicted_conversion_rates: {discount_percentage: conversion_rate}
            estimated_profits: {discount_percentage: profit}
        
        Returns:
            Episode training results
        """
        # Encode current state
        state = self.state_encoder.encode(user)
        
        # Select action using epsilon-greedy
        action = self.q_table.get_action_epsilon_greedy(state)
        discount_percentage = ActionSpace.get_discount_percentage(action)
        
        # Get predictions for this discount level
        predicted_conversion = predicted_conversion_rates.get(discount_percentage, 0.5)
        estimated_profit = estimated_profits.get(discount_percentage, 0.0)
        
        # Calculate reward using estimated values
        reward = self.reward_calculator.estimate_reward(
            user=user,
            discount_percentage=discount_percentage,
            predicted_conversion_rate=predicted_conversion,
            estimated_profit=estimated_profit
        )
        
        # For Q-learning, we assume this is a terminal episode (one-step decision)
        # In a more complex scenario, next_state would be different
        next_state = state  # Same state (terminal)
        done = True
        
        # Update Q-table
        self.q_table.update(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )
        
        return {
            'state': state,
            'action': action,
            'discount_percentage': discount_percentage,
            'reward': reward,
            'predicted_conversion': predicted_conversion,
            'estimated_profit': estimated_profit
        }
    
    def train(self, 
              users: List[User],
              agent_predictions: List[Dict[str, Dict[int, float]]],
              epochs: int = 10,
              epsilon_decay: bool = True,
              epsilon_decay_rate: float = 0.995,
              min_epsilon: float = 0.01) -> Dict[str, Any]:
        """
        Batch training over multiple epochs
        
        Args:
            users: List of users for training
            agent_predictions: List of {
                'conversion_rates': {discount: rate},
                'profits': {discount: profit}
            }
            epochs: Number of training epochs
            epsilon_decay: Whether to decay exploration rate
            epsilon_decay_rate: Rate of epsilon decay
            min_epsilon: Minimum epsilon value
        
        Returns:
            Training summary statistics
        """
        if len(users) != len(agent_predictions):
            raise ValueError("Users and agent_predictions must have same length")
        
        training_rewards = []
        epoch_stats = []
        
        # Training loop
        for epoch in range(epochs):
            epoch_rewards = []
            
            # Progress bar for current epoch
            pbar = tqdm(
                zip(users, agent_predictions), 
                total=len(users),
                desc=f"Epoch {epoch+1}/{epochs} (ε={self.q_table.epsilon:.3f})"
            )
            
            for user, predictions in pbar:
                # Train on this user
                episode_result = self.train_episode(
                    user=user,
                    predicted_conversion_rates=predictions['conversion_rates'],
                    estimated_profits=predictions['profits']
                )
                
                epoch_rewards.append(episode_result['reward'])
                
                # Update progress bar with recent reward
                pbar.set_postfix({
                    'reward': f"{episode_result['reward']:.3f}",
                    'discount': f"{episode_result['discount_percentage']}%"
                })
            
            # Mark episode finished
            self.q_table.episode_finished()
            
            # Decay epsilon
            if epsilon_decay:
                self.q_table.decay_epsilon(epsilon_decay_rate, min_epsilon)
            
            # Record epoch statistics
            avg_reward = np.mean(epoch_rewards)
            training_rewards.extend(epoch_rewards)
            
            epoch_stats.append({
                'epoch': epoch + 1,
                'avg_reward': avg_reward,
                'epsilon': self.q_table.epsilon,
                'episodes_this_epoch': len(epoch_rewards)
            })
            
            print(f"Epoch {epoch+1} completed - Avg Reward: {avg_reward:.4f}, ε: {self.q_table.epsilon:.4f}")
        
        # Final training summary
        summary = {
            'total_epochs': epochs,
            'total_episodes': len(training_rewards),
            'final_epsilon': self.q_table.epsilon,
            'avg_reward_overall': np.mean(training_rewards),
            'reward_std': np.std(training_rewards),
            'min_reward': np.min(training_rewards),
            'max_reward': np.max(training_rewards),
            'epoch_stats': epoch_stats,
            'q_table_stats': self.q_table.get_statistics()
        }
        
        return summary
    
    def predict(self, user: User) -> Tuple[int, float]:
        """
        Make prediction for a user (inference mode)
        
        Args:
            user: User to make prediction for
        
        Returns:
            (discount_percentage, q_value) - Best action and its Q-value
        """
        # Encode state
        state = self.state_encoder.encode(user)
        
        # Get best action (greedy, no exploration)
        best_action = self.q_table.get_best_action(state)
        discount_percentage = ActionSpace.get_discount_percentage(best_action)
        
        # Get Q-value for this action
        q_value = self.q_table.get_q_value(state, best_action)
        
        return discount_percentage, q_value
    
    def predict_with_details(self, user: User) -> Dict[str, Any]:
        """
        Make detailed prediction with all action Q-values
        
        Args:
            user: User to analyze
        
        Returns:
            Detailed prediction results
        """
        state = self.state_encoder.encode(user)
        
        # Get Q-values for all actions
        all_q_values = {}
        for action in DiscountAction:
            discount = ActionSpace.get_discount_percentage(action)
            q_value = self.q_table.get_q_value(state, action)
            all_q_values[discount] = q_value
        
        # Get best action
        best_action = self.q_table.get_best_action(state)
        best_discount = ActionSpace.get_discount_percentage(best_action)
        
        return {
            'user_id': user.user_id,
            'state': state,
            'state_readable': self.state_encoder.decode_readable(state),
            'best_discount': best_discount,
            'best_q_value': all_q_values[best_discount],
            'all_q_values': all_q_values,
            'state_visits': self.q_table.state_visits.get(state, 0)
        }
