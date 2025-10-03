#!/usr/bin/env python3
"""
Q-Learning Training Script for Voyager Coupon System

Trains Q-Learning model using offline data with proxy conversion and profit functions.
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from adapters.external.data_analysis_client import DataAnalysisClient
from learning.q_table import QTable
from learning.state_encoder import StateEncoder
from learning.action_space import ActionSpace, DiscountAction
from learning.reward_function import RewardCalculator
from learning.trainer import QLearningTrainer
from core.domain.user import User
from core.domain.enums import DomainType, SegmentType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def proxy_conversion_rate(discount: int, user: User) -> float:
    """
    Proxy function to estimate conversion rate based on user characteristics and discount
    
    This simulates real conversion data we don't have yet.
    In production, this would be replaced with actual historical conversion data.
    """
    # Base conversion rates by segment
    base_rates = {
        SegmentType.PREMIUM_CUSTOMERS: 0.8,
        SegmentType.HIGH_VALUE_CUSTOMERS: 0.7,
        SegmentType.PRICE_SENSITIVE_CUSTOMERS: 0.5,
        SegmentType.AT_RISK_CUSTOMERS: 0.4,
        SegmentType.STANDARD_CUSTOMERS: 0.6
    }
    
    base_rate = base_rates.get(user.segment, 0.5)
    
    # Discount boost: each 1% discount adds conversion probability
    discount_boost = discount * 0.015  # 1.5% boost per discount point
    
    # Score-based adjustments
    churn_penalty = user.scores.churn_score * 0.2  # High churn reduces conversion
    activity_boost = user.scores.activity_score * 0.1  # Active users convert better
    price_sensitivity_boost = user.scores.price_sensitivity * discount * 0.003
    family_boost = (1 - user.scores.family_score) * 0.05  # Family buyers more likely to convert
    
    # Cart abandonment penalty
    cart_penalty = user.scores.cart_abandon_score * 0.15
    
    # Calculate final conversion rate
    conversion_rate = (base_rate + discount_boost + activity_boost + 
                      price_sensitivity_boost + family_boost - churn_penalty - cart_penalty)
    
    # Add some noise to make it realistic
    noise = np.random.normal(0, 0.02)
    conversion_rate += noise
    
    return max(0.05, min(0.95, conversion_rate))  # Bound between 5% and 95%

def proxy_expected_profit(discount: int, user: User) -> float:
    """
    Proxy function to estimate profit based on user characteristics and discount
    
    This simulates real profit data calculation.
    """
    # Base order values by segment
    base_values = {
        SegmentType.PREMIUM_CUSTOMERS: 6000,
        SegmentType.HIGH_VALUE_CUSTOMERS: 5000,
        SegmentType.PRICE_SENSITIVE_CUSTOMERS: 1500,
        SegmentType.AT_RISK_CUSTOMERS: 2000,
        SegmentType.STANDARD_CUSTOMERS: 2500
    }
    
    base_value = base_values.get(user.segment, 2000)
    
    # Family booking adjustment
    if user.scores.family_score < 0.4:  # Family buyer
        base_value *= 1.3  # Family bookings are typically larger
    
    # Domain adjustments
    domain_multipliers = {
        DomainType.ENUYGUN_HOTEL: 1.2,
        DomainType.ENUYGUN_FLIGHT: 1.0,
        DomainType.ENUYGUN_CAR_RENTAL: 0.8,
        DomainType.ENUYGUN_BUS: 0.6,
        DomainType.WINGIE_FLIGHT: 0.9
    }
    
    base_value *= domain_multipliers.get(user.domain, 1.0)
    
    # Calculate profit margin (base 20% minus discount)
    profit_margin = max(0.05, 0.20 - (discount / 100))  # Minimum 5% margin
    
    expected_profit = base_value * profit_margin
    
    # Add some realistic noise
    noise = np.random.normal(0, expected_profit * 0.05)  # 5% noise
    expected_profit += noise
    
    return max(0, expected_profit)

def create_agent_predictions(users: list[User]) -> list[dict]:
    """Create mock agent predictions for training"""
    predictions = []
    
    for user in users:
        # Generate predictions for all discount levels
        conversion_rates = {}
        profits = {}
        
        for action in DiscountAction:
            discount = ActionSpace.get_discount_percentage(action)
            conversion_rates[discount] = proxy_conversion_rate(discount, user)
            profits[discount] = proxy_expected_profit(discount, user)
        
        predictions.append({
            'conversion_rates': conversion_rates,
            'profits': profits
        })
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description='Train Q-Learning model for coupon optimization')
    parser.add_argument('--csv', required=True, help='Path to structured customer data CSV')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Q-Learning learning rate')
    parser.add_argument('--epsilon', type=float, default=0.3, help='Initial epsilon for exploration')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--save-path', default='data/q_table.pkl', help='Path to save trained Q-table')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    logger.info("🚀 Starting Q-Learning training for Voyager Coupon System")
    logger.info(f"Configuration: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
    
    try:
        # Load data
        logger.info(f"📂 Loading data from {args.csv}")
        client = DataAnalysisClient(args.csv)
        all_users = client.load_users()
        
        if not all_users:
            logger.error("❌ No users loaded. Check your CSV file.")
            return 1
        
        logger.info(f"✅ Loaded {len(all_users)} users")
        
        # Initialize components
        logger.info("🔧 Initializing Q-Learning components...")
        state_encoder = StateEncoder()
        q_table = QTable(
            state_space_size=state_encoder.state_space_size,
            action_space_size=ActionSpace.total_actions(),
            learning_rate=args.learning_rate,
            epsilon=args.epsilon
        )
        reward_calculator = RewardCalculator()
        
        logger.info(f"✅ Q-Learning setup complete - {state_encoder.state_space_size:,} total states")
        
        # Create trainer
        trainer = QLearningTrainer(q_table, state_encoder, reward_calculator)
        
        # Training loop
        logger.info("🎯 Starting training...")
        
        best_avg_reward = -float('inf')
        training_history = []
        
        for epoch in range(args.epochs):
            logger.info(f"📊 Epoch {epoch + 1}/{args.epochs}")
            
            # Sample batch of users
            if len(all_users) > args.batch_size:
                batch_indices = np.random.choice(len(all_users), args.batch_size, replace=False)
                batch_users = [all_users[i] for i in batch_indices]
            else:
                batch_users = all_users
            
            # Generate agent predictions for this batch
            logger.info("🤖 Generating agent predictions...")
            agent_predictions = create_agent_predictions(batch_users)
            
            # Train on this batch
            logger.info("🧠 Training Q-Learning model...")
            results = trainer.train(
                users=batch_users,
                agent_predictions=agent_predictions,
                epochs=1,  # Single epoch per batch
                epsilon_decay=epoch < args.epochs - 5,  # Stop decay in last 5 epochs
                epsilon_decay_rate=args.epsilon_decay
            )
            
            # Log results
            avg_reward = results['avg_reward_overall']
            current_epsilon = results['final_epsilon']
            
            logger.info(f"📈 Epoch {epoch + 1} Results:")
            logger.info(f"   Average Reward: {avg_reward:.4f}")
            logger.info(f"   Epsilon: {current_epsilon:.4f}")
            logger.info(f"   States Visited: {results['q_table_stats']['total_states_visited']}")
            
            training_history.append({
                'epoch': epoch + 1,
                'avg_reward': avg_reward,
                'epsilon': current_epsilon,
                'states_visited': results['q_table_stats']['total_states_visited']
            })
            
            # Save best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                logger.info(f"💾 New best model! Saving to {args.save_path}")
                q_table.save(args.save_path)
        
        # Final statistics
        logger.info("📊 Training Complete!")
        logger.info(f"   Best Average Reward: {best_avg_reward:.4f}")
        logger.info(f"   Final Epsilon: {q_table.epsilon:.4f}")
        logger.info(f"   Total States Explored: {q_table.get_statistics()['total_states_visited']}")
        logger.info(f"   State Space Coverage: {q_table.get_statistics()['state_space_coverage']:.2%}")
        
        # Test the trained model
        logger.info("🧪 Testing trained model on sample users...")
        test_users = all_users[:10]  # Test on first 10 users
        
        for i, user in enumerate(test_users):
            discount, q_value = trainer.predict(user)
            logger.info(f"   User {user.user_id}: {discount}% discount (Q-value: {q_value:.3f})")
        
        logger.info("🎉 Training pipeline completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
