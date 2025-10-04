#!/usr/bin/env python3
"""
Policy Tree Training Script for WEGATHON Dataset

Trains Policy Tree model using the real WEGATHON 2025 Voyager dataset.
"""

import argparse
import logging
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from adapters.external.wegathon_data_client import WegathonDataClient
from policy_tree.train import train_pipeline
from policy_tree.feasible import choose_actions_batch, validate_action_selection
from policy_tree.tags import extract_tags_batch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_wegathon_data(limit: int = None) -> pd.DataFrame:
    """
    Prepare WEGATHON data for policy tree training
    
    Args:
        limit: Maximum number of users to process
        
    Returns:
        DataFrame ready for policy tree training
    """
    logger.info("Loading WEGATHON dataset...")
    
    # Load users from WEGATHON data
    data_client = WegathonDataClient()
    users = data_client.load_users(limit=limit)
    
    logger.info(f"Loaded {len(users)} users from WEGATHON dataset")
    
    # Convert to DataFrame format expected by policy tree
    user_data = []
    for user in users:
        # Create synthetic recommendation data based on user characteristics
        # In a real scenario, this would come from Voyager's recommendation outputs
        
        # Base discount recommendation based on segment
        segment_base_discounts = {
            'at_risk_customers': 12,
            'high_value_customers': 7,
            'standard_customers': 8,
            'price_sensitive_customers': 14,
            'premium_customers': 6
        }
        
        segment_str = user.segment.value if hasattr(user.segment, 'value') else str(user.segment)
        base_discount = segment_base_discounts.get(segment_str, 10)
        
        # Adjust based on user scores
        if user.scores.churn_score > 0.7:
            base_discount += 2  # Higher discount for high churn risk
        if user.scores.price_sensitivity > 0.6:
            base_discount += 1  # Slightly higher for price sensitive
        
        # Cap discount within reasonable range
        recommended_discount = max(5, min(20, base_discount))
        
        # Estimate expected metrics based on user characteristics
        expected_conversion = 0.4 + (recommended_discount / 100) * 2  # Higher discount = higher conversion
        expected_conversion = min(0.95, expected_conversion)
        
        # Estimate profit (higher for premium segments, adjusted by discount)
        segment_base_profits = {
            'at_risk_customers': 150,
            'high_value_customers': 400,
            'standard_customers': 250,
            'price_sensitive_customers': 120,
            'premium_customers': 600
        }
        
        base_profit = segment_base_profits.get(segment_str, 200)
        expected_profit = base_profit * (1 - recommended_discount / 100) * 1.2  # Profit after discount
        
        # Generate decision factors text
        decision_factors = f"User in {segment_str} segment with "
        factors = []
        
        if user.scores.churn_score > 0.7:
            factors.append("high churn risk")
        if user.scores.price_sensitivity > 0.6:
            factors.append("price sensitive behavior")
        if user.scores.cart_abandon_score > 0.6:
            factors.append("cart abandonment tendency")
        if user.scores.family_score < 0.4:
            factors.append("family travel pattern")
        if user.scores.activity_score < 0.3:
            factors.append("low activity")
        
        if factors:
            decision_factors += ", ".join(factors)
        else:
            decision_factors += "standard behavioral patterns"
        
        user_dict = {
            'user_id': user.user_id,
            'segment': segment_str,
            'domain': user.domain.value if hasattr(user.domain, 'value') else str(user.domain),
            'churn_score': user.scores.churn_score,
            'activity_score': user.scores.activity_score,
            'cart_abandon_score': user.scores.cart_abandon_score,
            'price_sensitivity': user.scores.price_sensitivity,
            'family_score': user.scores.family_score,
            'is_oneway': user.is_oneway,
            'user_basket': user.user_basket,
            'recommended_discount_pct': recommended_discount,
            'expected_profit': expected_profit,
            'expected_conversion': expected_conversion,
            'confidence_score': 0.7 + (user.scores.activity_score * 0.2),  # Higher confidence for active users
            'decision_factors': decision_factors,
            'options': []  # Will be populated if needed
        }
        user_data.append(user_dict)
    
    df = pd.DataFrame(user_data)
    logger.info(f"Prepared {len(df)} user records for training")
    
    return df

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Policy Tree on WEGATHON Dataset')
    parser.add_argument('--limit', type=int, default=5000, help='Limit number of users (default: 5000)')
    parser.add_argument('--output', default='artifacts/policy_tree_wegathon.joblib', help='Output model path')
    parser.add_argument('--max-depth', type=int, default=4, help='Maximum tree depth')
    parser.add_argument('--min-leaf-frac', type=float, default=0.02, help='Min leaf fraction')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    try:
        # Prepare WEGATHON data
        logger.info("Preparing WEGATHON dataset for policy tree training...")
        df = prepare_wegathon_data(limit=args.limit)
        
        # Display data statistics
        logger.info("Dataset statistics:")
        logger.info(f"  Total users: {len(df)}")
        logger.info(f"  Segments: {df['segment'].value_counts().to_dict()}")
        logger.info(f"  Domains: {df['domain'].value_counts().to_dict()}")
        logger.info(f"  Avg churn score: {df['churn_score'].mean():.3f}")
        logger.info(f"  Avg price sensitivity: {df['price_sensitivity'].mean():.3f}")
        
        # Run training pipeline
        logger.info("Starting policy tree training pipeline...")
        saved_path, metrics = train_pipeline(
            df,
            output_path=args.output,
            max_depth=args.max_depth,
            min_samples_leaf_frac=args.min_leaf_frac,
            test_size=args.test_size,
            random_state=args.random_seed
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {saved_path}")
        logger.info(f"Training metrics: {metrics}")
        
        # Display final results
        print("\n" + "="*60)
        print("🎉 WEGATHON Policy Tree Training Complete!")
        print("="*60)
        print(f"📊 Dataset: {len(df)} users from WEGATHON 2025")
        print(f"📁 Model: {saved_path}")
        print(f"🎯 Test Accuracy: {metrics.get('test_accuracy', 'N/A')}")
        print(f"🔍 Violations: {metrics['validation_report']['violation_rate']:.1%}")
        print(f"📈 Total Cohorts: Estimated 6-12 cohorts")
        print("\n🚀 Ready to use with:")
        print(f"   GET /policy/cohorts")
        print(f"   POST /policy/preview")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
