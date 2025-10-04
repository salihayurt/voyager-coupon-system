#!/usr/bin/env python3
"""
Direct test of cohort generation without API server
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from policy_tree.train import load_model
from policy_tree.inference import generate_cohorts
from adapters.external.wegathon_data_client import WegathonDataClient

def main():
    """Test cohort generation directly"""
    print("🧪 Testing Policy Tree Cohorts with WEGATHON Data...")
    print("="*60)
    
    try:
        # Find the latest trained model
        import glob
        model_files = glob.glob("artifacts/policy_tree_wegathon_*.joblib")
        if not model_files:
            print("❌ No trained model found. Please run training first:")
            print("   python scripts/train_policy_wegathon.py --limit 500")
            return
        
        latest_model = max(model_files, key=lambda x: Path(x).stat().st_mtime)
        print(f"📁 Using model: {latest_model}")
        
        # Load the trained model
        estimator, metadata = load_model(latest_model)
        print(f"✅ Model loaded - Accuracy: {metadata.get('training_accuracy', 0):.3f}")
        print(f"   Tree depth: {metadata.get('tree_depth', 0)}")
        print(f"   Features: {metadata.get('n_features', 0)}")
        
        # Load sample user data
        print("\n👥 Loading sample users...")
        data_client = WegathonDataClient()
        users = data_client.load_users(limit=200)
        print(f"   Loaded {len(users)} users")
        
        # Convert users to DataFrame format
        user_data = []
        for user in users:
            # Create decision factors text
            decision_factors = f"User in {user.segment.value} segment with "
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
                'segment': user.segment.value if hasattr(user.segment, 'value') else str(user.segment),
                'domain': user.domain.value if hasattr(user.domain, 'value') else str(user.domain),
                'churn_score': user.scores.churn_score,
                'activity_score': user.scores.activity_score,
                'cart_abandon_score': user.scores.cart_abandon_score,
                'price_sensitivity': user.scores.price_sensitivity,
                'family_score': user.scores.family_score,
                'is_oneway': user.is_oneway,
                'user_basket': user.user_basket,
                'recommended_discount_pct': 12,  # Default
                'expected_profit': 200.0,
                'expected_conversion': 0.6,
                'confidence_score': 0.75,
                'decision_factors': decision_factors,
            }
            user_data.append(user_dict)
        
        import pandas as pd
        df = pd.DataFrame(user_data)
        
        # Extract tags from decision factors
        from policy_tree.tags import extract_tags_batch
        df = extract_tags_batch(df)
        
        # Add chosen actions (simulate the pipeline)
        from policy_tree.feasible import choose_actions_batch
        df = choose_actions_batch(df)
        
        # Generate cohorts
        print("\n🎯 Generating cohorts...")
        cohorts = generate_cohorts(estimator, metadata, df, min_support=10)
        
        print(f"✅ Generated {len(cohorts)} cohorts")
        print("\n📊 Cohort Summary:")
        print("-" * 80)
        
        total_users_covered = 0
        total_expected_profit = 0
        
        for i, cohort in enumerate(cohorts, 1):
            print(f"\n{i}. {cohort['name']}")
            print(f"   Rule: {cohort['rule']}")
            print(f"   Action: {cohort['action']}%")
            print(f"   Size: {cohort['size']} users")
            print(f"   Avg Profit: ${cohort['avg_expected_profit']:.2f}")
            print(f"   Avg Conversion: {cohort['avg_expected_conversion']:.1%}")
            print(f"   Confidence: {cohort['mean_confidence']:.2f}")
            print(f"   Tags: {', '.join(cohort['why_tags'])}")
            
            total_users_covered += cohort['size']
            total_expected_profit += cohort['size'] * cohort['avg_expected_profit']
        
        print("\n" + "="*60)
        print("📈 Overall Statistics:")
        print(f"   Total Users Covered: {total_users_covered}/{len(df)} ({total_users_covered/len(df)*100:.1f}%)")
        print(f"   Total Expected Profit: ${total_expected_profit:,.2f}")
        print(f"   Avg Profit per User: ${total_expected_profit/total_users_covered:.2f}")
        print(f"   Cohorts Generated: {len(cohorts)}")
        
        print("\n✅ Cohort generation test completed successfully!")
        print("\n🚀 Ready for API usage:")
        print("   GET /policy/cohorts")
        print("   POST /policy/preview")
        print("   GET /policy/stats")
        
    except Exception as e:
        print(f"❌ Error testing cohorts: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
