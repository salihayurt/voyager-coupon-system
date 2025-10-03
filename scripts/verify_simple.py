#!/usr/bin/env python3
"""
Simplified verification script for Voyager Coupon System
Tests core components without external dependencies
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test if all core modules can be imported"""
    print("🔍 Testing core imports...")
    
    try:
        # Test domain models
        from core.domain.enums import DomainType, SegmentType, ActionType
        from core.domain.user import User, UserScores
        from core.domain.coupon import CouponDecision
        print("✅ Core domain models imported successfully")
        
        # Test basic functionality
        user_scores = UserScores(
            churn_score=0.5,
            activity_score=0.7,
            cart_abandon_score=0.3,
            price_sensitivity=0.8,
            family_score=0.4
        )
        
        user = User(
            user_id=12345,
            domain=DomainType.ENUYGUN_FLIGHT,
            is_oneway=1,
            user_basket=0,
            segment=SegmentType.PRICE_SENSITIVE_CUSTOMERS,
            scores=user_scores
        )
        
        print(f"✅ Created test user: {user.user_id}")
        
        # Test business rules
        from core.rules.business_rules import RulesEngine, LoginNotificationRule
        rules_engine = RulesEngine()
        rules = rules_engine.apply_rules(user)
        print(f"✅ Business rules working: {len(rules)} rules applied")
        
        # Test state encoder
        from learning.state_encoder import StateEncoder
        encoder = StateEncoder()  
        state = encoder.encode(user)
        readable = encoder.decode_readable(state)
        print(f"✅ State encoder working: state size = {len(state)}")
        print(f"   Readable state: {readable['segment']}, {readable['churn']}")
        
        # Test action space
        from learning.action_space import ActionSpace, DiscountAction
        discount = ActionSpace.get_discount_percentage(DiscountAction.DISCOUNT_10)
        print(f"✅ Action space working: 10% discount = {discount}%")
        
        # Test Q-table (basic initialization)
        from learning.q_table import QTable
        q_table = QTable(encoder.state_space_size, ActionSpace.total_actions())
        print(f"✅ Q-table initialized: {encoder.state_space_size:,} states, {ActionSpace.total_actions()} actions")
        
        return True
        
    except Exception as e:
        print(f"❌ Import/functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_structure():
    """Test if required files exist"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        "core/domain/enums.py",
        "core/domain/user.py", 
        "core/domain/coupon.py",
        "core/rules/business_rules.py",
        "learning/state_encoder.py",
        "learning/action_space.py",
        "learning/q_table.py",
        "learning/reward_function.py",
        "learning/trainer.py",
        "app/main.py",
        "scripts/train_qlearning.py",
        "scripts/make_splits.py",
        "Makefile",
        "pyproject.toml",
        "requirements.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True

def test_csv_structure():
    """Test CSV file structure"""
    print("\n📊 Testing CSV structure...")
    
    csv_path = project_root / "data" / "structured_customer_data.csv"
    
    if not csv_path.exists():
        print(f"⚠️  CSV file not found: {csv_path}")
        print("   This is expected if you haven't added your data yet")
        return True
    
    try:
        from adapters.external.data_analysis_client import DataAnalysisClient
        client = DataAnalysisClient(str(csv_path))
        validation = client.validate_csv_format()
        
        if validation['valid']:
            print(f"✅ CSV structure valid: {validation.get('rows', 'unknown')} rows")
            
            # Test loading a few users
            users = client.load_users(limit=5)
            print(f"✅ Successfully loaded {len(users)} sample users")
            
            if users:
                user = users[0]
                print(f"   Sample user: ID={user.user_id}, segment={user.segment.value}")
                
        else:
            print(f"❌ CSV validation failed: {validation['error']}")
            return False
            
    except ImportError as e:
        print(f"⚠️  Cannot test CSV (missing dependencies): {e}")
        return True
    except Exception as e:
        print(f"❌ CSV test failed: {e}")
        return False
    
    return True

def main():
    """Main verification function"""
    print("🚀 Voyager Coupon System - Verification Test")
    print("=" * 50)
    
    all_passed = True
    
    # Test file structure
    if not test_file_structure():
        all_passed = False
    
    # Test imports and basic functionality
    if not test_imports():
        all_passed = False
    
    # Test CSV structure (if available)
    if not test_csv_structure():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: make install")
        print("2. Split your data: make split")
        print("3. Train the model: make train")
        print("4. Start the API: make run")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
