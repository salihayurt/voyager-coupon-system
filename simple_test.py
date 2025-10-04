#!/usr/bin/env python3
"""
Simple direct test of the Policy Tree system without API server
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all imports work"""
    print("🧪 Testing Imports...")
    try:
        from adapters.external.wegathon_data_client import WegathonDataClient
        print("   ✅ WegathonDataClient imported")
        
        from policy_tree.train import load_model
        print("   ✅ Policy tree training imported")
        
        from policy_tree.inference import generate_cohorts
        print("   ✅ Policy tree inference imported")
        
        return True
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False

def test_data_loading():
    """Test WEGATHON data loading"""
    print("\n📊 Testing Data Loading...")
    try:
        from adapters.external.wegathon_data_client import WegathonDataClient
        client = WegathonDataClient()
        
        # Test basic stats
        stats = client.get_dataset_statistics()
        print(f"   ✅ Dataset loaded: {stats['total_records']:,} records")
        print(f"   ✅ Unique users: {stats['unique_users']:,}")
        
        # Test user loading
        users = client.load_users(limit=10)
        print(f"   ✅ User profiles: {len(users)} generated")
        
        return True
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        return False

def test_model_loading():
    """Test model loading"""
    print("\n🧠 Testing Model Loading...")
    try:
        import glob
        from policy_tree.train import load_model
        
        # Find trained model
        model_files = glob.glob("artifacts/policy_tree_wegathon_*.joblib")
        if not model_files:
            print("   ❌ No trained model found")
            print("   💡 Run: python scripts/train_policy_wegathon.py --limit 500")
            return False
        
        latest_model = max(model_files, key=lambda x: Path(x).stat().st_mtime)
        estimator, metadata = load_model(latest_model)
        
        print(f"   ✅ Model loaded: {Path(latest_model).name}")
        print(f"   ✅ Accuracy: {metadata.get('training_accuracy', 0):.3f}")
        print(f"   ✅ Tree depth: {metadata.get('tree_depth', 0)}")
        
        return True
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        return False

def test_basic_workflow():
    """Test basic cohort generation workflow"""
    print("\n🎯 Testing Basic Workflow...")
    try:
        # This is a minimal test without full feature pipeline
        from adapters.external.wegathon_data_client import WegathonDataClient
        
        client = WegathonDataClient()
        users = client.load_users(limit=5)
        
        print(f"   ✅ Loaded {len(users)} users")
        
        # Show sample user
        if users:
            user = users[0]
            print(f"   ✅ Sample user: ID={user.user_id}, Segment={user.segment.value}")
            print(f"   ✅ Scores: churn={user.scores.churn_score:.2f}, price_sens={user.scores.price_sensitivity:.2f}")
        
        return True
    except Exception as e:
        print(f"   ❌ Workflow test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Simple Policy Tree System Test")
    print("="*50)
    
    tests = [
        ("Imports", test_imports),
        ("Data Loading", test_data_loading), 
        ("Model Loading", test_model_loading),
        ("Basic Workflow", test_basic_workflow)
    ]
    
    passed = 0
    for name, test_func in tests:
        if test_func():
            passed += 1
    
    print("\n" + "="*50)
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! System is working correctly.")
        print("\n🚀 Next steps:")
        print("   1. Train model: python scripts/train_policy_wegathon.py --limit 500")
        print("   2. Start server: python start_server.py")
        print("   3. Test API: python test_local.py")
        print("   4. Visit: http://localhost:8080/docs")
    else:
        print(f"\n⚠️  {len(tests)-passed} tests failed. Check the errors above.")
        
        if passed >= 2:
            print("\n💡 Core system working! Issues might be with:")
            print("   • Model training (run training script)")
            print("   • API server (check imports)")

if __name__ == "__main__":
    main()
