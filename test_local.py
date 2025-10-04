#!/usr/bin/env python3
"""
Local testing script for Policy Tree API
"""

import sys
import time
import requests
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_endpoint(url, method="GET", data=None, description=""):
    """Test a single API endpoint"""
    try:
        print(f"\n🧪 Testing: {description}")
        print(f"   {method} {url}")
        
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        else:
            print(f"   ❌ Unsupported method: {method}")
            return False
            
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success")
            
            # Show sample of response
            if isinstance(result, dict):
                if len(result) <= 3:
                    print(f"   Response: {result}")
                else:
                    # Show first few keys for large responses
                    sample_keys = list(result.keys())[:3]
                    sample = {k: result[k] for k in sample_keys}
                    print(f"   Sample: {sample}...")
            elif isinstance(result, list):
                print(f"   Items: {len(result)}")
                if result:
                    print(f"   First item keys: {list(result[0].keys()) if isinstance(result[0], dict) else 'N/A'}")
            
            return True
        else:
            print(f"   ❌ Failed: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Connection failed - Is the server running?")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Run local API tests"""
    print("🧪 Local API Testing for Voyager Policy Tree")
    print("="*60)
    
    base_url = "http://localhost:8080"
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(2)
    
    # Test endpoints
    tests = [
        {
            "url": f"{base_url}/health",
            "method": "GET",
            "description": "Health Check"
        },
        {
            "url": f"{base_url}/policy/health", 
            "method": "GET",
            "description": "Policy Health Check"
        },
        {
            "url": f"{base_url}/policy/stats",
            "method": "GET", 
            "description": "Policy Statistics"
        },
        {
            "url": f"{base_url}/policy/cohorts",
            "method": "GET",
            "description": "List Cohorts"
        },
        {
            "url": f"{base_url}/policy/cohorts?min_size=50",
            "method": "GET",
            "description": "List Cohorts (filtered)"
        }
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        success = test_endpoint(
            test["url"], 
            test["method"], 
            test.get("data"),
            test["description"]
        )
        if success:
            passed += 1
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Results Summary")
    print("="*60)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total-passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! API is working correctly.")
        print("\n🔗 Try these URLs in your browser:")
        print(f"   • {base_url}/health")
        print(f"   • {base_url}/policy/cohorts")
        print(f"   • {base_url}/policy/stats")
    else:
        print(f"\n⚠️  Some tests failed. Check server logs for details.")
    
    print("\n📋 Manual Testing:")
    print("   Open your browser and visit:")
    print(f"   {base_url}/docs (FastAPI documentation)")
    print(f"   {base_url}/policy/cohorts (JSON response)")

if __name__ == "__main__":
    main()
