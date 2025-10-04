#!/usr/bin/env python3
"""
Simple server startup script for local testing
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Start the FastAPI server for local testing"""
    print("🚀 Starting Voyager Coupon System with Policy Tree...")
    print("="*60)
    
    try:
        # Import after path setup
        from app.main import app
        import uvicorn
        
        print("✅ Imports successful")
        print("📡 Starting server at http://localhost:8080")
        print("📋 Available endpoints:")
        print("   • GET  /health - Health check")
        print("   • POST /recommend - Individual recommendations") 
        print("   • GET  /policy/cohorts - List cohorts")
        print("   • POST /policy/preview - Preview cohort users")
        print("   • GET  /policy/stats - System statistics")
        print("   • GET  /policy/health - Policy system health")
        print("\n🔄 Starting server...")
        print("   Press Ctrl+C to stop")
        print("="*60)
        
        # Start server
        uvicorn.run(
            app,
            host="localhost",
            port=8080,
            log_level="info",
            reload=False  # Disable reload for stability
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n🔧 Try installing dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
