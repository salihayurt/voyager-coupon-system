#!/usr/bin/env python3
"""
Test script to validate WEGATHON data loading and processing
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from adapters.external.wegathon_data_client import WegathonDataClient

def main():
    """Test WEGATHON data loading"""
    print("🔍 Testing WEGATHON Data Loading...")
    print("="*50)
    
    try:
        # Initialize client
        client = WegathonDataClient()
        
        # Test dataset statistics
        print("📊 Dataset Statistics:")
        stats = client.get_dataset_statistics()
        print(f"  Total records: {stats['total_records']:,}")
        print(f"  Unique users: {stats['unique_users']:,}")
        print(f"  Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        print(f"  Transaction types: {stats['transaction_types']}")
        print(f"  Domains: {stats['domains']}")
        print(f"  Total revenue: {stats['total_revenue']:,.2f}")
        print()
        
        # Test data quality
        print("🔍 Data Quality Check:")
        quality = client.validate_data_quality()
        print(f"  Sample size: {quality['sample_size']:,}")
        print(f"  Missing values: {sum(quality['missing_values'].values())}")
        if quality['potential_issues']:
            print(f"  Issues found: {quality['potential_issues']}")
        else:
            print("  ✅ No major issues detected")
        print()
        
        # Test user loading
        print("👥 Loading Sample Users:")
        users = client.load_users(limit=100)
        print(f"  Loaded {len(users)} users")
        
        if users:
            sample_user = users[0]
            print(f"  Sample user ID: {sample_user.user_id}")
            print(f"  Segment: {sample_user.segment.value}")
            print(f"  Domain: {sample_user.domain.value}")
            print(f"  Scores: churn={sample_user.scores.churn_score:.3f}, "
                  f"activity={sample_user.scores.activity_score:.3f}, "
                  f"price_sens={sample_user.scores.price_sensitivity:.3f}")
        
        # Test segment distribution
        print("\n📈 Segment Distribution:")
        segment_counts = {}
        for user in users:
            segment = user.segment.value
            segment_counts[segment] = segment_counts.get(segment, 0) + 1
        
        for segment, count in sorted(segment_counts.items()):
            percentage = (count / len(users)) * 100
            print(f"  {segment}: {count} ({percentage:.1f}%)")
        
        print("\n✅ WEGATHON data loading test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing WEGATHON data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
