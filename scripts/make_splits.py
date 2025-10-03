#!/usr/bin/env python3
"""
Data Splitting Script for Voyager Coupon System

Splits structured customer data into train/validation/test sets (70/15/15).
Ensures balanced distribution across segments and domains.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from adapters.external.data_analysis_client import DataAnalysisClient
from core.domain.enums import DomainType, SegmentType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_data_distribution(df: pd.DataFrame, name: str):
    """Analyze and log data distribution"""
    logger.info(f"📊 {name} Dataset Analysis:")
    logger.info(f"   Total records: {len(df):,}")
    
    # Segment distribution
    segment_dist = df['segment'].value_counts()
    logger.info("   Segment distribution:")
    for segment, count in segment_dist.items():
        percentage = (count / len(df)) * 100
        logger.info(f"     {segment}: {count:,} ({percentage:.1f}%)")
    
    # Domain distribution
    domain_dist = df['domain'].value_counts()
    logger.info("   Domain distribution:")
    for domain, count in domain_dist.items():
        percentage = (count / len(df)) * 100
        logger.info(f"     {domain}: {count:,} ({percentage:.1f}%)")
    
    # Score statistics
    score_cols = ['churn_score', 'activity_score', 'cart_abandon_score', 'price_sensitivity', 'family_score']
    logger.info("   Score statistics:")
    for col in score_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            logger.info(f"     {col}: μ={mean_val:.3f}, σ={std_val:.3f}")

def stratified_split(df: pd.DataFrame, test_size: float, random_state: int = 42):
    """
    Perform stratified split maintaining distribution across segments and domains
    """
    # Create stratification column combining segment and domain
    df['strat_col'] = df['segment'].astype(str) + '_' + df['domain'].astype(str)
    
    # Calculate minimum group size
    min_group_size = df['strat_col'].value_counts().min()
    logger.info(f"Minimum group size: {min_group_size}")
    
    if min_group_size < 10:
        logger.warning("Some segment-domain combinations have very few samples. Using simple random split.")
        # Fall back to simple random split
        train_df, test_df = train_test_split(
            df.drop('strat_col', axis=1), 
            test_size=test_size, 
            random_state=random_state,
            shuffle=True
        )
        return train_df, test_df
    
    try:
        # Stratified split
        train_df, test_df = train_test_split(
            df.drop('strat_col', axis=1),
            test_size=test_size,
            stratify=df['strat_col'],
            random_state=random_state,
            shuffle=True
        )
        logger.info("✅ Stratified split successful")
        return train_df, test_df
        
    except ValueError as e:
        logger.warning(f"Stratified split failed: {e}. Using simple random split.")
        # Fall back to simple random split
        train_df, test_df = train_test_split(
            df.drop('strat_col', axis=1),
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )
        return train_df, test_df

def validate_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Validate that splits maintain reasonable distributions"""
    logger.info("🔍 Validating data splits...")
    
    total_records = len(train_df) + len(val_df) + len(test_df)
    train_pct = len(train_df) / total_records * 100
    val_pct = len(val_df) / total_records * 100
    test_pct = len(test_df) / total_records * 100
    
    logger.info(f"Split percentages: Train={train_pct:.1f}%, Val={val_pct:.1f}%, Test={test_pct:.1f}%")
    
    # Check for data leakage (overlapping user_ids)
    train_users = set(train_df['user_id'])
    val_users = set(val_df['user_id'])
    test_users = set(test_df['user_id'])
    
    train_val_overlap = len(train_users & val_users)
    train_test_overlap = len(train_users & test_users)
    val_test_overlap = len(val_users & test_users)
    
    if train_val_overlap > 0 or train_test_overlap > 0 or val_test_overlap > 0:
        logger.error(f"❌ Data leakage detected! Overlaps: train-val={train_val_overlap}, train-test={train_test_overlap}, val-test={val_test_overlap}")
        return False
    
    logger.info("✅ No data leakage detected")
    
    # Check segment distribution similarity
    for segment in train_df['segment'].unique():
        train_pct = (train_df['segment'] == segment).mean() * 100
        val_pct = (val_df['segment'] == segment).mean() * 100
        test_pct = (test_df['segment'] == segment).mean() * 100
        
        max_diff = max(abs(train_pct - val_pct), abs(train_pct - test_pct), abs(val_pct - test_pct))
        if max_diff > 5.0:  # More than 5% difference
            logger.warning(f"⚠️  Large distribution difference for {segment}: Train={train_pct:.1f}%, Val={val_pct:.1f}%, Test={test_pct:.1f}%")
        else:
            logger.info(f"✅ {segment} distribution balanced across splits")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Split customer data into train/validation/test sets')
    parser.add_argument('--csv', required=True, help='Path to structured customer data CSV')
    parser.add_argument('--output-dir', default='data', help='Output directory for split files')
    parser.add_argument('--train-size', type=float, default=0.7, help='Training set size (default: 0.7)')
    parser.add_argument('--val-size', type=float, default=0.15, help='Validation set size (default: 0.15)')
    parser.add_argument('--test-size', type=float, default=0.15, help='Test set size (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--validate', action='store_true', help='Validate data quality before splitting')
    
    args = parser.parse_args()
    
    # Validate split sizes
    if abs(args.train_size + args.val_size + args.test_size - 1.0) > 0.001:
        logger.error("❌ Split sizes must sum to 1.0")
        return 1
    
    # Set random seed
    np.random.seed(args.seed)
    
    logger.info("🚀 Starting data splitting for Voyager Coupon System")
    logger.info(f"Split configuration: Train={args.train_size:.1%}, Val={args.val_size:.1%}, Test={args.test_size:.1%}")
    
    try:
        # Load and validate data
        logger.info(f"📂 Loading data from {args.csv}")
        df = pd.read_csv(args.csv)
        
        if len(df) == 0:
            logger.error("❌ CSV file is empty")
            return 1
        
        logger.info(f"✅ Loaded {len(df):,} records")
        
        # Validate data quality if requested
        if args.validate:
            logger.info("🔍 Validating data quality...")
            client = DataAnalysisClient(args.csv)
            validation_result = client.validate_csv_format()
            
            if not validation_result.get('valid', False):
                logger.error(f"❌ Data validation failed: {validation_result}")
                return 1
            
            logger.info("✅ Data validation passed")
        
        # Analyze original data distribution
        analyze_data_distribution(df, "Original")
        
        # Shuffle data by user_id to ensure randomness
        logger.info("🔀 Shuffling data...")
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        
        # First split: separate test set
        logger.info("✂️  Creating initial train/test split...")
        train_val_df, test_df = stratified_split(df, test_size=args.test_size, random_state=args.seed)
        
        # Second split: separate validation from training
        logger.info("✂️  Creating train/validation split...")
        val_size_adjusted = args.val_size / (args.train_size + args.val_size)
        train_df, val_df = stratified_split(train_val_df, test_size=val_size_adjusted, random_state=args.seed + 1)
        
        # Validate splits
        if not validate_splits(train_df, val_df, test_df):
            logger.error("❌ Split validation failed")
            return 1
        
        # Analyze split distributions
        analyze_data_distribution(train_df, "Training")
        analyze_data_distribution(val_df, "Validation")
        analyze_data_distribution(test_df, "Test")
        
        # Save splits
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        train_path = output_dir / 'train_data.csv'
        val_path = output_dir / 'val_data.csv'
        test_path = output_dir / 'test_data.csv'
        
        logger.info("💾 Saving split files...")
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"✅ Files saved:")
        logger.info(f"   Training: {train_path} ({len(train_df):,} records)")
        logger.info(f"   Validation: {val_path} ({len(val_df):,} records)")
        logger.info(f"   Test: {test_path} ({len(test_df):,} records)")
        
        # Create metadata file
        metadata = {
            'original_file': args.csv,
            'total_records': len(df),
            'train_records': len(train_df),
            'val_records': len(val_df),
            'test_records': len(test_df),
            'train_size': args.train_size,
            'val_size': args.val_size,
            'test_size': args.test_size,
            'random_seed': args.seed,
            'split_timestamp': pd.Timestamp.now().isoformat()
        }
        
        metadata_path = output_dir / 'split_metadata.json'
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"📋 Metadata saved: {metadata_path}")
        logger.info("🎉 Data splitting completed successfully!")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Data splitting failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
