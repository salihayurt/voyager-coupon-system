"""
Data client for WEGATHON 2025 Voyager dataset

Loads and processes the real WEGATHON dataset, creating user behavioral scores
and segments from the available transaction data.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
from core.domain.user import User, UserScores
from core.domain.enums import DomainType, SegmentType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WegathonDataClient:
    """Client for processing WEGATHON dataset into user profiles"""
    
    def __init__(self, csv_path: str = "data/WEGATHON_2025_VOYAGER_DATA_V2.csv"):
        self.csv_path = csv_path
        self._cached_users: Optional[List[User]] = None
        self._cached_df: Optional[pd.DataFrame] = None
    
    def load_raw_data(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """Load raw WEGATHON dataset"""
        logger.info(f"Loading WEGATHON data from {self.csv_path}")
        
        if self._cached_df is not None and nrows is None:
            return self._cached_df
        
        df = pd.read_csv(self.csv_path, sep=';', nrows=nrows)
        
        # Clean and process basic fields
        df['user_id'] = df['user_id'].fillna(0).astype(int)
        df['total_amount_masked'] = df['total_amount_masked'].fillna(0)
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['user_created_at'] = pd.to_datetime(df['user_created_at'], errors='coerce')
        
        # Map domains to our enum system
        df['domain_mapped'] = df['domain'].map(self._map_domain)
        
        if nrows is None:
            self._cached_df = df
        
        logger.info(f"Loaded {len(df)} records from WEGATHON dataset")
        return df
    
    def _map_domain(self, domain: str) -> DomainType:
        """Map WEGATHON domains to our domain enum"""
        domain_mapping = {
            'ENUYGUN_HOTEL': DomainType.ENUYGUN_HOTEL,
            'ENUYGUN_FLIGHT': DomainType.ENUYGUN_FLIGHT,
            'ENUYGUN_CAR_RENTAL': DomainType.ENUYGUN_CAR_RENTAL,
            'ENUYGUN_BUS': DomainType.ENUYGUN_BUS,
            'WINGIE_FLIGHT': DomainType.WINGIE_FLIGHT,
        }
        return domain_mapping.get(domain, DomainType.ENUYGUN_HOTEL)  # Default to hotel
    
    def _calculate_user_scores(self, user_df: pd.DataFrame) -> UserScores:
        """Calculate behavioral scores for a user from their transaction history"""
        
        # Churn score: based on recency of last transaction
        if len(user_df) > 0:
            last_transaction = user_df['created_at'].max()
            if pd.notna(last_transaction):
                days_since_last = (datetime.now() - last_transaction).days
                churn_score = min(1.0, days_since_last / 365.0)  # Scale by year
            else:
                churn_score = 0.8  # High churn if no valid dates
        else:
            churn_score = 1.0
        
        # Activity score: based on transaction frequency and recency
        transaction_count = len(user_df)
        if transaction_count == 0:
            activity_score = 0.0
        elif transaction_count == 1:
            activity_score = 0.3
        elif transaction_count <= 3:
            activity_score = 0.6
        else:
            activity_score = min(1.0, transaction_count / 10.0)
        
        # Cart abandon score: ratio of reservations to sales
        reservations = len(user_df[user_df['transaction_type'] == 'RESERVATION'])
        sales = len(user_df[user_df['transaction_type'] == 'SALE'])
        total_transactions = reservations + sales
        
        if total_transactions == 0:
            cart_abandon_score = 0.5  # Neutral
        else:
            cart_abandon_score = reservations / total_transactions
        
        # Price sensitivity: based on route_avg_price vs total_amount_masked
        price_ratios = []
        for _, row in user_df.iterrows():
            if pd.notna(row['route_avg_price']) and pd.notna(row['total_amount_masked']):
                if row['route_avg_price'] > 0:
                    ratio = row['total_amount_masked'] / row['route_avg_price']
                    price_ratios.append(ratio)
        
        if price_ratios:
            avg_price_ratio = np.mean(price_ratios)
            # Lower ratio = more price sensitive (chose cheaper options)
            price_sensitivity = max(0.0, min(1.0, 2.0 - avg_price_ratio))
        else:
            price_sensitivity = 0.5  # Neutral
        
        # Family score: based on adult_count and child_count
        total_adults = user_df['adult_count'].fillna(1).sum()
        total_children = user_df['child_count'].fillna(0).sum()
        total_travelers = total_adults + total_children
        
        if total_travelers <= 1:
            family_score = 1.0  # Solo traveler
        elif total_travelers == 2:
            family_score = 0.7  # Couple
        elif total_children > 0:
            family_score = 0.1  # Family with children
        else:
            family_score = 0.4  # Group travel
        
        return UserScores(
            churn_score=max(0.0, min(1.0, churn_score)),
            activity_score=max(0.0, min(1.0, activity_score)),
            cart_abandon_score=max(0.0, min(1.0, cart_abandon_score)),
            price_sensitivity=max(0.0, min(1.0, price_sensitivity)),
            family_score=max(0.0, min(1.0, family_score))
        )
    
    def _determine_segment(self, user_df: pd.DataFrame, scores: UserScores) -> SegmentType:
        """Determine user segment based on transaction history and scores"""
        
        total_amount = user_df['total_amount_masked'].sum()
        transaction_count = len(user_df)
        
        # High value customers: high spending, multiple transactions
        if total_amount > 20000 and transaction_count >= 3:
            return SegmentType.HIGH_VALUE_CUSTOMERS
        
        # Premium customers: very high spending regardless of frequency
        if total_amount > 50000:
            return SegmentType.PREMIUM_CUSTOMERS
        
        # At risk customers: high churn score or high cart abandon
        if scores.churn_score > 0.7 or scores.cart_abandon_score > 0.8:
            return SegmentType.AT_RISK_CUSTOMERS
        
        # Price sensitive: high price sensitivity score
        if scores.price_sensitivity > 0.7:
            return SegmentType.PRICE_SENSITIVE_CUSTOMERS
        
        # Default to standard customers
        return SegmentType.STANDARD_CUSTOMERS
    
    def load_users(self, limit: Optional[int] = None) -> List[User]:
        """Load and process users from WEGATHON dataset"""
        
        if self._cached_users is not None and limit is None:
            return self._cached_users
        
        # Load raw data
        df = self.load_raw_data(nrows=limit * 10 if limit else None)  # Load more rows to get enough unique users
        
        # Group by user_id to create user profiles
        users = []
        user_groups = df.groupby('user_id')
        
        processed_users = 0
        for user_id, user_df in user_groups:
            if limit and processed_users >= limit:
                break
                
            if user_id == 0 or pd.isna(user_id):
                continue  # Skip invalid user IDs
            
            try:
                # Calculate behavioral scores
                scores = self._calculate_user_scores(user_df)
                
                # Determine segment
                segment = self._determine_segment(user_df, scores)
                
                # Get most common domain for this user
                domain_counts = user_df['domain_mapped'].value_counts()
                primary_domain = domain_counts.index[0] if len(domain_counts) > 0 else DomainType.ENUYGUN_HOTEL
                
                # Get user characteristics
                is_oneway = int(user_df['is_oneway'].mode().iloc[0]) if len(user_df) > 0 else 1
                user_basket = int(user_df['user_basket'].mode().iloc[0]) if len(user_df) > 0 else 1
                
                # Get previous domains (unique domains user has used)
                previous_domains = list(user_df['domain_mapped'].unique())
                if len(previous_domains) <= 1:
                    previous_domains = None
                
                user = User(
                    user_id=int(user_id),
                    domain=primary_domain,
                    is_oneway=is_oneway,
                    user_basket=user_basket,
                    segment=segment,
                    scores=scores,
                    previous_domains=previous_domains
                )
                
                users.append(user)
                processed_users += 1
                
                if processed_users % 1000 == 0:
                    logger.info(f"Processed {processed_users} users...")
                    
            except Exception as e:
                logger.warning(f"Error processing user {user_id}: {e}")
                continue
        
        logger.info(f"Successfully created {len(users)} user profiles from WEGATHON data")
        
        if limit is None:
            self._cached_users = users
        
        return users
    
    def get_user_transaction_summary(self, user_id: int) -> Dict[str, Any]:
        """Get transaction summary for a specific user"""
        df = self.load_raw_data()
        user_df = df[df['user_id'] == user_id]
        
        if len(user_df) == 0:
            return {"error": f"No transactions found for user {user_id}"}
        
        return {
            "user_id": user_id,
            "total_transactions": len(user_df),
            "total_amount": user_df['total_amount_masked'].sum(),
            "transaction_types": user_df['transaction_type'].value_counts().to_dict(),
            "domains": user_df['domain'].value_counts().to_dict(),
            "date_range": {
                "first_transaction": user_df['created_at'].min(),
                "last_transaction": user_df['created_at'].max()
            },
            "destinations": {
                "cities": user_df['travel_end_city'].value_counts().head(5).to_dict(),
                "countries": user_df['travel_end_country'].value_counts().to_dict()
            },
            "travel_patterns": {
                "domestic_ratio": user_df['is_domestic'].mean(),
                "oneway_ratio": user_df['is_oneway'].mean(),
                "avg_adult_count": user_df['adult_count'].mean(),
                "avg_child_count": user_df['child_count'].mean()
            }
        }
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get overall dataset statistics"""
        df = self.load_raw_data()
        
        return {
            "total_records": len(df),
            "unique_users": df['user_id'].nunique(),
            "date_range": {
                "start": df['created_at'].min(),
                "end": df['created_at'].max()
            },
            "transaction_types": df['transaction_type'].value_counts().to_dict(),
            "domains": df['domain'].value_counts().to_dict(),
            "countries": df['travel_end_country'].value_counts().head(10).to_dict(),
            "total_revenue": df['total_amount_masked'].sum(),
            "avg_transaction_value": df['total_amount_masked'].mean(),
            "domestic_vs_international": {
                "domestic": (df['is_domestic'] == 1).sum(),
                "international": (df['is_domestic'] == 0).sum()
            }
        }
    
    def validate_data_quality(self) -> Dict[str, Any]:
        """Validate data quality and completeness"""
        df = self.load_raw_data(nrows=10000)  # Sample for validation
        
        validation_report = {
            "sample_size": len(df),
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.to_dict(),
            "unique_values": {
                col: df[col].nunique() for col in df.columns
            },
            "potential_issues": []
        }
        
        # Check for potential data quality issues
        if df['user_id'].isnull().sum() > 0:
            validation_report["potential_issues"].append("Missing user_id values")
        
        if df['total_amount_masked'].min() < 0:
            validation_report["potential_issues"].append("Negative transaction amounts")
        
        if df['created_at'].isnull().sum() > len(df) * 0.1:
            validation_report["potential_issues"].append("Many missing transaction dates")
        
        return validation_report
