"""
Unit tests for Policy Tree Cohorting module
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

from policy_tree.constraints import (
    get_allowed_actions, is_premium_reward_eligible, 
    snap_to_allowed_discount, validate_action
)
from policy_tree.feasible import choose_action, choose_actions_batch, validate_action_selection
from policy_tree.tags import make_tags, extract_tags_batch
from policy_tree.features import bin_scores, create_one_hot_features, build_features
from core.domain.enums import SegmentType, DomainType

class TestConstraints:
    """Test segment constraints and validation"""
    
    def test_get_allowed_actions_enum(self):
        """Test getting allowed actions for SegmentType enum"""
        allowed = get_allowed_actions(SegmentType.AT_RISK_CUSTOMERS)
        assert allowed == {10, 11, 12, 13}
        
        allowed = get_allowed_actions(SegmentType.HIGH_VALUE_CUSTOMERS)
        assert allowed == {5, 6, 7, 8, "premium_reward"}
    
    def test_get_allowed_actions_string(self):
        """Test getting allowed actions for string segment"""
        allowed = get_allowed_actions("AT_RISK_CUSTOMERS")
        assert allowed == {10, 11, 12, 13}
        
        allowed = get_allowed_actions("PREMIUM_CUSTOMERS")
        assert allowed == {5, 6, 7, 8, "premium_reward"}
    
    def test_premium_reward_eligibility(self):
        """Test premium reward eligibility logic"""
        # Eligible: HIGH_VALUE with low price sensitivity
        assert is_premium_reward_eligible(SegmentType.HIGH_VALUE_CUSTOMERS, 0.3)
        
        # Not eligible: HIGH_VALUE with high price sensitivity  
        assert not is_premium_reward_eligible(SegmentType.HIGH_VALUE_CUSTOMERS, 0.5)
        
        # Not eligible: AT_RISK regardless of price sensitivity
        assert not is_premium_reward_eligible(SegmentType.AT_RISK_CUSTOMERS, 0.2)
    
    def test_snap_to_allowed_discount(self):
        """Test snapping discount to nearest allowed value"""
        # Snap 14% to 13% for AT_RISK (closest allowed)
        snapped = snap_to_allowed_discount(14, SegmentType.AT_RISK_CUSTOMERS)
        assert snapped == 13
        
        # Snap 9% to 10% for AT_RISK  
        snapped = snap_to_allowed_discount(9, SegmentType.AT_RISK_CUSTOMERS)
        assert snapped == 10
        
        # Snap 4% to 5% for HIGH_VALUE
        snapped = snap_to_allowed_discount(4, SegmentType.HIGH_VALUE_CUSTOMERS)
        assert snapped == 5
    
    def test_validate_action(self):
        """Test action validation"""
        # Valid actions
        assert validate_action(12, SegmentType.AT_RISK_CUSTOMERS)
        assert validate_action("premium_reward", SegmentType.HIGH_VALUE_CUSTOMERS)
        
        # Invalid actions
        assert not validate_action(20, SegmentType.AT_RISK_CUSTOMERS)  # Too high
        assert not validate_action("premium_reward", SegmentType.AT_RISK_CUSTOMERS)  # Not allowed

class TestFeasibleActions:
    """Test feasible action selection"""
    
    def create_sample_user_row(self, **kwargs):
        """Create sample user data row"""
        defaults = {
            'segment': SegmentType.STANDARD_CUSTOMERS,
            'recommended_discount_pct': 10,
            'expected_profit': 200.0,
            'confidence_score': 0.75,
            'price_sensitivity': 0.5,
            'options': []
        }
        defaults.update(kwargs)
        return pd.Series(defaults)
    
    def test_choose_action_basic(self):
        """Test basic action selection"""
        user_row = self.create_sample_user_row()
        action, utility = choose_action(user_row)
        
        # Should return valid action for STANDARD_CUSTOMERS
        assert action in {5, 6, 7, 8, 9, 10}
        assert utility > 0
    
    def test_choose_action_with_options(self):
        """Test action selection with provided options"""
        user_row = self.create_sample_user_row(
            options=[{'discount': 8}, {'discount': 12}]  # 12 not allowed for STANDARD
        )
        action, utility = choose_action(user_row)
        
        # Should choose 8% (allowed option)
        assert action == 8
    
    def test_choose_action_premium_reward(self):
        """Test premium reward selection"""
        user_row = self.create_sample_user_row(
            segment=SegmentType.HIGH_VALUE_CUSTOMERS,
            price_sensitivity=0.3,  # Low enough for premium reward
            expected_profit=500.0
        )
        action, utility = choose_action(user_row)
        
        # Could be premium_reward or regular discount
        assert action in {5, 6, 7, 8, "premium_reward"}
    
    def test_choose_actions_batch(self):
        """Test batch action selection"""
        df = pd.DataFrame([
            {
                'segment': SegmentType.STANDARD_CUSTOMERS,
                'recommended_discount_pct': 10,
                'expected_profit': 200.0,
                'confidence_score': 0.75,
                'price_sensitivity': 0.5
            },
            {
                'segment': SegmentType.AT_RISK_CUSTOMERS,
                'recommended_discount_pct': 15,
                'expected_profit': 150.0,
                'confidence_score': 0.8,
                'price_sensitivity': 0.7
            }
        ])
        
        result_df = choose_actions_batch(df)
        
        assert 'chosen_action' in result_df.columns
        assert 'utility_score' in result_df.columns
        assert len(result_df) == 2
    
    def test_validate_action_selection_valid(self):
        """Test validation of valid action selection"""
        df = pd.DataFrame([
            {'segment': SegmentType.STANDARD_CUSTOMERS, 'chosen_action': 8},
            {'segment': SegmentType.AT_RISK_CUSTOMERS, 'chosen_action': 12}
        ])
        
        report = validate_action_selection(df)
        assert report['is_valid']
        assert report['violations'] == 0
    
    def test_validate_action_selection_invalid(self):
        """Test validation with invalid actions"""
        df = pd.DataFrame([
            {'segment': SegmentType.STANDARD_CUSTOMERS, 'chosen_action': 15},  # Invalid
            {'segment': SegmentType.AT_RISK_CUSTOMERS, 'chosen_action': 12}   # Valid
        ])
        
        report = validate_action_selection(df)
        assert not report['is_valid']
        assert report['violations'] == 1
        assert report['violation_rate'] == 0.5

class TestTags:
    """Test tag extraction from decision factors"""
    
    def test_make_tags_score_based(self):
        """Test tag extraction based on scores"""
        scores = {
            'churn_score': 0.8,
            'price_sensitivity': 0.7,
            'activity_score': 0.2,
            'family_score': 0.3
        }
        
        tags = make_tags("", scores)
        
        assert 'high_churn' in tags
        assert 'high_price_sens' in tags
        assert 'low_activity' in tags
        assert 'family_pattern' in tags
    
    def test_make_tags_keyword_based(self):
        """Test tag extraction based on keywords"""
        decision_factors = "User shows high churn risk and is price sensitive. Last minute booking."
        scores = {'churn_score': 0.5, 'price_sensitivity': 0.5}
        
        tags = make_tags(decision_factors, scores)
        
        assert 'high_churn' in tags  # From keyword
        assert 'high_price_sens' in tags  # From keyword
        assert 'time_urgent' in tags  # From "last minute"
    
    def test_make_tags_segment_based(self):
        """Test tag extraction based on segment"""
        tags = make_tags("", {}, segment="HIGH_VALUE_CUSTOMERS")
        assert 'loyal_high_value' in tags
        
        tags = make_tags("", {}, segment="STANDARD_CUSTOMERS")
        assert 'loyal_high_value' not in tags
    
    def test_extract_tags_batch(self):
        """Test batch tag extraction"""
        df = pd.DataFrame([
            {
                'decision_factors': 'High churn risk user',
                'churn_score': 0.8,
                'price_sensitivity': 0.4,
                'activity_score': 0.6,
                'cart_abandon_score': 0.3,
                'family_score': 0.2,
                'segment': 'HIGH_VALUE_CUSTOMERS'
            }
        ])
        
        result_df = extract_tags_batch(df)
        
        assert 'tags' in result_df.columns
        tags = result_df.iloc[0]['tags']
        assert isinstance(tags, list)
        assert 'high_churn' in tags
        assert 'loyal_high_value' in tags

class TestFeatures:
    """Test feature engineering"""
    
    def create_sample_dataframe(self):
        """Create sample DataFrame for testing"""
        return pd.DataFrame([
            {
                'segment': SegmentType.STANDARD_CUSTOMERS,
                'domain': DomainType.ENUYGUN_FLIGHT,
                'churn_score': 0.3,
                'price_sensitivity': 0.7,
                'activity_score': 0.8,
                'tags': ['high_price_sens', 'low_activity']
            },
            {
                'segment': SegmentType.AT_RISK_CUSTOMERS,
                'domain': DomainType.ENUYGUN_HOTEL,
                'churn_score': 0.9,
                'price_sensitivity': 0.4,
                'activity_score': 0.2,
                'tags': ['high_churn']
            }
        ])
    
    def test_bin_scores(self):
        """Test score binning"""
        df = self.create_sample_dataframe()
        binned_df, metadata = bin_scores(df, ['churn_score', 'price_sensitivity'])
        
        assert 'churn_score_binned' in binned_df.columns
        assert 'price_sensitivity_binned' in binned_df.columns
        assert 'churn_score' in metadata
        assert 'price_sensitivity' in metadata
    
    def test_create_one_hot_features(self):
        """Test one-hot encoding"""
        df = self.create_sample_dataframe()
        one_hot_df, feature_names = create_one_hot_features(df, ['segment', 'domain'])
        
        # Check that one-hot columns were created
        segment_cols = [col for col in one_hot_df.columns if col.startswith('segment_')]
        domain_cols = [col for col in one_hot_df.columns if col.startswith('domain_')]
        
        assert len(segment_cols) > 0
        assert len(domain_cols) > 0
        assert len(feature_names) > 0
    
    def test_build_features(self):
        """Test complete feature building"""
        df = self.create_sample_dataframe()
        feature_df, metadata = build_features(df)
        
        # Check metadata structure
        assert 'score_bins' in metadata
        assert 'categorical_features' in metadata
        assert 'tag_features' in metadata
        assert 'feature_columns' in metadata
        
        # Check that features were created
        assert len(metadata['feature_columns']) > 0

class TestIntegration:
    """Integration tests for complete pipeline"""
    
    def create_full_sample_data(self):
        """Create comprehensive sample data"""
        return pd.DataFrame([
            {
                'user_id': 1,
                'segment': SegmentType.STANDARD_CUSTOMERS,
                'domain': DomainType.ENUYGUN_FLIGHT,
                'churn_score': 0.3,
                'activity_score': 0.8,
                'cart_abandon_score': 0.2,
                'price_sensitivity': 0.7,
                'family_score': 0.4,
                'recommended_discount_pct': 10,
                'expected_profit': 200.0,
                'expected_conversion': 0.6,
                'confidence_score': 0.75,
                'decision_factors': 'Price sensitive user with good activity',
                'options': []
            },
            {
                'user_id': 2,
                'segment': SegmentType.AT_RISK_CUSTOMERS,
                'domain': DomainType.ENUYGUN_HOTEL,
                'churn_score': 0.9,
                'activity_score': 0.2,
                'cart_abandon_score': 0.8,
                'price_sensitivity': 0.8,
                'family_score': 0.2,
                'recommended_discount_pct': 15,
                'expected_profit': 150.0,
                'expected_conversion': 0.4,
                'confidence_score': 0.8,
                'decision_factors': 'High churn risk customer needs retention',
                'options': []
            }
        ])
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from data to features"""
        df = self.create_full_sample_data()
        
        # Step 1: Extract tags
        df = extract_tags_batch(df)
        assert 'tags' in df.columns
        
        # Step 2: Choose feasible actions
        df = choose_actions_batch(df)
        assert 'chosen_action' in df.columns
        assert 'utility_score' in df.columns
        
        # Step 3: Validate actions
        validation_report = validate_action_selection(df)
        assert validation_report['is_valid']  # Should be valid
        
        # Step 4: Build features
        feature_df, metadata = build_features(df)
        assert len(metadata['feature_columns']) > 0
        
        # Verify no illegal actions were selected
        for _, row in df.iterrows():
            segment = row['segment']
            action = row['chosen_action']
            assert validate_action(action, segment)

if __name__ == "__main__":
    pytest.main([__file__])
