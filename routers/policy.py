"""
FastAPI router for Policy Tree Cohorting endpoints

Provides REST API endpoints for cohort management and policy recommendations.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
import pandas as pd

from schemas.policy import (
    CohortListRequest, CohortListResponse, CohortRule,
    CohortPreviewRequest, CohortPreviewResponse, UserPreview,
    PolicyStatsResponse, ModelInfo, TrainingRequest, TrainingResponse,
    CohortFilters, ErrorResponse
)
from policy_tree import (
    load_cohorts_from_artifact, generate_cohorts, 
    filter_cohorts, get_cohort_preview, validate_cohorts
)
from policy_tree.train import train_pipeline, load_model
from adapters.external.data_analysis_client import DataAnalysisClient
from adapters.external.wegathon_data_client import WegathonDataClient

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/policy",
    tags=["Policy Tree Cohorts"],
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
        404: {"model": ErrorResponse, "description": "Resource not found"},
        400: {"model": ErrorResponse, "description": "Bad request"}
    }
)

# Global variables for caching
_cached_cohorts: Optional[List[Dict[str, Any]]] = None
_cached_model_metadata: Optional[Dict[str, Any]] = None
_cache_timestamp: Optional[datetime] = None
_model_path: str = "artifacts/policy_tree.joblib"

def _get_latest_model_path() -> str:
    """Find the latest model file"""
    artifacts_dir = Path("artifacts")
    if not artifacts_dir.exists():
        raise HTTPException(status_code=404, detail="No model artifacts found")
    
    # Look for policy tree models
    model_files = list(artifacts_dir.glob("policy_tree_*.joblib"))
    if not model_files:
        # Fallback to default name
        default_path = artifacts_dir / "policy_tree.joblib"
        if default_path.exists():
            return str(default_path)
        raise HTTPException(status_code=404, detail="No policy tree model found")
    
    # Return the most recent model
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    return str(latest_model)

def _load_user_data() -> pd.DataFrame:
    """Load user data for cohort generation"""
    try:
        # Use WEGATHON data client for real dataset
        data_client = WegathonDataClient()
        users = data_client.load_users(limit=10000)  # Load sample for faster processing
        
        # Convert to DataFrame
        user_data = []
        for user in users:
            user_dict = {
                'user_id': user.user_id,
                'segment': user.segment.value if hasattr(user.segment, 'value') else user.segment,
                'domain': user.domain.value if hasattr(user.domain, 'value') else user.domain,
                'churn_score': user.scores.churn_score,
                'activity_score': user.scores.activity_score,
                'cart_abandon_score': user.scores.cart_abandon_score,
                'price_sensitivity': user.scores.price_sensitivity,
                'family_score': user.scores.family_score,
                'is_oneway': user.is_oneway,
                'user_basket': user.user_basket,
                # Add placeholder values for missing fields
                'recommended_discount_pct': 10,  # Default
                'expected_profit': 200.0,  # Default
                'expected_conversion': 0.6,  # Default
                'confidence_score': 0.75,  # Default
                'decision_factors': f"User in {user.segment.value} segment with scores",
                'tags': []  # Will be populated by tag extraction
            }
            user_data.append(user_dict)
        
        return pd.DataFrame(user_data)
        
    except Exception as e:
        logger.error(f"Failed to load user data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load user data: {str(e)}")

def _refresh_cohorts_cache(force: bool = False) -> None:
    """Refresh the cohorts cache"""
    global _cached_cohorts, _cached_model_metadata, _cache_timestamp
    
    # Check if cache is still valid (refresh every hour)
    if not force and _cache_timestamp and _cached_cohorts:
        age = (datetime.now() - _cache_timestamp).seconds
        if age < 3600:  # 1 hour
            return
    
    try:
        # Load model and generate cohorts
        model_path = _get_latest_model_path()
        user_df = _load_user_data()
        
        cohorts, metadata = load_cohorts_from_artifact(model_path, user_df)
        
        _cached_cohorts = cohorts
        _cached_model_metadata = metadata
        _cache_timestamp = datetime.now()
        
        logger.info(f"Refreshed cohorts cache: {len(cohorts)} cohorts")
        
    except Exception as e:
        logger.error(f"Failed to refresh cohorts cache: {e}")
        # Keep existing cache if refresh fails
        if not _cached_cohorts:
            raise HTTPException(status_code=500, detail=f"Failed to load cohorts: {str(e)}")

@router.get("/cohorts", response_model=CohortListResponse)
async def get_cohorts(
    segment: Optional[str] = Query(None, description="Filter by segment"),
    domain: Optional[str] = Query(None, description="Filter by domain"), 
    min_size: Optional[int] = Query(None, ge=1, description="Minimum cohort size"),
    min_profit: Optional[float] = Query(None, description="Minimum average profit"),
    max_cohorts: int = Query(100, ge=1, le=1000, description="Maximum cohorts to return"),
    sort_by: str = Query("profit_impact", description="Sort by: profit_impact, size, conversion"),
    sort_desc: bool = Query(True, description="Sort descending")
) -> CohortListResponse:
    """
    Get list of policy cohorts with optional filtering
    
    Returns cohorts generated from the latest trained policy tree model.
    Cohorts represent groups of users with similar characteristics and
    recommended actions that respect segment constraints.
    """
    try:
        # Refresh cache if needed
        _refresh_cohorts_cache()
        
        if not _cached_cohorts:
            raise HTTPException(status_code=404, detail="No cohorts available")
        
        # Apply filters
        filtered_cohorts = _cached_cohorts.copy()
        
        if segment:
            filtered_cohorts = [
                c for c in filtered_cohorts 
                if segment.lower() in c.get('rule', '').lower()
            ]
        
        if domain:
            filtered_cohorts = [
                c for c in filtered_cohorts 
                if domain.lower() in c.get('rule', '').lower()
            ]
        
        if min_size:
            filtered_cohorts = [c for c in filtered_cohorts if c.get('size', 0) >= min_size]
        
        if min_profit:
            filtered_cohorts = [
                c for c in filtered_cohorts 
                if c.get('avg_expected_profit', 0) >= min_profit
            ]
        
        # Sort cohorts
        if sort_by == "profit_impact":
            key_func = lambda x: x.get('size', 0) * x.get('avg_expected_profit', 0)
        elif sort_by == "size":
            key_func = lambda x: x.get('size', 0)
        elif sort_by == "conversion":
            key_func = lambda x: x.get('avg_expected_conversion', 0)
        else:
            key_func = lambda x: x.get('avg_expected_profit', 0)
        
        filtered_cohorts.sort(key=key_func, reverse=sort_desc)
        
        # Limit results
        filtered_cohorts = filtered_cohorts[:max_cohorts]
        
        # Convert to response format
        cohort_rules = []
        for cohort in filtered_cohorts:
            cohort_rule = CohortRule(
                name=cohort.get('name', ''),
                rule=cohort.get('rule', ''),
                action=str(cohort.get('action', '')),
                size=cohort.get('size', 0),
                avg_expected_profit=cohort.get('avg_expected_profit', 0.0),
                avg_expected_conversion=cohort.get('avg_expected_conversion', 0.0),
                mean_confidence=cohort.get('mean_confidence', 0.0),
                why_tags=cohort.get('why_tags', []),
                leaf_id=cohort.get('leaf_id')
            )
            cohort_rules.append(cohort_rule)
        
        total_users = sum(c.size for c in cohort_rules)
        model_version = _cached_model_metadata.get('version') if _cached_model_metadata else None
        
        return CohortListResponse(
            cohorts=cohort_rules,
            total_cohorts=len(cohort_rules),
            total_users_covered=total_users,
            model_version=model_version,
            generated_at=_cache_timestamp.isoformat() if _cache_timestamp else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cohorts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cohorts: {str(e)}")

@router.post("/preview", response_model=CohortPreviewResponse)
async def get_cohort_preview(
    request: CohortPreviewRequest
) -> CohortPreviewResponse:
    """
    Get preview of users in a specific cohort
    
    Returns a sample of users that match the cohort criteria,
    useful for understanding the composition of a cohort.
    """
    try:
        # Refresh cache if needed
        _refresh_cohorts_cache()
        
        if not _cached_cohorts:
            raise HTTPException(status_code=404, detail="No cohorts available")
        
        # Find matching cohort
        target_cohort = None
        
        if request.cohort_name:
            target_cohort = next(
                (c for c in _cached_cohorts if c.get('name') == request.cohort_name),
                None
            )
        elif request.rule_conditions:
            # Find cohort with matching rule
            target_cohort = next(
                (c for c in _cached_cohorts 
                 if request.rule_conditions.lower() in c.get('rule', '').lower()),
                None
            )
        
        if not target_cohort:
            raise HTTPException(status_code=404, detail="Cohort not found")
        
        # Get user preview
        user_ids = target_cohort.get('user_ids', [])[:request.max_users]
        
        # Load user data for preview
        user_df = _load_user_data()
        preview_users = user_df[user_df['user_id'].isin(user_ids)]
        
        # Convert to response format
        user_previews = []
        for _, user in preview_users.iterrows():
            user_preview = UserPreview(
                user_id=int(user['user_id']),
                segment=user['segment'],
                domain=user['domain'],
                recommended_action=str(target_cohort.get('action', '')),
                expected_profit=float(user.get('expected_profit', 0)),
                expected_conversion=float(user.get('expected_conversion', 0)),
                confidence=float(user.get('confidence_score', 0.5)),
                tags=user.get('tags', []) if isinstance(user.get('tags'), list) else []
            )
            user_previews.append(user_preview)
        
        # Create cohort rule
        cohort_rule = CohortRule(
            name=target_cohort.get('name', ''),
            rule=target_cohort.get('rule', ''),
            action=str(target_cohort.get('action', '')),
            size=target_cohort.get('size', 0),
            avg_expected_profit=target_cohort.get('avg_expected_profit', 0.0),
            avg_expected_conversion=target_cohort.get('avg_expected_conversion', 0.0),
            mean_confidence=target_cohort.get('mean_confidence', 0.0),
            why_tags=target_cohort.get('why_tags', []),
            leaf_id=target_cohort.get('leaf_id')
        )
        
        return CohortPreviewResponse(
            cohort=cohort_rule,
            users=user_previews,
            total_users_in_cohort=target_cohort.get('size', 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cohort preview: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cohort preview: {str(e)}")

@router.get("/stats", response_model=PolicyStatsResponse)
async def get_policy_stats() -> PolicyStatsResponse:
    """
    Get policy system statistics and model information
    
    Returns comprehensive statistics about the current policy tree model,
    cohort distribution, and system performance metrics.
    """
    try:
        # Refresh cache if needed
        _refresh_cohorts_cache()
        
        if not _cached_cohorts or not _cached_model_metadata:
            raise HTTPException(status_code=404, detail="No model or cohorts available")
        
        # Model info
        model_info = ModelInfo(
            version=_cached_model_metadata.get('version', 'unknown'),
            training_date=_cached_model_metadata.get('training_timestamp', ''),
            n_samples=_cached_model_metadata.get('n_samples', 0),
            n_features=_cached_model_metadata.get('n_features', 0),
            tree_depth=_cached_model_metadata.get('tree_depth', 0),
            n_leaves=_cached_model_metadata.get('n_leaves', 0),
            training_accuracy=_cached_model_metadata.get('training_accuracy', 0.0)
        )
        
        # Cohort statistics
        total_users = sum(c.get('size', 0) for c in _cached_cohorts)
        avg_cohort_size = total_users / len(_cached_cohorts) if _cached_cohorts else 0
        
        cohort_stats = {
            'total_cohorts': len(_cached_cohorts),
            'avg_cohort_size': round(avg_cohort_size, 1),
            'total_users_covered': total_users,
            'avg_profit_per_cohort': round(
                sum(c.get('avg_expected_profit', 0) for c in _cached_cohorts) / len(_cached_cohorts), 2
            ) if _cached_cohorts else 0,
            'avg_conversion_per_cohort': round(
                sum(c.get('avg_expected_conversion', 0) for c in _cached_cohorts) / len(_cached_cohorts), 3
            ) if _cached_cohorts else 0
        }
        
        # Action distribution
        action_counts = {}
        segment_counts = {}
        
        for cohort in _cached_cohorts:
            action = str(cohort.get('action', 'unknown'))
            size = cohort.get('size', 0)
            
            action_counts[action] = action_counts.get(action, 0) + size
            
            # Extract segment from rule (simplified)
            rule = cohort.get('rule', '')
            for segment in ['at_risk_customers', 'high_value_customers', 'standard_customers', 
                           'price_sensitive_customers', 'premium_customers']:
                if segment in rule.lower():
                    segment_counts[segment] = segment_counts.get(segment, 0) + size
                    break
        
        return PolicyStatsResponse(
            model_info=model_info,
            cohort_stats=cohort_stats,
            action_distribution=action_counts,
            segment_coverage=segment_counts
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting policy stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get policy stats: {str(e)}")

@router.post("/retrain", response_model=TrainingResponse)
async def retrain_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
) -> TrainingResponse:
    """
    Trigger model retraining (async)
    
    Initiates retraining of the policy tree model with new data.
    Training runs in the background and updates the model artifacts.
    """
    try:
        # Load training data
        if request.data_source:
            # Load from specified source
            data_path = Path(request.data_source)
            if not data_path.exists():
                raise HTTPException(status_code=404, detail=f"Data source not found: {request.data_source}")
            
            if data_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(data_path)
            else:
                df = pd.read_csv(data_path)
        else:
            # Use default user data
            df = _load_user_data()
        
        # Check if recent model exists and force_retrain is False
        try:
            latest_model = _get_latest_model_path()
            model_age = (datetime.now() - datetime.fromtimestamp(
                Path(latest_model).stat().st_mtime
            )).days
            
            if not request.force_retrain and model_age < 1:
                return TrainingResponse(
                    status="skipped",
                    message=f"Recent model exists (age: {model_age} days). Use force_retrain=true to override.",
                    model_path=latest_model
                )
        except:
            pass  # No existing model, proceed with training
        
        # Generate output path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"artifacts/policy_tree_{timestamp}.joblib"
        
        # Run training pipeline
        start_time = datetime.now()
        
        saved_path, metrics = train_pipeline(
            df,
            output_path=output_path,
            max_depth=request.max_depth,
            min_samples_leaf_frac=request.min_samples_leaf_frac,
            test_size=request.test_size
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Clear cache to force refresh
        global _cached_cohorts, _cached_model_metadata, _cache_timestamp
        _cached_cohorts = None
        _cached_model_metadata = None
        _cache_timestamp = None
        
        # Generate cohorts to get count
        try:
            estimator, metadata = load_model(saved_path)
            cohorts = generate_cohorts(estimator, metadata, df)
            n_cohorts = len(cohorts)
        except:
            n_cohorts = None
        
        return TrainingResponse(
            status="success",
            model_path=saved_path,
            training_accuracy=metrics.get('validation_report', {}).get('test_accuracy'),
            test_accuracy=metrics.get('test_accuracy'),
            n_cohorts=n_cohorts,
            training_time=training_time,
            message=f"Model trained successfully with {len(df)} samples"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        return TrainingResponse(
            status="error",
            message=f"Training failed: {str(e)}"
        )

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check for policy tree system
    """
    try:
        # Check if model exists
        model_path = _get_latest_model_path()
        model_exists = True
        model_age_days = (datetime.now() - datetime.fromtimestamp(
            Path(model_path).stat().st_mtime
        )).days
        
    except:
        model_exists = False
        model_age_days = None
    
    # Check cache status
    cache_valid = _cached_cohorts is not None and _cache_timestamp is not None
    cache_age = None
    if _cache_timestamp:
        cache_age = (datetime.now() - _cache_timestamp).seconds
    
    return {
        "status": "healthy" if model_exists else "degraded",
        "model_exists": model_exists,
        "model_age_days": model_age_days,
        "cache_valid": cache_valid,
        "cache_age_seconds": cache_age,
        "cached_cohorts": len(_cached_cohorts) if _cached_cohorts else 0,
        "timestamp": datetime.now().isoformat()
    }
