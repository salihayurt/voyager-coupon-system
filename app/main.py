from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
from contextlib import asynccontextmanager

# Import all system components
from adapters.external.data_analysis_client import DataAnalysisClient
from orchestration.workflow_engine import WorkflowEngine
from learning.reward_function import RewardCalculator
from core.rules.business_rules import RulesEngine
from core.domain.user import User, UserScores
from core.domain.coupon import CouponDecision
from agents.profitability_agent.agent import ProfitabilityAgent
from agents.conversion_agent.agent import ConversionAgent
from agents.coordinator_agent.agent import CoordinatorAgent
from core.domain.enums import SegmentType, DomainType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for system components
workflow_engine: Optional[WorkflowEngine] = None
data_client: Optional[DataAnalysisClient] = None
user_index: Optional[dict] = None

# Pydantic models for API
class RecommendRequest(BaseModel):
    user_id: Optional[int] = None
    user: Optional[User] = None
    user_ids: Optional[List[int]] = None
class SingleUserOptionsRequest(BaseModel):
    user_id: int

class SegmentRecommendationRequest(BaseModel):
    segment: SegmentType
    domain: Optional[DomainType] = None
    sample_size: int = 100


class RecommendResponse(BaseModel):
    user_id: int
    action_type: str
    discount_percentage: int
    target_domain: Optional[str]
    expected_conversion_rate: float
    expected_profit: float
    reasoning: List[str]
    agent_votes: Dict[str, int]
    confidence: float

class ExplainRequest(BaseModel):
    user: User
    discount: int

class ExplainResponse(BaseModel):
    user_id: int
    discount_percentage: int
    explanation: Dict[str, Any]
    feature_importance: Dict[str, float]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize system components on startup"""
    global workflow_engine, data_client
    
    logger.info("🚀 Initializing Voyager Coupon System...")
    
    try:
        # Initialize data client
        data_client = DataAnalysisClient()
        logger.info("✅ Data client initialized")
        # Preload and index users for fast lookups
        try:
            users = data_client.load_users()
            global user_index
            user_index = {u.user_id: u for u in users}
            logger.info(f"✅ Cached {len(user_index)} users in memory")
        except Exception as e:
            logger.warning(f"⚠️ Could not preload users: {e}")
        
        # Remove Q-Learning from startup in segment-based architecture
        state_encoder = None
        q_table = None
        
        # Initialize business rules
        rules_engine = RulesEngine()
        logger.info(f"✅ Business rules loaded - {len(rules_engine.rules)} rules")
        
        # Initialize agents
        prof_agent = ProfitabilityAgent()
        conv_agent = ConversionAgent()
        coord_agent = CoordinatorAgent()
        logger.info("✅ AI agents initialized")
        
        # Create workflow engine (segment-based)
        workflow_engine = WorkflowEngine(
            rules_engine=rules_engine,
            profitability_agent=prof_agent,
            conversion_agent=conv_agent,
            coordinator_agent=coord_agent
        )
        # Attach data_client for segment-level ops via app globals
        app.state.data_client = data_client
        logger.info("✅ Workflow engine ready")
        logger.info("🎉 Voyager Coupon System startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("🔄 Shutting down Voyager Coupon System...")

# Create FastAPI app
app = FastAPI(
    title="Voyager Coupon System API",
    description="AI-powered coupon recommendation system using Q-Learning and multi-agent coordination",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "system": "Voyager Coupon System",
        "version": "1.0.0",
        "components": {
            "workflow_engine": workflow_engine is not None,
            "data_client": data_client is not None
        }
    }

@app.post("/recommend", response_model=RecommendResponse)
async def recommend_coupon(request: RecommendRequest):
    """
    Get coupon recommendation for a user
    
    Input: user_id or full User JSON
    Returns: Recommendation with discount, reasoning, and metrics
    """
    if workflow_engine is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Get user data
        if request.user:
            user = request.user
        elif request.user_id:
            if data_client is None:
                raise HTTPException(status_code=503, detail="Data client not available")
            
            # Load user from in-memory index if available
            user = user_index.get(request.user_id) if user_index else None
            if user is None:
                # Fallback to loading from source
                users = data_client.load_users()
                user = next((u for u in users if u.user_id == request.user_id), None)
            if not user:
                raise HTTPException(status_code=404, detail=f"User {request.user_id} not found")
        else:
            raise HTTPException(status_code=400, detail="Either user_id or user must be provided")
        
        # Process through workflow
        logger.info(f"Processing recommendation for user {user.user_id}")
        decision = workflow_engine.process_user(user)
        
        # Get detailed processing info for agent votes (segment-based)
        detailed_result = workflow_engine.process_user_with_details(user)
        proposals = detailed_result.get("agent_proposals", {})
        agent_votes = {
            "profit": proposals.get("profitability", {}).get("discount", 0),
            "conversion": proposals.get("conversion", {}).get("discount", 0),
            "coordinator": proposals.get("coordination", {}).get("discount", decision.discount_percentage),
        }
        
        # Build response
        response = RecommendResponse(
            user_id=user.user_id,
            action_type=decision.action_type.value,
            discount_percentage=decision.discount_percentage,
            target_domain=decision.target_domain.value if decision.target_domain else None,
            expected_conversion_rate=decision.expected_conversion_rate,
            expected_profit=decision.expected_profit,
            reasoning=decision.reasoning,
            agent_votes=agent_votes,
            confidence=(decision.profitability_score + decision.conversion_score) / 2
        )
        
        logger.info(f"Recommendation complete for user {user.user_id}: {decision.discount_percentage}%")
        return response
        
    except Exception as e:
        logger.error(f"Error processing recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@app.post("/recommendations/user")
async def get_user_recommendations(request: SingleUserOptionsRequest):
    if workflow_engine is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    user = None
    if user_index:
        user = user_index.get(request.user_id)
    if user is None and data_client:
        users = data_client.load_users()
        user = next((u for u in users if u.user_id == request.user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return workflow_engine.process_single_user_with_options(user)

@app.post("/recommendations/segment")
async def get_segment_recommendation(request: SegmentRecommendationRequest):
    if workflow_engine is None or data_client is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    return workflow_engine.process_segment_recommendation(
        segment=request.segment,
        domain=request.domain,
        user_sample_size=request.sample_size,
        data_client=data_client
    )

@app.post("/recommendations/batch")
async def recommend_batch(request: RecommendRequest):
    """
    Returns segment-grouped recommendations for a list of user_ids or all users if none specified.
    """
    if workflow_engine is None or data_client is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    users = data_client.load_users()
    if request.user_ids:
        users = [u for u in users if u.user_id in set(request.user_ids)]
        if not users:
            raise HTTPException(status_code=404, detail="No users found for provided ids")

    result = workflow_engine.process_segment_batch(users)
    return result

@app.post("/explain", response_model=ExplainResponse)
async def explain_decision(request: ExplainRequest):
    """
    Explain why a specific discount was recommended for a user
    
    Uses feature importance analysis and counterfactual explanations
    """
    if workflow_engine is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        user = request.user
        discount = request.discount
        
        # Get detailed processing
        detailed_result = workflow_engine.process_user_with_details(user)
        
        # Calculate feature importance based on user scores
        feature_importance = {
            "churn_score": abs(user.scores.churn_score - 0.5) * 2,  # Distance from neutral
            "activity_score": user.scores.activity_score,
            "cart_abandon_score": user.scores.cart_abandon_score,
            "price_sensitivity": user.scores.price_sensitivity,
            "family_score": abs(user.scores.family_score - 0.5) * 2,
            "segment": 0.3,  # Segment baseline importance
            "domain": 0.2,   # Domain baseline importance
            "is_oneway": 0.1 if user.is_oneway else 0.0,
            "user_basket": 0.1 if user.user_basket else 0.0
        }
        
        # Normalize feature importance
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            feature_importance = {k: v/total_importance for k, v in feature_importance.items()}
        
        # Build explanation
        explanation = {
            "state_encoding": detailed_result.get("state", {}).get("readable", {}),
            "business_rules_applied": len(detailed_result.get("business_rules", [])),
            "q_learning_confidence": detailed_result.get("q_learning", {}).get("q_value", 0),
            "agent_consensus": detailed_result.get("agent_proposals", {}),
            "primary_factors": [
                k for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            ]
        }
        
        response = ExplainResponse(
            user_id=user.user_id,
            discount_percentage=discount,
            explanation=explanation,
            feature_importance=feature_importance
        )
        
        logger.info(f"Explanation generated for user {user.user_id} with {discount}% discount")
        return response
        
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

@app.get("/stats")
async def get_system_stats():
    """Get system statistics and health metrics"""
    if workflow_engine is None:
        return {"error": "System not initialized"}
    
    try:
        # Segment-based stats
        rules_info = {
            "total_rules": len(workflow_engine.rules_engine.rules),
            "rule_names": [rule.__class__.__name__ for rule in workflow_engine.rules_engine.rules]
        }
        
        return {
            "system_status": "operational",
            "users_cached": len(user_index) if user_index else 0,
            "business_rules": rules_info,
            "agents": {
                "profitability": str(workflow_engine.profitability_agent),
                "conversion": str(workflow_engine.conversion_agent),
                "coordinator": str(workflow_engine.coordinator_agent)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        return {"error": f"Failed to get stats: {str(e)}"}

# Development mode check
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
