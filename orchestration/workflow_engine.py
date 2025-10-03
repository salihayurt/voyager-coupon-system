from typing import List
from tqdm import tqdm

# Learning components
from learning.q_table import QTable
from learning.state_encoder import StateEncoder

# Core components
from core.rules.business_rules import RulesEngine, RuleResult
from core.domain.user import User
from core.domain.coupon import CouponDecision
from core.domain.enums import ActionType, DomainType

# Agents
from agents.profitability_agent.agent import ProfitabilityAgent
from agents.conversion_agent.agent import ConversionAgent
from agents.coordinator_agent.agent import CoordinatorAgent
from agents.shared.context import SharedContext

class WorkflowEngine:
    """Orchestrates the entire agent pipeline for coupon decision making"""
    
    def __init__(self,
                 q_table: QTable,
                 state_encoder: StateEncoder,
                 rules_engine: RulesEngine,
                 profitability_agent: ProfitabilityAgent,
                 conversion_agent: ConversionAgent,
                 coordinator_agent: CoordinatorAgent):
        """
        Initialize workflow engine with all components
        
        Args:
            q_table: Q-Learning table for historical recommendations
            state_encoder: Converts users to Q-Learning states
            rules_engine: Business rules engine
            profitability_agent: Profit optimization agent
            conversion_agent: Conversion optimization agent
            coordinator_agent: Strategic coordination agent
        """
        self.q_table = q_table
        self.state_encoder = state_encoder
        self.rules_engine = rules_engine
        self.profitability_agent = profitability_agent
        self.conversion_agent = conversion_agent
        self.coordinator_agent = coordinator_agent
    
    def process_user(self, user: User) -> CouponDecision:
        """
        Main pipeline for processing a single user
        
        Args:
            user: User to process
            
        Returns:
            CouponDecision with optimized discount and reasoning
        """
        # Step 1: Apply business rules
        business_rules = self.rules_engine.apply_rules(user)
        
        # Step 2: Encode state and get Q-Learning suggestion
        state = self.state_encoder.encode(user)
        q_suggestion_action = self.q_table.get_best_action(state)
        q_suggestion = ActionSpace.get_discount_percentage(q_suggestion_action)
        q_value = self.q_table.get_q_value(state, q_suggestion_action)
        
        # Step 3: Create SharedContext
        context = SharedContext(
            user=user,
            business_rules=business_rules,
            q_learning_suggestion=q_suggestion,
            q_value=q_value
        )
        
        # Step 4: Run ProfitabilityAgent
        prof_proposal = self.profitability_agent.make_proposal(context)
        
        # Step 5: Add profitability proposal to context
        context.add_agent_proposal("ProfitabilityAgent", prof_proposal)
        
        # Step 6: Run ConversionAgent
        conv_proposal = self.conversion_agent.make_proposal(context)
        
        # Step 7: Add conversion proposal to context
        context.add_agent_proposal("ConversionAgent", conv_proposal)
        
        # Step 8: Run CoordinatorAgent to merge proposals
        final_decision = self.coordinator_agent.merge_proposals(
            prof_proposal, conv_proposal, context
        )
        
        # Step 9: Create CouponDecision object
        coupon_decision = self._create_coupon_decision(user, final_decision, business_rules)
        
        # Step 10: Return final decision
        return coupon_decision
    
    def process_batch(self, users: List[User]) -> List[CouponDecision]:
        """
        Process multiple users with progress tracking
        
        Args:
            users: List of users to process
            
        Returns:
            List of CouponDecision objects
        """
        decisions = []
        
        # Process with progress bar
        for user in tqdm(users, desc="Processing users", unit="user"):
            try:
                decision = self.process_user(user)
                decisions.append(decision)
            except Exception as e:
                print(f"Error processing user {user.user_id}: {e}")
                # Create fallback decision
                fallback_decision = self._create_fallback_decision(user, e)
                decisions.append(fallback_decision)
        
        return decisions
    
    def _create_coupon_decision(self, user: User, decision: dict, business_rules: List[RuleResult]) -> CouponDecision:
        """
        Convert agent decision dict to CouponDecision entity
        
        Args:
            user: Original user
            decision: Final coordinated decision from agents
            business_rules: Applied business rules
            
        Returns:
            CouponDecision object ready for system use
        """
        # Determine action type based on discount and rules
        action_type = self._determine_action_type(decision["discount"], business_rules)
        
        # Determine target domain
        target_domain = self._determine_target_domain(user, business_rules)
        
        # Build comprehensive reasoning
        reasoning = self._build_comprehensive_reasoning(decision, business_rules)
        
        # Extract agent scores from decision
        profitability_score = 0.8  # Default - could be extracted from context
        conversion_score = 0.8     # Default - could be extracted from context
        
        return CouponDecision(
            user_id=user.user_id,
            action_type=action_type,
            discount_percentage=decision["discount"],
            target_domain=target_domain,
            expected_conversion_rate=decision["expected_conversion"],
            expected_profit=decision["expected_profit"],
            reasoning=reasoning,
            profitability_score=profitability_score,
            conversion_score=conversion_score
        )
    
    def _determine_action_type(self, discount: int, business_rules: List[RuleResult]) -> ActionType:
        """Determine the action type based on discount and business rules"""
        
        # Check if business rules specify specific actions
        for rule in business_rules:
            if rule.applies and rule.action_type:
                # Business rules take precedence
                return rule.action_type
        
        # Default to discount offer for standard discounts
        return ActionType.OFFER_DISCOUNT
    
    def _determine_target_domain(self, user: User, business_rules: List[RuleResult]) -> DomainType:
        """Determine target domain for the coupon"""
        
        # Check business rules for specific domain targeting
        for rule in business_rules:
            if rule.applies and rule.target_domain:
                return rule.target_domain
        
        # Default to user's current domain
        return user.domain
    
    def _build_comprehensive_reasoning(self, decision: dict, business_rules: List[RuleResult]) -> List[str]:
        """Build comprehensive reasoning combining all sources"""
        reasoning = []
        
        # Add coordinated decision reasoning
        if decision.get("reasoning"):
            reasoning.extend(decision["reasoning"])
        
        # Add business rules reasoning
        for rule in business_rules:
            if rule.applies and rule.message:
                reasoning.append(f"Business rule: {rule.message}")
        
        # Add system-level reasoning
        reasoning.append(f"Multi-agent coordination: Profit vs Conversion optimization")
        reasoning.append(f"Q-Learning integration: Historical data-backed suggestions")
        
        return reasoning
    
    def _create_fallback_decision(self, user: User, error: Exception) -> CouponDecision:
        """Create fallback decision when processing fails"""
        return CouponDecision(
            user_id=user.user_id,
            action_type=ActionType.OFFER_DISCOUNT,
            discount_percentage=10,  # Safe default
            target_domain=user.domain,
            expected_conversion_rate=0.5,
            expected_profit=200.0,
            reasoning=[
                f"System error occurred: {str(error)}",
                "Fallback to safe 10% discount",
                "Manual review recommended"
            ],
            profitability_score=0.5,
            conversion_score=0.5
        )
    
    def get_pipeline_stats(self) -> dict:
        """Get statistics about the pipeline components"""
        return {
            "q_table_stats": self.q_table.get_statistics(),
            "state_space_size": self.state_encoder.state_space_size,
            "rules_count": len(self.rules_engine.rules),
            "agents": {
                "profitability": str(self.profitability_agent),
                "conversion": str(self.conversion_agent),
                "coordinator": str(self.coordinator_agent)
            }
        }
    
    def process_user_with_details(self, user: User) -> dict:
        """
        Process user and return detailed pipeline execution information
        
        Args:
            user: User to process
            
        Returns:
            Detailed execution information including intermediate steps
        """
        # Step 1: Business rules
        business_rules = self.rules_engine.apply_rules(user)
        
        # Step 2: Q-Learning
        state = self.state_encoder.encode(user)
        state_readable = self.state_encoder.decode_readable(state)
        q_suggestion_action = self.q_table.get_best_action(state)
        q_suggestion = ActionSpace.get_discount_percentage(q_suggestion_action)
        q_value = self.q_table.get_q_value(state, q_suggestion_action)
        
        # Step 3: Context creation
        context = SharedContext(
            user=user,
            business_rules=business_rules,
            q_learning_suggestion=q_suggestion,
            q_value=q_value
        )
        
        # Step 4 & 5: Profitability agent
        prof_proposal = self.profitability_agent.make_proposal(context)
        context.add_agent_proposal("ProfitabilityAgent", prof_proposal)
        
        # Step 6 & 7: Conversion agent
        conv_proposal = self.conversion_agent.make_proposal(context)
        context.add_agent_proposal("ConversionAgent", conv_proposal)
        
        # Step 8: Coordination
        final_decision = self.coordinator_agent.merge_proposals(
            prof_proposal, conv_proposal, context
        )
        
        # Step 9: Final coupon decision
        coupon_decision = self._create_coupon_decision(user, final_decision, business_rules)
        
        # Return detailed information
        return {
            "user": user,
            "state": {
                "encoded": state,
                "readable": state_readable,
                "visits": self.q_table.state_visits.get(state, 0)
            },
            "business_rules": business_rules,
            "q_learning": {
                "suggestion": q_suggestion,
                "q_value": q_value,
                "action": q_suggestion_action
            },
            "agent_proposals": {
                "profitability": prof_proposal,
                "conversion": conv_proposal,
                "coordination": final_decision
            },
            "final_decision": coupon_decision
        }

# Import ActionSpace for Q-Learning integration
from learning.action_space import ActionSpace
