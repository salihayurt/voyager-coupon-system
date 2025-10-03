from typing import List, Dict
from collections import Counter, defaultdict
from tqdm import tqdm

# Core components
from core.rules.business_rules import RulesEngine, RuleResult
from core.domain.user import User
from core.domain.coupon import CouponDecision
from core.domain.enums import ActionType, DomainType, SegmentType

# Agents
from agents.profitability_agent.agent import ProfitabilityAgent
from agents.conversion_agent.agent import ConversionAgent
from agents.coordinator_agent.agent import CoordinatorAgent
from agents.shared.context import SharedContext

class WorkflowEngine:
    """Orchestrates the entire agent pipeline for coupon decision making"""
    
    def __init__(self,
                 rules_engine: RulesEngine,
                 profitability_agent: ProfitabilityAgent,
                 conversion_agent: ConversionAgent,
                 coordinator_agent: CoordinatorAgent):
        """
        Initialize workflow engine with all components
        
        Args:
            rules_engine: Business rules engine
            profitability_agent: Profit optimization agent
            conversion_agent: Conversion optimization agent
            coordinator_agent: Strategic coordination agent
        """
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
        
        # Segment-based context
        context = SharedContext(
            user=None,
            business_rules=business_rules,
            segment_type=user.segment,
            user_count=None,
        )

        # Agents
        prof_proposal = self.profitability_agent.make_proposal(context)
        context.add_agent_proposal("ProfitabilityAgent", prof_proposal)
        conv_proposal = self.conversion_agent.make_proposal(context)
        context.add_agent_proposal("ConversionAgent", conv_proposal)

        # Coordination
        final_decision = self.coordinator_agent.merge_proposals(prof_proposal, conv_proposal, context)

        # Coupon decision
        return self._create_coupon_decision(user, final_decision, business_rules)
    
    def process_user_with_details(self, user: User) -> dict:
        """
        Process user and return detailed analysis including all intermediate results
        
        Returns:
            dict: Detailed processing results including state, q-learning, agent proposals, etc.
        """
        # Apply business rules
        business_rules = self.rules_engine.apply_rules(user)

        # Segment-based context
        context = SharedContext(
            user=None,
            business_rules=business_rules,
            segment_type=user.segment,
            user_count=None,
        )

        prof_proposal = self.profitability_agent.make_proposal(context)
        context.add_agent_proposal("profitability", prof_proposal)
        conv_proposal = self.conversion_agent.make_proposal(context)
        context.add_agent_proposal("conversion", conv_proposal)
        final_decision = self.coordinator_agent.merge_proposals(prof_proposal, conv_proposal, context)

        return {
            "user": user.dict(),
            "business_rules": [
                {
                    "applies": rule.applies,
                    "message": rule.message,
                    "action_type": rule.action_type.value if rule.action_type else None,
                    "discount_boost": rule.discount_boost,
                }
                for rule in business_rules
            ],
            "agent_proposals": {
                "profitability": prof_proposal,
                "conversion": conv_proposal,
                "coordination": final_decision,
            },
        }

    def process_single_user_with_options(self, user: User) -> dict:
        """Process one user and return multiple discount options."""
        rules = self.rules_engine.apply_rules(user)
        context = SharedContext(
            user=user,
            business_rules=rules,
            segment_type=user.segment,
            processing_mode="single_user"
        )
        prof = self.profitability_agent.make_proposal(context)
        context.profitability_proposal = prof
        conv = self.conversion_agent.make_proposal(context)
        context.conversion_proposal = conv
        options = self.coordinator_agent.generate_multiple_options(context)

        return {
            'user_id': user.user_id,
            'segment': user.segment.value,
            'domain': user.domain.value,
            'business_rules_applied': [r.message for r in rules if r.applies],
            'discount_options': options,
        }

    def process_segment_recommendation(self,
                                       segment: SegmentType,
                                       domain: DomainType | None = None,
                                       user_sample_size: int = 100,
                                       data_client=None) -> dict:
        """
        Generate segment-level recommendation (strategic decision).
        """
        if data_client is None:
            raise ValueError("data_client is required for segment processing")

        all_users = data_client.load_users()
        seg_users = [u for u in all_users if u.segment == segment]
        if domain:
            seg_users = [u for u in seg_users if u.domain == domain]

        import random
        sample = random.sample(seg_users, min(user_sample_size, len(seg_users)))

        discount_votes: list[int] = []
        for u in sample:
            res = self.process_single_user_with_options(u)
            balanced = [opt for opt in res['discount_options'] if opt['strategy'] == 'balanced']
            if balanced:
                discount_votes.append(balanced[0]['discount'])

        from collections import Counter
        most_common = Counter(discount_votes).most_common(1)[0] if discount_votes else (10, 1)
        recommended_discount = most_common[0]
        frequency = most_common[1] / max(1, len(discount_votes))

        # Acceptance via ConversionAgent's campaign client if available
        acceptance_rate = 0.0
        try:
            from adapters.external.campaign_data_client import CampaignDataClient
            client = CampaignDataClient()
            client.load_campaign_history("data/customer_data_with_campaigns_v3.csv")
            acceptance_rate = client.get_acceptance_rate_by_segment(segment, recommended_discount)
        except Exception:
            pass

        total_segment_size = len(seg_users)
        estimated_conversions = int(total_segment_size * acceptance_rate)
        avg_order_value = self._estimate_avg_order_value(segment)
        estimated_revenue = estimated_conversions * avg_order_value * (1 - recommended_discount/100)

        from core.domain.segment_constraints import get_allowed_discounts
        return {
            'segment': segment.value,
            'domain': domain.value if domain else 'ALL',
            'recommended_discount': recommended_discount,
            'recommendation_confidence': round(frequency, 3),
            'reasoning': f"{frequency:.0%} of sampled users optimal at {recommended_discount}%",
            'expected_impact': {
                'total_segment_users': total_segment_size,
                'expected_conversion_rate': round(acceptance_rate, 3),
                'estimated_conversions': estimated_conversions,
                'estimated_revenue': round(estimated_revenue, 2),
                'avg_order_value': avg_order_value
            },
            'allowed_discount_range': get_allowed_discounts(segment)
        }

    def _estimate_avg_order_value(self, segment: SegmentType) -> float:
        avg_values = {
            SegmentType.PREMIUM_CUSTOMERS: 5000,
            SegmentType.HIGH_VALUE_CUSTOMERS: 3500,
            SegmentType.AT_RISK_CUSTOMERS: 2000,
            SegmentType.PRICE_SENSITIVE_CUSTOMERS: 1500,
            SegmentType.STANDARD_CUSTOMERS: 2000
        }
        return avg_values.get(segment, 2000.0)
    
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

    def process_segment_batch(self, users: List[User]) -> Dict[str, dict]:
        """
        Process users grouped by segment and return segment-level recommendations and per-user results.
        """
        # Group users by segment
        segment_to_users: Dict = defaultdict(list)
        for user in users:
            segment_to_users[user.segment].append(user)

        recommendations: Dict[str, dict] = {}

        for segment, seg_users in segment_to_users.items():
            segment_decisions: List[dict] = []

            for user in seg_users:
                rules = self.rules_engine.apply_rules(user)

                # Create segment-focused context
                context = SharedContext(
                    user=None,  # segment mode does not require full user for agents
                    business_rules=rules,
                    segment_type=segment,
                    user_count=len(seg_users)
                )

                prof_prop = self.profitability_agent.make_proposal(context)
                context.profitability_proposal = prof_prop

                conv_prop = self.conversion_agent.make_proposal(context)
                context.conversion_proposal = conv_prop

                final = self.coordinator_agent.merge_proposals(prof_prop, conv_prop, context)

                segment_decisions.append({
                    'user_id': user.user_id,
                    'discount': final['discount'],
                    'reasoning': final.get('reasoning', [])
                })

            recommendations[segment.value] = {
                'user_count': len(seg_users),
                'coupon_options': self._aggregate_options(segment_decisions),
                'users': segment_decisions,
            }

        return recommendations

    def _aggregate_options(self, decisions: List[dict]) -> List[dict]:
        """Aggregate discount options with counts and percentages for a segment."""
        if not decisions:
            return []
        counts = Counter([d['discount'] for d in decisions])
        total = len(decisions)
        return [
            {
                'discount': disc,
                'user_count': cnt,
                'percentage': f"{(cnt/total)*100:.1f}%",
            }
            for disc, cnt in counts.most_common()
        ]
    
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
        """Get statistics about the pipeline components (segment-based)."""
        return {
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
        
        # Segment context and proposals
        context = SharedContext(
            user=None,
            business_rules=business_rules,
            segment_type=user.segment,
            user_count=None
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
            "user": user.dict(),
            "business_rules": [br.__dict__ for br in business_rules],
            "agent_proposals": {
                "profitability": prof_proposal,
                "conversion": conv_proposal,
                "coordination": final_decision
            },
            "final_decision": coupon_decision.dict(),
        }


