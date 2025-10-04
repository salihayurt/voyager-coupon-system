from typing import Dict, Any, List
import re
from agents.shared.base_agent import BaseVoyagerAgent
from agents.shared.context import SharedContext
from core.domain.enums import SegmentType
from core.domain.segment_constraints import get_allowed_discounts, SEGMENT_DISCOUNT_CONSTRAINTS
from adapters.external.campaign_data_client import CampaignDataClient
from core.rules.business_rules import RuleResult

class ConversionAgent(BaseVoyagerAgent):
    """Agent focused on maximizing user conversion rates through strategic discount optimization"""
    
    def __init__(self):
        super().__init__(name="ConversionAgent", temperature=0.5)
        self.campaign_client = CampaignDataClient()
        # Load once; caller must ensure file exists
        try:
            self.campaign_client.load_campaign_history("data/customer_data_with_campaigns_v3.csv")
        except Exception:
            # Fallback: operate with zero acceptance data
            pass
    
    def _setup_instructions(self) -> str:
        """Setup conversion-focused instructions for the agent"""
        instructions = [
            "You are a conversion optimization and user psychology expert",
            "Goal: Maximize user conversion rate through optimal discount strategies",
            "Approach:",
            "- Analyze all 5 user scores as continuous features (churn, activity, cart_abandon, price_sensitivity, family_score)",
            "- Use Q-Learning suggestions as learned patterns from historical conversion data",
            "- Q-Learning has discovered optimal thresholds through conversion experience",
            "- Focus on conversion rate while considering profit impact",
            "- Family score: 0=family buyer (may book multiple items), 1=solo buyer (individual conversion)",
            "- Available discounts: [5, 7, 10, 12, 15, 20]",
            "- User segment is mentioned only for explanation context, NOT for decision logic",
            "- Balance conversion optimization with reasonable discount levels",
            "- Return only the discount number"
        ]
        return "\n".join(instructions)
    
    def make_proposal(self, context: SharedContext) -> Dict[str, Any]:
        """Conversion decision. Uses LLM if enabled and full user is present; otherwise data-driven."""
        segment: SegmentType = context.segment_type or (context.user.segment if context.user else SegmentType.STANDARD_CUSTOMERS)
        allowed = get_allowed_discounts(segment)
        if not allowed:
            chosen = 10
            acceptance = 0.0
        else:
            acceptance_map = {d: self.campaign_client.get_acceptance_rate_by_segment(segment, d) for d in allowed}
            # Proximity to profitability suggestion if available
            prof_disc = None
            if context.profitability_proposal and 'discount' in context.profitability_proposal:
                prof_disc = context.profitability_proposal['discount']
                candidates = [d for d in allowed if abs(d - prof_disc) <= 2] or allowed
            else:
                candidates = allowed
            # Default data-driven choice (tie -> higher discount)
            data_choice = max(candidates, key=lambda d: (acceptance_map.get(d, 0.0), d))
            acceptance = acceptance_map.get(data_choice, 0.0)

            # If LLM available and full user present, get LLM suggestion and reconcile
            if getattr(self, 'agent', None) is not None and context.user is not None:
                try:
                    prompt = self._build_analysis_prompt(context.user, context)
                    response = self.agent.run(prompt)
                    response_text = str(response.content) if hasattr(response, 'content') else str(response)
                    llm_disc = self._parse_discount_response(response_text)
                    # Clamp to nearest allowed
                    llm_choice = min(allowed, key=lambda x: abs(x - llm_disc))
                    # Prefer higher of (data_choice vs llm_choice) by predicted acceptance, tie -> higher
                    pair = [data_choice, llm_choice]
                    chosen = max(pair, key=lambda d: (acceptance_map.get(d, 0.0), d))
                    acceptance = acceptance_map.get(chosen, acceptance)
                except Exception:
                    chosen = data_choice
            else:
                chosen = data_choice

        reasoning = [
            f"Selected {chosen}% based on highest acceptance in {segment.value}",
            f"Allowed range: {SEGMENT_DISCOUNT_CONSTRAINTS[segment][0]}-{SEGMENT_DISCOUNT_CONSTRAINTS[segment][1]}%",
        ]

        return {
            "discount": chosen,
            "reasoning": reasoning,
            "confidence": 0.85,
            "expected_conversion": acceptance,
            "expected_profit": 0.0,
        }
    
    def _build_analysis_prompt(self, user, context: SharedContext) -> str:
        """Build conversion-focused analysis prompt for the LLM"""
        business_rules_text = self._format_business_rules(context.business_rules)
        q_suggestion = context.q_learning_suggestion or "No suggestion"
        q_value = context.q_value or 0.0
        
        prompt = f"""
Analyze this user for optimal CONVERSION-MAXIMIZING discount:

USER PROFILE:
- ID: {user.user_id}
- Segment: {user.segment.value} (for context only - do not use for decision logic)
- Domain: {user.domain.value}
- All Scores (analyze as continuous features for conversion patterns):
  * Churn Risk: {user.scores.churn_score:.3f} (0=stable, 1=likely to leave)
  * Activity Level: {user.scores.activity_score:.3f} (0=inactive, 1=highly engaged)
  * Cart Abandon Risk: {user.scores.cart_abandon_score:.3f} (0=completes, 1=abandons frequently)
  * Price Sensitivity: {user.scores.price_sensitivity:.3f} (0=price insensitive, 1=highly sensitive)
  * Family Pattern: {user.scores.family_score:.3f} (0=family buyer, 1=solo buyer)
- Transaction Context: Is One-way: {user.is_oneway}, Has Basket: {user.user_basket}

BUSINESS RULES APPLIED:
{business_rules_text}

Q-LEARNING RECOMMENDATION: {q_suggestion}% (confidence: {q_value:.3f})
- This reflects learned conversion patterns from historical data
- Q-Learning has discovered which score combinations respond best to different discounts
- These patterns were learned through actual conversion outcomes, not predefined rules

CONVERSION DECISION APPROACH:
- Consider ALL FIVE scores as a pattern that influences conversion probability
- Q-Learning suggestion reflects successful conversions from similar score profiles
- High churn/cart abandon scores may indicate need for stronger incentives
- Family buyers (low family_score) may have different conversion triggers than solo buyers
- Price sensitivity combined with other factors creates complex conversion patterns
- Available discounts: [5, 7, 10, 12, 15, 20]

Focus on CONVERSION RATE optimization based on learned score patterns.
Return only the recommended discount percentage as a number.
"""
        return prompt
    
    def _parse_discount_response(self, response: str) -> int:
        """Parse discount percentage from LLM response"""
        # Look for numbers in the response
        numbers = re.findall(r'\b(\d+)\b', response)
        
        valid_discounts = [5, 7, 10, 12, 15, 20]
        
        for num_str in numbers:
            num = int(num_str)
            if num in valid_discounts:
                return num
        
        # If no valid discount found, use conversion-focused default
        return 12
    
    def _format_business_rules(self, rules: List[RuleResult]) -> str:
        """Format business rules as readable bullet points"""
        if not rules:
            return "- No business rules applied"
        
        formatted = []
        for rule in rules:
            if rule.applies:
                boost_text = f" (+{rule.discount_boost*100:.0f}% boost)" if rule.discount_boost > 0 else ""
                formatted.append(f"- {rule.message}{boost_text}")
        
        return "\n".join(formatted) if formatted else "- No applicable business rules"
    
    def _calculate_confidence(self, user, discount: int) -> float:
        """Calculate confidence based on Q-Learning alignment and score patterns"""
        segment = user.segment
        churn_score = user.scores.churn_score
        
        # High confidence for optimal conversion scenarios based on learned patterns
        if segment == SegmentType.AT_RISK_CUSTOMERS and discount >= 15:
            return 0.95
        elif churn_score > 0.7 and discount >= 12:
            return 0.9
        elif segment == SegmentType.PRICE_SENSITIVE_CUSTOMERS and discount >= 10:
            return 0.85
        elif segment == SegmentType.STANDARD_CUSTOMERS and 10 <= discount <= 12:
            return 0.8
        else:
            return 0.7
    
    def _estimate_conversion(self, user, discount: int) -> float:
        """Estimate conversion rate based on user characteristics including family patterns"""
        # Base conversion rates by segment
        base_rates = {
            SegmentType.PREMIUM_CUSTOMERS: 0.8,
            SegmentType.HIGH_VALUE_CUSTOMERS: 0.7,
            SegmentType.PRICE_SENSITIVE_CUSTOMERS: 0.5,
            SegmentType.AT_RISK_CUSTOMERS: 0.4,
            SegmentType.STANDARD_CUSTOMERS: 0.6
        }
        
        base_conversion = base_rates.get(user.segment, 0.5)  # Default fallback
        
        # Discount boost: each 1% discount adds 2% conversion probability
        discount_boost = discount * 0.02
        
        # Churn penalty: high churn reduces base conversion
        churn_penalty = 0.1 if user.scores.churn_score > 0.7 else 0.0
        
        # Cart abandon boost: if low cart abandon score, slight boost
        cart_boost = 0.05 if user.scores.cart_abandon_score < 0.3 else 0.0
        
        # Family pattern adjustment: family buyers may have different conversion patterns
        # Family buyers (low family_score) might be more motivated when they find good deals
        family_boost = 0.0
        if user.scores.family_score < 0.4:  # Family buyer
            family_boost = 0.03  # Family buyers respond well to discounts
        
        # Price sensitivity interaction with discount
        price_sensitivity_boost = user.scores.price_sensitivity * discount * 0.001  # Sensitive users respond more to discounts
        
        # Calculate final conversion rate
        estimated_conversion = (base_conversion + discount_boost - churn_penalty + 
                               cart_boost + family_boost + price_sensitivity_boost)
        
        # Cap at 95% maximum conversion rate
        return min(0.95, max(0.1, estimated_conversion))  # Minimum 10%, maximum 95%
    
    def _estimate_profit(self, user, discount: int) -> float:
        """Estimate profit (secondary concern for conversion agent)"""
        # Same logic as ProfitabilityAgent but with different base values
        # Conversion agent assumes slightly lower base values due to higher discounts
        base_values = {
            SegmentType.PREMIUM_CUSTOMERS: 5500,
            SegmentType.HIGH_VALUE_CUSTOMERS: 4500,
            SegmentType.PRICE_SENSITIVE_CUSTOMERS: 1400,
            SegmentType.AT_RISK_CUSTOMERS: 1800,
            SegmentType.STANDARD_CUSTOMERS: 2300
        }
        
        base_value = base_values.get(user.segment, 1800)  # Default fallback
        
        # Calculate profit margin (20% base minus discount)
        profit_margin = 0.20 - (discount / 100)
        
        # Ensure non-negative profit
        estimated_profit = max(0, base_value * profit_margin)
        
        return round(estimated_profit, 2)
    
    def _build_reasoning(self, user, discount: int, context: SharedContext) -> List[str]:
        """Build reasoning for the conversion-focused discount decision"""
        reasoning = []
        
        # Segment-based reasoning
        if user.segment == SegmentType.AT_RISK_CUSTOMERS:
            reasoning.append(f"AT_RISK segment: Aggressive {discount}% discount to retain user")
        elif user.segment == SegmentType.PRICE_SENSITIVE_CUSTOMERS:
            reasoning.append(f"PRICE_SENSITIVE segment: {discount}% discount for strong conversion signal")
        elif user.segment == SegmentType.HIGH_VALUE_CUSTOMERS:
            reasoning.append(f"HIGH_VALUE segment: {discount}% discount for conversion optimization")
        else:
            reasoning.append(f"{user.segment.value} segment: {discount}% discount for conversion optimization")
        
        # Churn-based reasoning
        if user.scores.churn_score > 0.7:
            reasoning.append(f"High churn risk ({user.scores.churn_score:.2f}) - generous discount to retain")
        elif user.scores.churn_score < 0.3:
            reasoning.append(f"Low churn risk ({user.scores.churn_score:.2f}) - moderate discount sufficient")
        
        # Activity and engagement reasoning
        if user.scores.activity_score < 0.4:
            reasoning.append(f"Low activity score ({user.scores.activity_score:.2f}) - discount to re-engage")
        
        # Cart abandonment reasoning
        if user.scores.cart_abandon_score > 0.6:
            reasoning.append(f"High cart abandon risk ({user.scores.cart_abandon_score:.2f}) - discount to complete purchase")
        
        # Business rules impact
        applicable_rules = [r for r in context.business_rules if r.applies]
        if applicable_rules:
            total_boost = sum(r.discount_boost for r in applicable_rules) * 100
            if total_boost > 0:
                reasoning.append(f"Business rules add {total_boost:.0f}% boost for conversion")
        
        # Q-Learning reference
        if context.q_learning_suggestion:
            reasoning.append(f"Q-Learning suggests {context.q_learning_suggestion}% (considered)")
        
        # Conversion estimate
        conversion_rate = self._estimate_conversion(user, discount)
        reasoning.append(f"Expected conversion rate: {conversion_rate:.1%}")
        
        return reasoning
