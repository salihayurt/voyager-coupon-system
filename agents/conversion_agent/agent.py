from typing import Dict, Any, List
import re
from agents.shared.base_agent import BaseVoyagerAgent
from agents.shared.context import SharedContext
from core.domain.enums import SegmentType
from core.rules.business_rules import RuleResult

class ConversionAgent(BaseVoyagerAgent):
    """Agent focused on maximizing user conversion rates through strategic discount optimization"""
    
    def __init__(self):
        """Initialize with medium temperature for balanced conversion decisions"""
        super().__init__(name="ConversionAgent", temperature=0.5)
    
    def _setup_instructions(self) -> str:
        """Setup conversion-focused instructions for the agent"""
        instructions = [
            "You are a conversion optimization and user psychology expert",
            "Goal: Maximize user conversion rate",
            "Principles:",
            "- Higher discounts increase conversion probability",
            "- AT_RISK segment needs aggressive discounts (15-20%)",
            "- High churn score (>0.7) → be generous (12-20%)",
            "- PRICE_SENSITIVE users respond well to visible discounts (12-15%)",
            "- NEW_USER segment → moderate discount to hook them (10-12%)",
            "- Discount options: [5, 7, 10, 12, 15, 20]",
            "- Balance conversion with reasonable discount",
            "- Return only the number"
        ]
        return "\n".join(instructions)
    
    def make_proposal(self, context: SharedContext) -> Dict[str, Any]:
        """
        Make conversion-optimized discount proposal
        
        Args:
            context: Shared context with user data and other information
            
        Returns:
            Proposal dict with discount, reasoning, confidence, etc.
        """
        user = context.user
        
        # Build analysis prompt focused on conversion factors
        prompt = self._build_analysis_prompt(user, context)
        
        try:
            # Get LLM response
            response = self.agent.run(prompt)
            
            # Parse discount from response
            discount = self._parse_discount_response(response)
            
        except Exception as e:
            # Fallback to conversion-focused default
            discount = 12
            print(f"ConversionAgent error: {e}. Using fallback discount: {discount}")
        
        # Calculate confidence and estimates
        confidence = self._calculate_confidence(user, discount)
        expected_conversion = self._estimate_conversion(user, discount)
        expected_profit = self._estimate_profit(user, discount)
        
        # Build reasoning
        reasoning = self._build_reasoning(user, discount, context)
        
        return {
            "discount": discount,
            "reasoning": reasoning,
            "confidence": confidence,
            "expected_conversion": expected_conversion,
            "expected_profit": expected_profit
        }
    
    def _build_analysis_prompt(self, user, context: SharedContext) -> str:
        """Build conversion-focused analysis prompt for the LLM"""
        business_rules_text = self._format_business_rules(context.business_rules)
        q_suggestion = context.q_learning_suggestion or "No suggestion"
        
        prompt = f"""
Analyze this user for optimal conversion-maximizing discount:

USER PROFILE (Conversion Focus):
- ID: {user.user_id}
- Segment: {user.segment.value}
- Domain: {user.domain.value}
- Churn Risk: {user.scores.churn_score:.2f} (HIGH PRIORITY)
- Activity Score: {user.scores.activity_score:.2f}
- Cart Abandon Score: {user.scores.cart_abandon_score:.2f}
- Price Sensitivity: {user.scores.price_sensitivity:.2f}
- Is One-way: {user.is_oneway}
- Has Basket: {user.user_basket}

BUSINESS RULES APPLIED:
{business_rules_text}

Q-LEARNING SUGGESTION: {q_suggestion}%

CONVERSION DECISION CRITERIA:
- AT_RISK segment: Use aggressive discounts (15-20%) to retain
- High churn risk (>0.7): Be generous with discounts (12-20%)
- PRICE_SENSITIVE: Visible discounts work well (12-15%)
- NEW_USER: Hook them with moderate discount (10-12%)
- High cart abandon score: Needs incentive to complete purchase
- Available discounts: [5, 7, 10, 12, 15, 20]

Focus on CONVERSION RATE over profit margins.
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
        """Calculate confidence score based on conversion optimization alignment"""
        segment = user.segment
        churn_score = user.scores.churn_score
        
        # High confidence for optimal conversion scenarios
        if segment == SegmentType.AT_RISK and discount >= 15:
            return 0.95
        elif churn_score > 0.7 and discount >= 12:
            return 0.9
        elif segment == SegmentType.PRICE_SENSITIVE and discount >= 10:
            return 0.85
        elif segment == SegmentType.NEW_USER and 10 <= discount <= 12:
            return 0.8
        else:
            return 0.7
    
    def _estimate_conversion(self, user, discount: int) -> float:
        """Estimate conversion rate based on user segment and discount"""
        # Base conversion rates by segment
        base_rates = {
            SegmentType.HIGH_VALUE: 0.7,
            SegmentType.FREQUENT_TRAVELER: 0.65,
            SegmentType.PRICE_SENSITIVE: 0.5,
            SegmentType.AT_RISK: 0.4,
            SegmentType.NEW_USER: 0.45,
            SegmentType.DORMANT: 0.3
        }
        
        base_conversion = base_rates.get(user.segment, 0.5)  # Default fallback
        
        # Discount boost: each 1% discount adds 2% conversion probability
        discount_boost = discount * 0.02
        
        # Churn penalty: high churn reduces base conversion
        churn_penalty = 0.1 if user.scores.churn_score > 0.7 else 0.0
        
        # Cart abandon boost: if low cart abandon score, slight boost
        cart_boost = 0.05 if user.scores.cart_abandon_score < 0.3 else 0.0
        
        # Calculate final conversion rate
        estimated_conversion = base_conversion + discount_boost - churn_penalty + cart_boost
        
        # Cap at 95% maximum conversion rate
        return min(0.95, max(0.1, estimated_conversion))  # Minimum 10%, maximum 95%
    
    def _estimate_profit(self, user, discount: int) -> float:
        """Estimate profit (secondary concern for conversion agent)"""
        # Same logic as ProfitabilityAgent but with different base values
        # Conversion agent assumes slightly lower base values due to higher discounts
        base_values = {
            SegmentType.HIGH_VALUE: 4500,  # Slightly lower than profit-focused
            SegmentType.FREQUENT_TRAVELER: 2800,
            SegmentType.PRICE_SENSITIVE: 1400,
            SegmentType.AT_RISK: 1800,
            SegmentType.NEW_USER: 950,
            SegmentType.DORMANT: 750
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
        if user.segment == SegmentType.AT_RISK:
            reasoning.append(f"AT_RISK segment: Aggressive {discount}% discount to retain user")
        elif user.segment == SegmentType.PRICE_SENSITIVE:
            reasoning.append(f"PRICE_SENSITIVE segment: {discount}% discount for strong conversion signal")
        elif user.segment == SegmentType.NEW_USER:
            reasoning.append(f"NEW_USER segment: {discount}% discount to create first purchase hook")
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
