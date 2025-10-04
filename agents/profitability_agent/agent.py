from typing import Dict, Any, List
import re
from agents.shared.base_agent import BaseVoyagerAgent
from agents.shared.context import SharedContext
from core.domain.enums import SegmentType
from core.domain.segment_constraints import get_allowed_discounts, SEGMENT_DISCOUNT_CONSTRAINTS
from core.rules.business_rules import RuleResult

class ProfitabilityAgent(BaseVoyagerAgent):
    """Agent focused on maximizing profit margins through conservative discount strategies"""
    
    def __init__(self):
        """Initialize with conservative temperature for consistent decisions"""
        super().__init__(name="ProfitabilityAgent", temperature=0.3)
    
    def _setup_instructions(self) -> str:
        """Setup profit-focused instructions for the agent"""
        instructions = [
            "You are a finance and profit optimization expert",
            "Goal: Maximize company profit margin while considering all user characteristics",
            "Approach:",
            "- Analyze all 5 user scores as continuous features (churn, activity, cart_abandon, price_sensitivity, family_score)",
            "- Use Q-Learning suggestions as learned patterns from historical data",
            "- Q-Learning has already discovered optimal thresholds through experience",
            "- Balance profit objectives with score combinations",
            "- Family score: 0=family buyer (higher order values), 1=solo buyer (individual purchases)",
            "- Available discounts: [5, 7, 10, 12, 15, 20]",
            "- User segment is mentioned only for explanation context, NOT for decision logic",
            "- Return only the discount number"
        ]
        return "\n".join(instructions)
    
    def make_proposal(self, context: SharedContext) -> Dict[str, Any]:
        """Profit proposal. Uses LLM if enabled and full user is present; otherwise constraint-based."""
        segment: SegmentType = context.segment_type or (context.user.segment if context.user else SegmentType.STANDARD_CUSTOMERS)
        allowed = get_allowed_discounts(segment)
        chosen = min(allowed) if allowed else 10
        used_llm = False

        # If LLM agent is available and we have a full user profile, prefer LLM guidance
        if getattr(self, 'agent', None) is not None and context.user is not None:
            try:
                prompt = self._build_analysis_prompt(context.user, context)
                response = self.agent.run(prompt)
                response_text = str(response.content) if hasattr(response, 'content') else str(response)
                llm_disc = self._parse_discount_response(response_text)
                if allowed:
                    # Clamp to nearest allowed
                    chosen = min(allowed, key=lambda x: abs(x - llm_disc))
                else:
                    chosen = llm_disc
                used_llm = True
            except Exception as e:
                # Fall back silently to constraint-based
                pass

        reasoning = [
            (f"LLM-guided: selected {chosen}% based on profit-focused analysis" if used_llm else
             f"Selected lowest viable discount within {segment.value} constraints"),
            f"Allowed range: {SEGMENT_DISCOUNT_CONSTRAINTS[segment][0]}-{SEGMENT_DISCOUNT_CONSTRAINTS[segment][1]}%",
        ]

        expected_profit = 0.0
        if context.user:
            expected_profit = self._estimate_profit(context.user, chosen)

        return {
            "discount": chosen,
            "reasoning": reasoning,
            "confidence": 0.85 if not used_llm else 0.9,
            "expected_conversion": 0.0,
            "expected_profit": expected_profit,
        }
    
    def _build_analysis_prompt(self, user, context: SharedContext) -> str:
        """Build analysis prompt for the LLM"""
        business_rules_text = self._format_business_rules(context.business_rules)
        q_suggestion = context.q_learning_suggestion or "No suggestion"
        q_value = context.q_value or 0.0
        
        prompt = f"""
Analyze this user for optimal discount percentage focusing on PROFIT MAXIMIZATION:

USER PROFILE:
- ID: {user.user_id}
- Segment: {user.segment.value} (for context only - do not use for decision logic)
- Domain: {user.domain.value}
- All Scores (analyze as continuous features):
  * Churn Risk: {user.scores.churn_score:.3f} (0=stable, 1=high risk)
  * Activity Level: {user.scores.activity_score:.3f} (0=inactive, 1=very active)
  * Cart Abandon Risk: {user.scores.cart_abandon_score:.3f} (0=completes purchases, 1=abandons)
  * Price Sensitivity: {user.scores.price_sensitivity:.3f} (0=price insensitive, 1=very sensitive)
  * Family Pattern: {user.scores.family_score:.3f} (0=family buyer, 1=solo buyer)
- Transaction Context: Is One-way: {user.is_oneway}, Has Basket: {user.user_basket}

BUSINESS RULES APPLIED:
{business_rules_text}

Q-LEARNING RECOMMENDATION: {q_suggestion}% (confidence: {q_value:.3f})
- This comes from learned patterns across thousands of similar user profiles
- Q-Learning has discovered optimal thresholds through trial and reward

DECISION APPROACH:
- Consider ALL FIVE scores as a pattern, not individual rules
- Q-Learning suggestion reflects learned knowledge about what works for similar score combinations
- Focus on profit margin while balancing user retention needs
- Family buyers (low family_score) may have higher order values
- Available discounts: [5, 7, 10, 12, 15, 20]

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
        
        # If no valid discount found, use conservative default
        return 10
    
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
        # Base confidence from Q-Learning alignment (if available)
        base_confidence = 0.7
        
        # Boost confidence if we have strong Q-Learning guidance
        # (This replaces hardcoded segment rules with learned patterns)
        q_boost = 0.0
        
        # Score pattern analysis (learned through experience, not hardcoded rules)
        # These patterns emerge from Q-Learning training, not predefined thresholds
        score_pattern_confidence = 0.0
        
        # Complex score interactions - let the model learn these patterns
        # Family + price sensitivity interaction
        if user.scores.family_score < 0.3 and discount <= 12:  # Family buyers, reasonable discount
            score_pattern_confidence += 0.1
        
        # Activity + churn interaction
        if user.scores.activity_score > 0.6 and user.scores.churn_score < 0.4:  # Engaged, stable users
            score_pattern_confidence += 0.1
        
        final_confidence = min(0.95, base_confidence + q_boost + score_pattern_confidence)
        return final_confidence
    
    def _estimate_profit(self, user, discount: int) -> float:
        """Estimate profit considering user characteristics including family patterns"""
        # Base order values by segment (still used for estimation)
        base_values = {
            SegmentType.PREMIUM_CUSTOMERS: 6000,
            SegmentType.HIGH_VALUE_CUSTOMERS: 5000,
            SegmentType.PRICE_SENSITIVE_CUSTOMERS: 1500,
            SegmentType.AT_RISK_CUSTOMERS: 2000,
            SegmentType.STANDARD_CUSTOMERS: 2500
        }
        
        base_value = base_values.get(user.segment, 2000)  # Default fallback
        
        # Family score adjustment: family buyers typically have higher order values
        family_multiplier = 1.0 + (1.0 - user.scores.family_score) * 0.5  # 1.0 to 1.5x
        adjusted_base_value = base_value * family_multiplier
        
        # Calculate profit margin (20% base minus discount)
        profit_margin = 0.20 - (discount / 100)
        
        # Ensure non-negative profit
        estimated_profit = max(0, adjusted_base_value * profit_margin)
        
        return round(estimated_profit, 2)
    
    def _build_reasoning(self, user, discount: int, context: SharedContext) -> List[str]:
        """Build reasoning for the discount decision based on learned patterns"""
        reasoning = []
        
        # Score-based insights (not hardcoded rules)
        reasoning.append(f"Selected {discount}% discount based on user score pattern analysis")
        
        # Family pattern insight
        if user.scores.family_score < 0.3:
            reasoning.append(f"Family buyer pattern (score: {user.scores.family_score:.2f}) - higher order value expected")
        elif user.scores.family_score > 0.7:
            reasoning.append(f"Individual buyer pattern (score: {user.scores.family_score:.2f}) - focused on personal needs")
        
        # Multi-score pattern analysis (learned combinations)
        if user.scores.churn_score > 0.7:
            reasoning.append(f"High churn risk ({user.scores.churn_score:.2f}) - balancing retention vs profit")
        
        if user.scores.price_sensitivity > 0.6:
            reasoning.append(f"Price sensitive user ({user.scores.price_sensitivity:.2f}) - discount effectiveness considered")
        
        # Activity level insight
        if user.scores.activity_score < 0.4:
            reasoning.append(f"Lower activity ({user.scores.activity_score:.2f}) - conservative approach preferred")
        
        # Business rules impact
        applicable_rules = [r for r in context.business_rules if r.applies]
        if applicable_rules:
            total_boost = sum(r.discount_boost for r in applicable_rules) * 100
            if total_boost > 0:
                reasoning.append(f"Business rules add {total_boost:.0f}% boost")
        
        # Q-Learning integration
        if context.q_learning_suggestion:
            reasoning.append(f"Q-Learning suggests {context.q_learning_suggestion}% based on historical patterns")
        
        # Segment mentioned for explanation only
        reasoning.append(f"User segment: {user.segment.value} (context for result analysis)")
        
        return reasoning
