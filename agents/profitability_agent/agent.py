from typing import Dict, Any, List
import re
from agents.shared.base_agent import BaseVoyagerAgent
from agents.shared.context import SharedContext
from core.domain.enums import SegmentType
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
            "Goal: Maximize company profit margin",
            "Principles:",
            "- Suggest lowest possible discount",
            "- HIGH_VALUE segment → minimum discount (5-7%)",
            "- PRICE_SENSITIVE users → slightly higher but still conservative (10-12%)",
            "- If churn risk > 0.8 → be more aggressive",
            "- Discount options: [5, 7, 10, 12, 15, 20]",
            "- Return only the number"
        ]
        return "\n".join(instructions)
    
    def make_proposal(self, context: SharedContext) -> Dict[str, Any]:
        """
        Make profit-optimized discount proposal
        
        Args:
            context: Shared context with user data and other information
            
        Returns:
            Proposal dict with discount, reasoning, confidence, etc.
        """
        user = context.user
        
        # Build analysis prompt
        prompt = self._build_analysis_prompt(user, context)
        
        try:
            # Get LLM response
            response = self.agent.run(prompt)
            
            # Parse discount from response
            discount = self._parse_discount_response(response)
            
        except Exception as e:
            # Fallback to conservative default
            discount = 10
            print(f"ProfitabilityAgent error: {e}. Using fallback discount: {discount}")
        
        # Calculate confidence and estimates
        confidence = self._calculate_confidence(user, discount)
        expected_profit = self._estimate_profit(user, discount)
        
        # Build reasoning
        reasoning = self._build_reasoning(user, discount, context)
        
        return {
            "discount": discount,
            "reasoning": reasoning,
            "confidence": confidence,
            "expected_conversion": 0.0,  # Profitability agent doesn't predict conversions
            "expected_profit": expected_profit
        }
    
    def _build_analysis_prompt(self, user, context: SharedContext) -> str:
        """Build analysis prompt for the LLM"""
        business_rules_text = self._format_business_rules(context.business_rules)
        q_suggestion = context.q_learning_suggestion or "No suggestion"
        
        prompt = f"""
Analyze this user for optimal discount percentage:

USER PROFILE:
- ID: {user.user_id}
- Segment: {user.segment.value}
- Domain: {user.domain.value} 
- Churn Risk: {user.scores.churn_score:.2f}
- Price Sensitivity: {user.scores.price_sensitivity:.2f}
- Activity Score: {user.scores.activity_score:.2f}
- Is One-way: {user.is_oneway}
- Has Basket: {user.user_basket}

BUSINESS RULES APPLIED:
{business_rules_text}

Q-LEARNING SUGGESTION: {q_suggestion}%

DECISION CRITERIA:
- HIGH_VALUE segment: Use 5-7% discount to maximize profit
- PRICE_SENSITIVE: Use 10-12% discount (balance profit vs conversion)
- Churn risk > 0.8: Be more aggressive with discounts
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
        """Calculate confidence score based on user segment and discount alignment"""
        segment = user.segment
        
        # High confidence for optimal segment-discount combinations
        if segment == SegmentType.HIGH_VALUE and discount <= 7:
            return 0.9
        elif segment == SegmentType.PRICE_SENSITIVE and discount >= 10:
            return 0.85
        elif segment == SegmentType.AT_RISK and discount >= 12:
            return 0.8
        else:
            return 0.7
    
    def _estimate_profit(self, user, discount: int) -> float:
        """Estimate profit based on user segment and discount"""
        # Base order values by segment
        base_values = {
            SegmentType.HIGH_VALUE: 5000,
            SegmentType.FREQUENT_TRAVELER: 3000,
            SegmentType.PRICE_SENSITIVE: 1500,
            SegmentType.AT_RISK: 2000,
            SegmentType.NEW_USER: 1000,
            SegmentType.DORMANT: 800
        }
        
        base_value = base_values.get(user.segment, 2000)  # Default fallback
        
        # Calculate profit margin (20% base minus discount)
        profit_margin = 0.20 - (discount / 100)
        
        # Ensure non-negative profit
        estimated_profit = max(0, base_value * profit_margin)
        
        return round(estimated_profit, 2)
    
    def _build_reasoning(self, user, discount: int, context: SharedContext) -> List[str]:
        """Build reasoning for the discount decision"""
        reasoning = []
        
        # Segment-based reasoning
        if user.segment == SegmentType.HIGH_VALUE:
            reasoning.append(f"HIGH_VALUE segment: Conservative {discount}% discount to maximize profit")
        elif user.segment == SegmentType.PRICE_SENSITIVE:
            reasoning.append(f"PRICE_SENSITIVE segment: Balanced {discount}% discount")
        else:
            reasoning.append(f"{user.segment.value} segment: {discount}% discount selected")
        
        # Risk-based reasoning
        if user.scores.churn_score > 0.8:
            reasoning.append(f"High churn risk ({user.scores.churn_score:.2f}) - being more aggressive")
        elif user.scores.churn_score < 0.3:
            reasoning.append(f"Low churn risk ({user.scores.churn_score:.2f}) - conservative approach")
        
        # Price sensitivity
        if user.scores.price_sensitivity > 0.7:
            reasoning.append(f"High price sensitivity ({user.scores.price_sensitivity:.2f}) considered")
        
        # Business rules impact
        applicable_rules = [r for r in context.business_rules if r.applies]
        if applicable_rules:
            total_boost = sum(r.discount_boost for r in applicable_rules) * 100
            if total_boost > 0:
                reasoning.append(f"Business rules add {total_boost:.0f}% boost")
        
        # Q-Learning reference
        if context.q_learning_suggestion:
            reasoning.append(f"Q-Learning suggests {context.q_learning_suggestion}% (reference)")
        
        return reasoning
