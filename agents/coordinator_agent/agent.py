from typing import Dict, Any, List, Tuple, Optional
import re
from agents.shared.base_agent import BaseVoyagerAgent
from agents.shared.context import SharedContext
from core.rules.business_rules import RuleResult

class CoordinatorAgent(BaseVoyagerAgent):
    """Strategic decision coordinator that balances profit and conversion optimization"""
    
    def __init__(self, profit_weight: float = 0.6, conversion_weight: float = 0.4):
        """
        Initialize coordinator with configurable profit/conversion weights
        
        Args:
            profit_weight: Weight for profit optimization (default 60%)
            conversion_weight: Weight for conversion optimization (default 40%)
        """
        super().__init__(name="CoordinatorAgent", temperature=0.4)
        
        # Ensure weights sum to 1.0
        total_weight = profit_weight + conversion_weight
        self.profit_weight = profit_weight / total_weight
        self.conversion_weight = conversion_weight / total_weight
    
    def _setup_instructions(self) -> str:
        """Setup strategic coordination instructions for the agent"""
        instructions = [
            "You are a strategic decision coordinator",
            "Goal: Balance profit and conversion optimally",
            "You receive proposals from ProfitabilityAgent and ConversionAgent",
            "Consider:",
            f"- Q-Learning's learned suggestion (has historical data backing)",
            "- Business rules (mandatory constraints)",
            "- Both agents' confidence levels",
            f"- Company priority: {self.profit_weight*100:.0f}% profit, {self.conversion_weight*100:.0f}% conversion",
            "Return only the final discount number"
        ]
        return "\n".join(instructions)
    
    def make_proposal(self, context: SharedContext) -> Dict[str, Any]:
        """
        Make final coordinated discount decision
        
        Args:
            context: Shared context with both agent proposals
            
        Returns:
            Final proposal dict with balanced decision
        """
        # Extract agent proposals
        prof_proposal = context.profitability_proposal
        conv_proposal = context.conversion_proposal
        
        # Fallback to Q-Learning if proposals are missing
        if not prof_proposal or not conv_proposal:
            return self._fallback_to_qlearning(context)
        
        # Build coordination prompt
        prompt = self._build_coordination_prompt(prof_proposal, conv_proposal, context)
        
        try:
            # Get LLM coordination decision
            response = self.agent.run(prompt)
            
            # Parse final discount from response
            final_discount = self._parse_discount_response(response)
            
        except Exception as e:
            # Fallback to weighted average
            final_discount = self._calculate_weighted_average(
                prof_proposal.get("discount", 10),
                conv_proposal.get("discount", 12)
            )
            print(f"CoordinatorAgent error: {e}. Using weighted average: {final_discount}")
        
        # Calculate final metrics
        expected_conversion, expected_profit = self._calculate_final_metrics(final_discount, context)
        
        # Build comprehensive reasoning
        reasoning = self._build_reasoning(
            prof_proposal, conv_proposal, final_discount, 
            context.q_learning_suggestion, context.business_rules
        )
        
        # Calculate confidence based on agent agreement and Q-Learning alignment
        confidence = self._calculate_coordination_confidence(
            prof_proposal, conv_proposal, final_discount, context
        )
        
        return {
            "discount": final_discount,
            "reasoning": reasoning,
            "confidence": confidence,
            "expected_conversion": expected_conversion,
            "expected_profit": expected_profit
        }
    
    def merge_proposals(self, prof_proposal: Dict, conv_proposal: Dict, context: SharedContext) -> Dict:
        """
        Main coordination logic - merges proposals from both agents
        
        Args:
            prof_proposal: ProfitabilityAgent's proposal
            conv_proposal: ConversionAgent's proposal
            context: Shared context
            
        Returns:
            Final coordinated decision
        """
        # Update context with proposals
        context.profitability_proposal = prof_proposal
        context.conversion_proposal = conv_proposal
        
        # Make coordinated decision
        return self.make_proposal(context)
    
    def _build_coordination_prompt(self, prof_proposal: Dict, conv_proposal: Dict, context: SharedContext) -> str:
        """Build coordination prompt for the LLM"""
        user = context.user
        q_suggestion = context.q_learning_suggestion or "No suggestion"
        q_value = context.q_value or 0.0
        
        business_rules_text = self._format_business_rules(context.business_rules)
        
        prompt = f"""
Make final discount decision by coordinating these proposals:

USER PROFILE:
- ID: {user.user_id}
- Segment: {user.segment.value}
- Churn Risk: {user.scores.churn_score:.2f}
- Price Sensitivity: {user.scores.price_sensitivity:.2f}

Q-LEARNING RECOMMENDATION:
- Suggested Discount: {q_suggestion}%
- Confidence (Q-value): {q_value:.3f}
- Based on historical learning

PROFITABILITY AGENT PROPOSAL:
- Discount: {prof_proposal.get('discount')}%
- Confidence: {prof_proposal.get('confidence', 0):.2f}
- Expected Profit: {prof_proposal.get('expected_profit', 0):.0f} TL
- Key Reasoning: {prof_proposal.get('reasoning', ['N/A'])[0] if prof_proposal.get('reasoning') else 'N/A'}

CONVERSION AGENT PROPOSAL:
- Discount: {conv_proposal.get('discount')}%
- Confidence: {conv_proposal.get('confidence', 0):.2f}
- Expected Conversion: {conv_proposal.get('expected_conversion', 0):.1%}
- Key Reasoning: {conv_proposal.get('reasoning', ['N/A'])[0] if conv_proposal.get('reasoning') else 'N/A'}

BUSINESS RULES:
{business_rules_text}

DECISION WEIGHTS:
- Profit Priority: {self.profit_weight*100:.0f}%
- Conversion Priority: {self.conversion_weight*100:.0f}%

Consider all factors and make the optimal balanced decision.
Available discounts: [5, 7, 10, 12, 15, 20]
Return only the final discount percentage as a number.
"""
        return prompt
    
    def _parse_discount_response(self, response: str) -> int:
        """Parse discount percentage from LLM response"""
        numbers = re.findall(r'\b(\d+)\b', response)
        valid_discounts = [5, 7, 10, 12, 15, 20]
        
        for num_str in numbers:
            num = int(num_str)
            if num in valid_discounts:
                return num
        
        # Fallback to middle ground
        return 12
    
    def _calculate_weighted_average(self, prof_discount: int, conv_discount: int) -> int:
        """Calculate weighted average of two proposals and round to valid discount"""
        weighted_avg = (self.profit_weight * prof_discount + 
                       self.conversion_weight * conv_discount)
        
        # Round to nearest valid discount
        valid_discounts = [5, 7, 10, 12, 15, 20]
        return min(valid_discounts, key=lambda x: abs(x - weighted_avg))
    
    def _calculate_final_metrics(self, final_discount: int, context: SharedContext) -> Tuple[float, float]:
        """Calculate final expected conversion and profit based on chosen discount"""
        prof_proposal = context.profitability_proposal or {}
        conv_proposal = context.conversion_proposal or {}
        
        # Get base estimates from proposals
        prof_conversion = prof_proposal.get("expected_conversion", 0.5)
        prof_profit = prof_proposal.get("expected_profit", 0.0)
        
        conv_conversion = conv_proposal.get("expected_conversion", 0.5)
        conv_profit = conv_proposal.get("expected_profit", 0.0)
        
        # Average the estimates, weighted by agent specialization
        # Conversion agent is better at predicting conversions
        expected_conversion = (0.3 * prof_conversion + 0.7 * conv_conversion)
        
        # Profitability agent is better at predicting profits
        expected_profit = (0.7 * prof_profit + 0.3 * conv_profit)
        
        # Adjust based on discount difference from proposals
        prof_discount = prof_proposal.get("discount", final_discount)
        conv_discount = conv_proposal.get("discount", final_discount)
        
        # If final discount is higher than both proposals, boost conversion, reduce profit
        avg_proposed_discount = (prof_discount + conv_discount) / 2
        discount_delta = final_discount - avg_proposed_discount
        
        if discount_delta > 0:  # Higher discount than average
            expected_conversion = min(0.95, expected_conversion + discount_delta * 0.01)
            expected_profit = max(0, expected_profit * (1 - discount_delta * 0.02))
        elif discount_delta < 0:  # Lower discount than average
            expected_conversion = max(0.1, expected_conversion + discount_delta * 0.01)
            expected_profit = expected_profit * (1 - discount_delta * 0.02)
        
        return round(expected_conversion, 3), round(expected_profit, 2)
    
    def _build_reasoning(self, prof_proposal: Dict, conv_proposal: Dict, 
                        final_discount: int, q_suggestion: Optional[int], 
                        business_rules: List[RuleResult]) -> List[str]:
        """Build comprehensive reasoning for the final decision"""
        reasoning = []
        
        prof_discount = prof_proposal.get("discount", 0)
        conv_discount = conv_proposal.get("discount", 0)
        prof_confidence = prof_proposal.get("confidence", 0)
        conv_confidence = conv_proposal.get("confidence", 0)
        
        # Decision summary
        reasoning.append(f"Final decision: {final_discount}% discount (Profit: {prof_discount}%, Conversion: {conv_discount}%)")
        
        # Agent agreement analysis
        if abs(prof_discount - conv_discount) <= 2:
            reasoning.append(f"High agent alignment - both suggest similar discounts")
        elif final_discount == prof_discount:
            reasoning.append(f"Favoring profit optimization ({prof_confidence:.1%} confidence)")
        elif final_discount == conv_discount:
            reasoning.append(f"Favoring conversion optimization ({conv_confidence:.1%} confidence)")
        else:
            reasoning.append(f"Balanced compromise between {prof_discount}% (profit) and {conv_discount}% (conversion)")
        
        # Q-Learning influence
        if q_suggestion:
            if final_discount == q_suggestion:
                reasoning.append(f"Aligns with Q-Learning's historical learning ({q_suggestion}%)")
            else:
                reasoning.append(f"Differs from Q-Learning suggestion ({q_suggestion}%) based on current context")
        
        # Business rules impact
        applicable_rules = [r for r in business_rules if r.applies]
        if applicable_rules:
            total_boost = sum(r.discount_boost for r in applicable_rules) * 100
            if total_boost > 0:
                reasoning.append(f"Business rules add {total_boost:.0f}% boost requirement")
        
        # Weight explanation
        reasoning.append(f"Decision weights: {self.profit_weight*100:.0f}% profit, {self.conversion_weight*100:.0f}% conversion")
        
        # Key agent insights
        if prof_proposal.get("reasoning"):
            reasoning.append(f"Profit insight: {prof_proposal['reasoning'][0]}")
        if conv_proposal.get("reasoning"):
            reasoning.append(f"Conversion insight: {conv_proposal['reasoning'][0]}")
        
        return reasoning
    
    def _calculate_coordination_confidence(self, prof_proposal: Dict, conv_proposal: Dict, 
                                         final_discount: int, context: SharedContext) -> float:
        """Calculate confidence in the coordinated decision"""
        prof_discount = prof_proposal.get("discount", 0)
        conv_discount = conv_proposal.get("discount", 0)
        prof_confidence = prof_proposal.get("confidence", 0.5)
        conv_confidence = conv_proposal.get("confidence", 0.5)
        
        # Base confidence from weighted agent confidences
        base_confidence = (self.profit_weight * prof_confidence + 
                          self.conversion_weight * conv_confidence)
        
        # Boost confidence if agents agree
        agreement_bonus = 0.0
        if abs(prof_discount - conv_discount) <= 2:  # Close agreement
            agreement_bonus = 0.1
        elif abs(prof_discount - conv_discount) <= 5:  # Moderate agreement
            agreement_bonus = 0.05
        
        # Q-Learning alignment bonus
        q_bonus = 0.0
        if context.q_learning_suggestion and final_discount == context.q_learning_suggestion:
            q_bonus = 0.05
        
        # Final confidence calculation
        final_confidence = min(0.95, base_confidence + agreement_bonus + q_bonus)
        return round(final_confidence, 3)
    
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
    
    def _fallback_to_qlearning(self, context: SharedContext) -> Dict[str, Any]:
        """Fallback to Q-Learning suggestion when agent proposals are missing"""
        fallback_discount = context.q_learning_suggestion or 12
        
        return {
            "discount": fallback_discount,
            "reasoning": [
                "Missing agent proposals - using Q-Learning fallback",
                f"Q-Learning suggests {fallback_discount}% based on historical data"
            ],
            "confidence": 0.6,  # Lower confidence for fallback
            "expected_conversion": 0.5,  # Conservative estimates
            "expected_profit": 200.0
        }
