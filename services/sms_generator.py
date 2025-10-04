from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from core.domain.enums import SegmentType, DomainType
from triggers.event_listeners import TriggerType, TriggerEvent
import os

# Optional LLM agent for SMS copywriting
try:
    from agents.shared.base_agent import BaseVoyagerAgent  # type: ignore
except Exception:
    BaseVoyagerAgent = None  # Fallback if not available


class SMSTemplate:
    """Simple SMS templates for each trigger type"""

    TEMPLATES = {
        TriggerType.CART_ABANDONMENT: [
            "🛒 You left items in your cart! Complete your booking now with {discount}% discount. Valid for 48 hours.",
            "Don't miss out! Your {product} is waiting. Get {discount}% off if you book today.",
        ],
        TriggerType.UNREGISTERED_USER: [
            "✈️ Join Enuygun to unlock exclusive deals! Register now and get {discount}% off your first booking.",
            "Create your account today and save {discount}% on {product}!",
        ],
        TriggerType.CROSS_SELL_OPPORTUNITY: [
            "🏨 You booked a flight! Complete your trip with {discount}% off on hotels.",
            "🚗 Need a car at your destination? Get {discount}% discount on car rentals!",
        ],
        TriggerType.PAST_TRIPS_REMINDER: [
            "✈️ Planning your annual trip to {destination}? Book now with {discount}% off!",
            "🌴 It's that time of year! {destination} awaits with {discount}% discount.",
        ],
        TriggerType.BUSINESS_RULE_COUPON: [
            "🎫 Complete your journey! Get {discount}% off on your return ticket.",
            "Special offer just for you: {discount}% discount on {product}!",
        ],
        TriggerType.SEGMENT_DISCOUNT_CAMPAIGN: [
            "🎉 Exclusive offer for valued customers: {discount}% off on all {product}!",
            "Limited time! {discount}% discount on {product} - book before it expires!",
        ],
    }

    @staticmethod
    def generate(trigger: TriggerEvent, discount: int, product: str = "bookings") -> str:
        """Generate SMS text. Uses LLM when ENABLE_LLM_SMS=true; falls back to templates."""
        # LLM path
        if os.getenv("ENABLE_LLM_SMS", "false").lower() in ("1", "true", "yes") and BaseVoyagerAgent is not None:
            try:
                agent = _SMSSimpleAgent()
                prompt = agent.build_prompt(trigger=trigger, discount=discount, product=product)
                if agent.agent is not None:
                    resp = agent.agent.run(prompt)
                    text = str(resp.content) if hasattr(resp, 'content') else str(resp)
                    return agent.postprocess(text)
            except Exception:
                pass

        # Template fallback
        import random
        templates = SMSTemplate.TEMPLATES.get(trigger.trigger_type, ["Special offer: {discount}% discount!"])
        template = random.choice(templates)
        destination = trigger.metadata.get('favorite_destination', 'your favorite destination')
        return template.format(discount=discount, product=product, destination=destination)


class _SMSSimpleAgent(BaseVoyagerAgent if BaseVoyagerAgent is not None else object):
    """Minimal SMS copywriter agent using existing LLM stack."""

    def __init__(self):
        if BaseVoyagerAgent is None:
            # No-op when LLM stack is not available
            return
        super().__init__(name="SMSSimpleAgent", temperature=0.4)

    def _setup_instructions(self) -> str:  # type: ignore[override]
        return (
            "You are a marketing copywriter.") + (
            " Write a single SMS suitable for carrier delivery:"
            " - 140-160 chars max"
            " - Clear CTA, include discount % and product"
            " - No links, no personally identifiable info"
            " - Use simple, friendly tone with 0-1 emoji max"
            " - Output only the SMS text, no extra commentary"
        )

    def build_prompt(self, trigger: TriggerEvent, discount: int, product: str) -> str:
        destination = trigger.metadata.get('favorite_destination', 'your favorite destination')
        context_lines = [
            f"Trigger: {trigger.trigger_type}",
            f"Discount: {discount}%",
            f"Product: {product}",
        ]
        if destination:
            context_lines.append(f"Destination: {destination}")
        return (
            "\n".join(context_lines)
            + "\n\nWrite the SMS now:"
        )

    def postprocess(self, text: str) -> str:
        # Keep it to a single line and trim length
        sms = text.strip().splitlines()[0]
        return sms[:200]


class SMSCampaign(BaseModel):
    campaign_id: str
    trigger_type: TriggerType
    segment: Optional[SegmentType] = None
    user_ids: list[int]
    discount_percentage: int
    sms_text: str
    created_at: datetime = datetime.now()
    approved_by: Optional[str] = None  # Product Manager ID
    approval_status: str = "pending"  # pending, approved, rejected
    sent_at: Optional[datetime] = None
    delivery_channel: str = "ui_notification"
    message_format: str = "sms"
    metadata: dict = {}


class SMSService:
    def create_campaign_from_trigger(self, trigger: TriggerEvent, discount: int) -> SMSCampaign:
        """Create SMS campaign from trigger event"""
        sms_text = SMSTemplate.generate(trigger, discount)
        return SMSCampaign(
            campaign_id=f"SMS_{trigger.trigger_type}_{datetime.now().timestamp()}",
            trigger_type=trigger.trigger_type,
            user_ids=[trigger.user_id],
            discount_percentage=discount,
            sms_text=sms_text,
            metadata=trigger.metadata,
        )

    def create_segment_campaign(self, segment: SegmentType, user_ids: list[int], discount: int, domain: DomainType) -> SMSCampaign:
        """Create campaign for entire segment (from /recommendations/segment)"""
        trigger = TriggerEvent(
            trigger_id=f"SEG_{segment}",
            trigger_type=TriggerType.SEGMENT_DISCOUNT_CAMPAIGN,
            user_id=0,
            timestamp=datetime.now(),
            metadata={'segment': segment.value, 'domain': domain.value},
        )
        sms_text = SMSTemplate.generate(trigger, discount, product=domain.value.lower())
        return SMSCampaign(
            campaign_id=f"SMS_SEGMENT_{segment}_{datetime.now().timestamp()}",
            trigger_type=TriggerType.SEGMENT_DISCOUNT_CAMPAIGN,
            segment=segment,
            user_ids=user_ids,
            discount_percentage=discount,
            sms_text=sms_text,
            metadata={'segment': segment.value, 'total_users': len(user_ids)},
        )


