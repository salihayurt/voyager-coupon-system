from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from services.sms_generator import SMSService
from services.approval_service import ApprovalService
from services.sms_generator import SMSCampaign
from triggers.event_listeners import EventListener, TriggerType, TriggerEvent
from core.domain.enums import SegmentType, DomainType
from core.domain.user import User
from orchestration.workflow_engine import WorkflowEngine
from adapters.external.data_analysis_client import DataAnalysisClient


router = APIRouter()
sms_service = SMSService()
approval_service = ApprovalService()
event_listener = EventListener()


class SingleUserOptionsRequest(BaseModel):
    user_id: int


class SegmentCampaignRequest(BaseModel):
    segment: SegmentType
    domain: Optional[DomainType] = None
    sample_size: int = 100


@router.post("/triggers/cart-abandonment")
async def trigger_cart_abandonment(user_id: int, cart_items: List[str], hours: int):
    trigger = event_listener.check_cart_abandonment(user_id, cart_items, hours)
    if not trigger:
        return {"message": "No trigger fired"}

    # Here we would fetch user object; for discount pick balanced option
    from app.main import workflow_engine, user_index, data_client
    user = user_index.get(user_id) if user_index else None
    if user is None and data_client:
        users = data_client.load_users()
        user = next((u for u in users if u.user_id == user_id), None)
    if not user:
        raise HTTPException(404, "User not found")

    result = workflow_engine.process_single_user_with_options(user)
    options = result.get('discount_options', [])
    discount = options[1]['discount'] if len(options) > 1 else (options[0]['discount'] if options else 10)

    campaign = sms_service.create_campaign_from_trigger(trigger, discount)
    campaign_id = approval_service.submit_for_approval(campaign)
    return {"campaign_id": campaign_id, "status": "pending_approval"}


@router.post("/triggers/segment-campaign")
async def trigger_segment_campaign(req: SegmentCampaignRequest):
    from app.main import workflow_engine, data_client
    if workflow_engine is None or data_client is None:
        raise HTTPException(503, "System not initialized")

    recommendation = workflow_engine.process_segment_recommendation(
        segment=req.segment, domain=req.domain, user_sample_size=req.sample_size, data_client=data_client
    )
    discount = recommendation['recommended_discount']

    all_users = data_client.load_users()
    segment_users = [u.user_id for u in all_users if u.segment == req.segment and (req.domain is None or u.domain == req.domain)]

    campaign = sms_service.create_segment_campaign(req.segment, segment_users, discount, req.domain or DomainType.ENUYGUN_FLIGHT)
    campaign_id = approval_service.submit_for_approval(campaign)
    return {
        "campaign_id": campaign_id,
        "status": "pending_approval",
        "total_users": len(segment_users),
        "sms_preview": campaign.sms_text,
    }


@router.get("/approvals/pending")
async def get_pending_approvals():
    campaigns = approval_service.get_pending_campaigns()
    return {
        "total_pending": len(campaigns),
        "campaigns": [
            {
                "campaign_id": c.campaign_id,
                "trigger_type": c.trigger_type,
                "segment": c.segment,
                "user_count": len(c.user_ids),
                "discount": c.discount_percentage,
                "sms_preview": c.sms_text,
                "created_at": c.created_at,
            }
            for c in campaigns
        ],
    }


@router.post("/approvals/{campaign_id}/approve")
async def approve_campaign(campaign_id: str, manager_id: str):
    success = approval_service.approve_campaign(campaign_id, manager_id)
    if success:
        return {"status": "approved", "message": "SMS campaign sent successfully"}
    raise HTTPException(404, "Campaign not found")


@router.post("/approvals/{campaign_id}/reject")
async def reject_campaign(campaign_id: str, manager_id: str, reason: str):
    approval_service.reject_campaign(campaign_id, manager_id, reason)
    return {"status": "rejected"}


@router.post("/recommendations/user/{user_id}/select-option")
async def select_user_option(user_id: int, selected_discount: int, manager_id: str):
    trigger = TriggerEvent(
        trigger_id=f"MANUAL_{user_id}",
        trigger_type=TriggerType.BUSINESS_RULE_COUPON,
        user_id=user_id,
        timestamp=datetime.now(),
        metadata={'selected_by_pm': manager_id},
    )
    campaign = sms_service.create_campaign_from_trigger(trigger, selected_discount)
    campaign.approved_by = manager_id
    campaign.approval_status = "approved"
    campaign.sent_at = datetime.now()
    approval_service._send_sms(campaign)
    return {"status": "sent", "campaign_id": campaign.campaign_id}


