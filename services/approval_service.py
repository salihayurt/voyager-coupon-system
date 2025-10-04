from datetime import datetime
from typing import Optional, List
from services.sms_generator import SMSCampaign


class ApprovalService:
    def __init__(self):
        self.pending_campaigns: List[SMSCampaign] = []  # In production: use database

    def submit_for_approval(self, campaign: SMSCampaign) -> str:
        """Submit campaign to Product Manager dashboard"""
        campaign.approval_status = "pending"
        self.pending_campaigns.append(campaign)
        return campaign.campaign_id

    def get_pending_campaigns(self) -> List[SMSCampaign]:
        """Get all campaigns awaiting approval"""
        return [c for c in self.pending_campaigns if c.approval_status == "pending"]

    def approve_campaign(self, campaign_id: str, manager_id: str) -> bool:
        """Product Manager approves campaign"""
        campaign = self._find_campaign(campaign_id)
        if campaign:
            campaign.approval_status = "approved"
            campaign.approved_by = manager_id
            campaign.sent_at = datetime.now()
            self._send_sms(campaign)
            return True
        return False

    def reject_campaign(self, campaign_id: str, manager_id: str, reason: str) -> None:
        """Product Manager rejects campaign"""
        campaign = self._find_campaign(campaign_id)
        if campaign:
            campaign.approval_status = "rejected"
            campaign.metadata['rejection_reason'] = reason
            campaign.metadata['rejected_by'] = manager_id

    def _send_sms(self, campaign: SMSCampaign) -> None:
        """Send SMS to users (integrate with SMS gateway)"""
        print(f"Sending SMS to {len(campaign.user_ids)} users: {campaign.sms_text}")

    def _find_campaign(self, campaign_id: str) -> Optional[SMSCampaign]:
        return next((c for c in self.pending_campaigns if c.campaign_id == campaign_id), None)


