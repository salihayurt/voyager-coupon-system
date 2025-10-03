import pandas as pd
from typing import Optional
from core.domain.enums import SegmentType


class CampaignDataClient:
    """Loads historical campaign acceptance and provides segment-level rates."""

    def __init__(self) -> None:
        self._df: Optional[pd.DataFrame] = None

    def load_campaign_history(self, csv_path: str) -> pd.DataFrame:
        """
        Load historical campaign acceptance data.
        Expected columns (semicolon separated):
        user_id;domain;is_oneway;user_basket;segment;churn_score;activity_score;
        cart_abandon_score;price_sensitivity;previous_domains;family_score;
        discount_campaign;campaign_accepted
        """
        df = pd.read_csv(csv_path, sep=';')
        # Normalize/ensure types we need
        df['segment'] = df['segment'].astype(str)
        df['discount_campaign'] = pd.to_numeric(df['discount_campaign'], errors='coerce')
        df['campaign_accepted'] = pd.to_numeric(df['campaign_accepted'], errors='coerce').fillna(0).astype(int)
        self._df = df
        return df

    def get_acceptance_rate_by_segment(self, segment: SegmentType, discount: int) -> float:
        """
        Calculate acceptance rate for a segment at given discount.
        Returns 0.0 if no matching data.
        """
        if self._df is None:
            return 0.0

        seg_key = segment.value
        df_seg = self._df[self._df['segment'] == seg_key]
        if df_seg.empty:
            return 0.0

        # Allow small tolerance around the discount bucket (±1) to avoid sparsity
        df_disc = df_seg[(df_seg['discount_campaign'] >= discount - 1) & (df_seg['discount_campaign'] <= discount + 1)]
        if df_disc.empty:
            return 0.0

        accepted = int(df_disc['campaign_accepted'].sum())
        total = int(len(df_disc))
        if total == 0:
            return 0.0
        return accepted / total


