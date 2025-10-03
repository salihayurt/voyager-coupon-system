from core.domain.enums import SegmentType

# Segment-based allowed discount ranges (inclusive)
SEGMENT_DISCOUNT_CONSTRAINTS = {
    SegmentType.AT_RISK_CUSTOMERS: (10, 13),
    SegmentType.HIGH_VALUE_CUSTOMERS: (5, 8),
    SegmentType.STANDARD_CUSTOMERS: (5, 10),
    SegmentType.PRICE_SENSITIVE_CUSTOMERS: (12, 15),
    SegmentType.PREMIUM_CUSTOMERS: (5, 8),
}

def get_allowed_discounts(segment: SegmentType) -> list[int]:
    """Return valid discount percentages for a segment based on constraints."""
    min_disc, max_disc = SEGMENT_DISCOUNT_CONSTRAINTS[segment]
    return [d for d in [5, 7, 10, 12, 15, 20] if min_disc <= d <= max_disc]


