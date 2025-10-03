from enum import Enum

class DomainType(str, Enum):
    """Enuygun platform domain'leri"""
    ENUYGUN_HOTEL = "ENUYGUN_HOTEL"
    ENUYGUN_FLIGHT = "ENUYGUN_FLIGHT"
    ENUYGUN_CAR_RENTAL = "ENUYGUN_CAR_RENTAL"
    ENUYGUN_BUS = "ENUYGUN_BUS"

class SegmentType(str, Enum):
    """Kullanıcı segmentleri (data analysis ekibinden gelecek)"""
    HIGH_VALUE = "high_value"
    AT_RISK = "at_risk"
    PRICE_SENSITIVE = "price_sensitive"
    FREQUENT_TRAVELER = "frequent_traveler"
    NEW_USER = "new_user"
    DORMANT = "dormant"

class ActionType(str, Enum):
    """Agent'ların alabileceği aksiyonlar"""
    OFFER_DISCOUNT = "offer_discount"
    SEND_LOGIN_NOTIFICATION = "send_login_notification"
    CROSS_SELL = "cross_sell"
    RETURN_TICKET_OFFER = "return_ticket_offer"
