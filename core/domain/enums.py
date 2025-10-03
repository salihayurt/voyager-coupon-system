from enum import Enum

class DomainType(str, Enum):
    """Enuygun platform domain'leri"""
    ENUYGUN_HOTEL = "ENUYGUN_HOTEL"
    ENUYGUN_FLIGHT = "ENUYGUN_FLIGHT"
    ENUYGUN_CAR_RENTAL = "ENUYGUN_CAR_RENTAL"
    ENUYGUN_BUS = "ENUYGUN_BUS"
    WINGIE_FLIGHT = "WINGIE_FLIGHT"

class SegmentType(str, Enum):
    """Kullanıcı segmentleri (data analysis ekibinden gelecek)"""
    PREMIUM_CUSTOMERS = "premium_customers"
    HIGH_VALUE_CUSTOMERS = "high_value_customers"
    PRICE_SENSITIVE_CUSTOMERS = "price_sensitive_customers"
    AT_RISK_CUSTOMERS = "at_risk_customers"
    STANDARD_CUSTOMERS = "standard_customers"

class ActionType(str, Enum):
    """Agent'ların alabileceği aksiyonlar"""
    OFFER_DISCOUNT = "offer_discount"
    SEND_LOGIN_NOTIFICATION = "send_login_notification"
    CROSS_SELL = "cross_sell"
    RETURN_TICKET_OFFER = "return_ticket_offer"
