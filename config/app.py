"""
Application configuration settings
"""

from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    API_HOST: str = "localhost"
    API_PORT: int = 8080
    DEBUG: bool = True
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    
    # Database (if needed)
    DATABASE_URL: str = "sqlite:///./voyager_coupon.db"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()
