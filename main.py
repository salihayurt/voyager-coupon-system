#!/usr/bin/env python3
"""
Voyager Coupon System - Main Entry Point
Agent-centric coupon recommendation system with Q-Learning
"""

import asyncio
import uvicorn
from adapters.api.main import app
from config.app import settings


def main():
    """Main function to run the FastAPI application"""
    uvicorn.run(
        "adapters.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )


if __name__ == "__main__":
    main()
