import pandas as pd
import logging
from typing import Optional
from pathlib import Path
from core.domain.user import User, UserScores
from core.domain.enums import DomainType, SegmentType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAnalysisClient:
    def __init__(self, csv_path: str = "data/structured_customer_data.csv"):
        self.csv_path = csv_path
    
    def load_users(self, limit: Optional[int] = None) -> list[User]:
        """Load users from CSV"""
        logger.info(f"Loading users from {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        
        if limit:
            df = df.head(limit)
        
        users = []
        errors = 0
        
        for idx, row in df.iterrows():
            try:
                # Clamp family_score into [0, 1] to avoid validation drops
                raw_family = float(row['family_score'])
                family = 0.0 if raw_family < 0.0 else (1.0 if raw_family > 1.0 else raw_family)

                user = User(
                    user_id=int(row['user_id']),
                    domain=DomainType(row['domain']),
                    is_oneway=int(row['is_oneway']),
                    user_basket=int(row['user_basket']),
                    segment=SegmentType(row['segment'].lower()),
                    scores=UserScores(
                        churn_score=float(row['churn_score']),
                        activity_score=float(row['activity_score']),
                        cart_abandon_score=float(row['cart_abandon_score']),
                        price_sensitivity=float(row['price_sensitivity']),
                        family_score=family
                    ),
                    previous_domains=self._parse_previous_domains(row['previous_domains'])
                )
                users.append(user)
            except Exception as e:
                errors += 1
                if errors < 10:
                    logger.warning(f"Error parsing row {idx}: {e}")
        
        logger.info(f"Successfully loaded {len(users)} users from {self.csv_path}")
        return users
    
    def _parse_previous_domains(self, domains_str) -> Optional[list[DomainType]]:
        if pd.isna(domains_str) or not domains_str:
            return None
        try:
            domains = []
            for d in str(domains_str).split('|'):
                try:
                    domains.append(DomainType(d.strip()))
                except ValueError:
                    pass
            return domains if domains else None
        except:
            return None
    
    def validate_csv_format(self) -> dict:
        """Validate CSV format and required columns"""
        try:
            if not Path(self.csv_path).exists():
                return {"valid": False, "error": f"File not found: {self.csv_path}"}
            
            df = pd.read_csv(self.csv_path, nrows=10)
            
            required_columns = [
                'user_id', 'domain', 'is_oneway', 'user_basket', 'segment',
                'churn_score', 'activity_score', 'cart_abandon_score',
                'price_sensitivity', 'family_score', 'previous_domains'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return {"valid": False, "error": f"Missing columns: {missing_columns}"}
            
            return {"valid": True, "rows": len(df)}
            
        except Exception as e:
            return {"valid": False, "error": str(e)}