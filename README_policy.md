# Policy Tree Cohorting for Voyager Coupon System

This module provides explainable cohort-based policy recommendations using shallow decision trees trained on Voyager's recommendation outputs. It groups users into actionable cohorts with feasible discount strategies that respect segment constraints.

## Overview

The Policy Tree Cohorting system:

1. **Processes Voyager outputs** into structured features and feasible actions
2. **Trains shallow decision trees** to create explainable user cohorts  
3. **Generates business-friendly rules** with clear segment/domain/score conditions
4. **Provides REST API endpoints** for cohort management and recommendations
5. **Ensures constraint compliance** - all actions respect segment discount limits

## Architecture

```
voyager/
├── policy_tree/           # Core policy tree module
│   ├── constraints.py     # Segment constraints and allowed actions
│   ├── feasible.py       # Action selection logic
│   ├── tags.py           # XAI decision factors → tags conversion  
│   ├── features.py       # Feature engineering and binning
│   ├── train.py          # Policy tree training
│   ├── inference.py      # Cohort generation and summarization
│   └── __main__.py       # CLI training entry point
├── routers/
│   └── policy.py         # FastAPI endpoints
├── schemas/
│   └── policy.py         # Pydantic models
├── config/
│   └── policy.yaml       # Configuration
└── tests/
    └── test_policy_tree.py # Unit tests
```

## Quick Start

### 1. Training a Model with WEGATHON Dataset

```bash
# Train with real WEGATHON data (recommended)
python scripts/train_policy_wegathon.py --limit 1000

# Train with custom parameters
python scripts/train_policy_wegathon.py \
    --limit 2000 \
    --max-depth 4 \
    --min-leaf-frac 0.02 \
    --test-size 0.2
```

### 1b. Alternative: Generic Training

```bash
# Train from generic CSV data
python -m policy_tree.train --input data/user_data.csv --output artifacts/policy_tree.joblib
```

### 2. API Usage

Start the Voyager API server (policy endpoints are automatically included):

```bash
make run
# or
uvicorn app.main:app --reload --port 8080
```

### 3. Get Cohorts

```bash
# Get all cohorts
curl "http://localhost:8080/policy/cohorts"

# Filter cohorts
curl "http://localhost:8080/policy/cohorts?segment=at_risk_customers&min_size=100"

# Get cohort preview
curl -X POST "http://localhost:8080/policy/preview" \
  -H "Content-Type: application/json" \
  -d '{"cohort_name": "AtRisk_Flight_PriceHigh_12", "max_users": 20}'
```

## API Endpoints

### Core Endpoints

- **`GET /policy/cohorts`** - List cohorts with optional filtering
- **`POST /policy/preview`** - Preview users in a specific cohort  
- **`GET /policy/stats`** - System statistics and model info
- **`POST /policy/retrain`** - Trigger model retraining
- **`GET /policy/health`** - Health check

### Example Response

```json
{
  "cohorts": [
    {
      "name": "AtRisk_Flight_PriceHigh_12",
      "rule": "segment=AT_RISK_CUSTOMERS & domain=ENUYGUN_FLIGHT & high_price_sensitivity & high_churn",
      "action": "12",
      "size": 14382,
      "avg_expected_profit": 5.12,
      "avg_expected_conversion": 0.091,
      "mean_confidence": 0.73,
      "why_tags": ["high_price_sens", "high_churn"]
    }
  ],
  "total_cohorts": 8,
  "total_users_covered": 50000
}
```

## Configuration

Edit `config/policy.yaml` to customize:

```yaml
# Segment constraints
segments:
  at_risk_customers:
    allowed_discounts: [10, 11, 12, 13]
    premium_reward_eligible: false

# Tree parameters
tree_config:
  max_depth: 4
  min_leaf_fraction: 0.02
  min_support: 50

# Tag extraction rules
tag_rules:
  high_price_sens:
    score_threshold:
      field: "price_sensitivity"
      value: 0.6
    keywords: ["price sensitive", "pricing"]
```

## Data Requirements

### WEGATHON 2025 Dataset (Primary)

The system now works directly with the **WEGATHON 2025 Voyager dataset** (`WEGATHON_2025_VOYAGER_DATA_V2.csv`):

**Real Dataset Fields:**
- `user_id`: Unique user identifier  
- `transaction_type`: SALE or RESERVATION
- `domain`: ENUYGUN_HOTEL, ENUYGUN_FLIGHT, ENUYGUN_BUS, etc.
- `total_amount_masked`: Transaction amount
- `route_avg_price`: Average route price
- `adult_count`, `child_count`: Travel party size
- `is_oneway`, `is_domestic`: Travel characteristics
- `created_at`: Transaction timestamp

**Calculated Behavioral Scores:**
- `churn_score`: Based on transaction recency (0-1)
- `activity_score`: Based on transaction frequency (0-1) 
- `cart_abandon_score`: Reservation to sale ratio (0-1)
- `price_sensitivity`: Based on price vs average ratios (0-1)
- `family_score`: Based on travel party composition (0-1)

**Auto-Generated Segments:**
- `AT_RISK_CUSTOMERS`: High churn or cart abandon
- `HIGH_VALUE_CUSTOMERS`: High spending, multiple transactions
- `PREMIUM_CUSTOMERS`: Very high spending
- `PRICE_SENSITIVE_CUSTOMERS`: High price sensitivity
- `STANDARD_CUSTOMERS`: Default segment

### Alternative: Generic CSV Format

For custom datasets, include these fields:

```csv
user_id,segment,domain,churn_score,price_sensitivity,recommended_discount_pct,expected_profit
12345,at_risk_customers,ENUYGUN_FLIGHT,0.8,0.7,12,150.5
67890,high_value_customers,ENUYGUN_HOTEL,0.2,0.3,7,450.0
```

## Segment Constraints

The system enforces these business rules:

| Segment | Allowed Discounts | Premium Reward |
|---------|-------------------|----------------|
| AT_RISK_CUSTOMERS | 10-13% | ❌ |
| HIGH_VALUE_CUSTOMERS | 5-8% | ✅ |
| STANDARD_CUSTOMERS | 5-10% | ❌ |
| PRICE_SENSITIVE_CUSTOMERS | 12-15% | ❌ |
| PREMIUM_CUSTOMERS | 5-8% | ✅ |

**Premium Reward**: Available for HIGH_VALUE/PREMIUM customers with `price_sensitivity ≤ 0.4`

## Tag System

The system automatically extracts tags from user scores and decision factors:

- **`high_price_sens`**: price_sensitivity ≥ 0.6 or "price sensitive" in text
- **`high_churn`**: churn_score ≥ 0.7 or "churn" in text  
- **`cart_abandon`**: cart_abandon_score ≥ 0.7 or "abandon" in text
- **`loyal_high_value`**: HIGH_VALUE/PREMIUM segments
- **`family_pattern`**: family_score ≤ 0.4 or "family" in text
- **`time_urgent`**: "last minute", "deadline" in text

## Training Pipeline

1. **Load Data** → Load user data with scores and recommendations
2. **Extract Tags** → Convert decision factors to structured tags
3. **Choose Actions** → Select feasible actions per segment constraints
4. **Validate** → Ensure 0% illegal actions
5. **Build Features** → One-hot encode segments/domains, bin scores, create tag features
6. **Train Tree** → Fit shallow decision tree with sample weighting
7. **Generate Cohorts** → Group users by leaf nodes with business rules
8. **Save Model** → Persist artifacts for inference

## Testing

```bash
# Run all tests
pytest tests/test_policy_tree.py -v

# Test specific component
pytest tests/test_policy_tree.py::TestConstraints -v

# Test with coverage
pytest tests/test_policy_tree.py --cov=policy_tree
```

## Performance

- **Training**: Offline process, typically 30-60 seconds for 50k users
- **Inference**: `GET /policy/cohorts` responds in <300ms with caching
- **Cache**: Cohorts cached for 1 hour, auto-refreshed
- **Scalability**: Handles 100k+ users with 4-depth trees

## Monitoring

The system tracks:

- **Constraint violations**: Must be 0%
- **Model performance**: Training/test accuracy
- **Cohort quality**: Positive profit, minimum size
- **API latency**: <300ms target for cohort endpoints

## Troubleshooting

### Common Issues

**"No policy tree model found"**
```bash
# Train a model first
python -m policy_tree.train --input data/user_data.csv
```

**"Action validation failed"**  
- Check segment constraints in `config/policy.yaml`
- Verify input data has valid segments

**"No cohorts generated"**
- Increase tree depth or reduce min_support
- Check data quality and feature variance

### Debug Mode

Set logging level to DEBUG in `config/policy.yaml`:

```yaml
logging:
  level: "DEBUG"
  log_training: true
  log_api_calls: true
```

## Integration with Existing Voyager

The Policy Tree module is designed as a **non-breaking add-on**:

- ✅ **No changes** to existing Voyager endpoints
- ✅ **No modifications** to core recommendation logic  
- ✅ **Consumes outputs** from existing `/recommend` and `/explain` endpoints
- ✅ **Adds new endpoints** under `/policy/*` prefix
- ✅ **Independent operation** - can be disabled without affecting Voyager

### Migration Path

1. **Phase 1**: Deploy policy module alongside existing system
2. **Phase 2**: Train initial models using historical recommendation data  
3. **Phase 3**: Managers start using cohort insights for campaign planning
4. **Phase 4**: Optional integration with campaign execution systems

The system complements rather than replaces Voyager's individual recommendations, providing strategic cohort-level insights for business decision-making.
