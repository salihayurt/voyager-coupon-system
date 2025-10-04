# Local Testing Guide for Voyager Policy Tree System

## 🚀 Quick Start (3 Steps)

### Step 1: Train the Model
```powershell
# Train with WEGATHON dataset (takes ~30 seconds)
python scripts/train_policy_wegathon.py --limit 500
```

### Step 2: Start the Server
```powershell
# Start the API server
python start_server.py
```

### Step 3: Test the API
Open a **new terminal** and run:
```powershell
# Run automated tests
python test_local.py
```

## 📋 Manual Testing

### Option A: Browser Testing
Open your browser and visit:
- http://localhost:8080/docs - Interactive API documentation
- http://localhost:8080/health - Health check
- http://localhost:8080/policy/cohorts - View cohorts (JSON)
- http://localhost:8080/policy/stats - System statistics

### Option B: PowerShell Testing
```powershell
# Test health endpoint
Invoke-RestMethod -Uri "http://localhost:8080/health" -Method GET

# Test policy cohorts
Invoke-RestMethod -Uri "http://localhost:8080/policy/cohorts" -Method GET

# Test with filters
Invoke-RestMethod -Uri "http://localhost:8080/policy/cohorts?min_size=20" -Method GET

# Test policy stats
Invoke-RestMethod -Uri "http://localhost:8080/policy/stats" -Method GET
```

### Option C: curl (if available)
```bash
# Test endpoints with curl
curl "http://localhost:8080/health"
curl "http://localhost:8080/policy/cohorts"
curl "http://localhost:8080/policy/stats"
```

## 🔧 Troubleshooting

### Issue: "ImportError: cannot import name..."
**Solution**: Run this to fix missing imports:
```powershell
python -c "from policy_tree import *; print('✅ Imports fixed')"
```

### Issue: "No policy tree model found"
**Solution**: Train a model first:
```powershell
python scripts/train_policy_wegathon.py --limit 500
```

### Issue: "Data client not available"
**Solution**: Ensure WEGATHON dataset is in the right location:
```
data/WEGATHON_2025_VOYAGER_DATA_V2.csv
```

### Issue: Server won't start
**Solution**: Check dependencies:
```powershell
pip install -r requirements.txt
```

## 📊 Expected Outputs

### Health Check Response
```json
{
  "status": "healthy",
  "system": "Voyager Coupon System", 
  "version": "1.0.0",
  "components": {
    "workflow_engine": true,
    "data_client": true
  }
}
```

### Policy Cohorts Response
```json
{
  "cohorts": [
    {
      "name": "AtRisk_Hotel_PriceHigh_13",
      "rule": "segment=AT_RISK_CUSTOMERS & domain=ENUYGUN_HOTEL & high_price_sensitivity",
      "action": "13",
      "size": 440,
      "avg_expected_profit": 180.50,
      "avg_expected_conversion": 0.078,
      "mean_confidence": 0.75,
      "why_tags": ["high_price_sens", "high_churn"]
    }
  ],
  "total_cohorts": 8,
  "total_users_covered": 500
}
```

### Policy Stats Response  
```json
{
  "model_info": {
    "version": "20241004_025614",
    "training_date": "2024-10-04T02:56:14",
    "n_samples": 400,
    "tree_depth": 4,
    "training_accuracy": 0.887
  },
  "cohort_stats": {
    "total_cohorts": 8,
    "avg_cohort_size": 62.5,
    "total_users_covered": 500
  }
}
```

## 🎯 Testing Scenarios

### Scenario 1: Basic Functionality
1. Start server: `python start_server.py`
2. Test health: Visit http://localhost:8080/health
3. View cohorts: Visit http://localhost:8080/policy/cohorts
4. Check stats: Visit http://localhost:8080/policy/stats

### Scenario 2: Filtered Cohorts
```powershell
# Get cohorts with minimum size filter
Invoke-RestMethod -Uri "http://localhost:8080/policy/cohorts?min_size=50" -Method GET

# Get cohorts for specific segment
Invoke-RestMethod -Uri "http://localhost:8080/policy/cohorts?segment=at_risk_customers" -Method GET
```

### Scenario 3: Cohort Preview
```powershell
# Preview users in a cohort (POST request)
$body = @{
    cohort_name = "AtRisk_Hotel_PriceHigh_13"
    max_users = 10
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8080/policy/preview" -Method POST -Body $body -ContentType "application/json"
```

## 📈 Performance Expectations

- **Server startup**: ~5-10 seconds
- **Cohort generation**: <300ms (cached)
- **Model loading**: ~1 second
- **User data loading**: ~2-5 seconds (first time)

## 🎉 Success Indicators

✅ **Server starts without errors**  
✅ **Health endpoint returns 200 OK**  
✅ **Policy cohorts return valid JSON**  
✅ **Stats show model information**  
✅ **All constraint violations = 0%**  
✅ **Cohorts have positive expected profit**

## 🔄 Development Workflow

1. **Make changes** to policy tree code
2. **Restart server**: Ctrl+C, then `python start_server.py`
3. **Test endpoints**: `python test_local.py`
4. **Check browser**: Visit http://localhost:8080/docs
5. **Iterate** as needed

## 📞 Quick Commands Reference

```powershell
# Setup (one time)
pip install -r requirements.txt
python scripts/train_policy_wegathon.py --limit 500

# Daily development
python start_server.py          # Start server
python test_local.py           # Test API
python scripts/wegathon_integration_summary.py  # Check status

# Browser testing
# http://localhost:8080/docs    # API documentation
# http://localhost:8080/policy/cohorts  # View cohorts
```

Happy testing! 🚀
