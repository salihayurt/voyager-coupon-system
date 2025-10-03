.PHONY: install run train verify clean test lint format help split

# Default Python interpreter
PYTHON := python3

# Project directories
DATA_DIR := data
MODELS_DIR := $(DATA_DIR)
SCRIPTS_DIR := scripts

# Default data file
CSV_FILE := $(DATA_DIR)/structured_customer_data.csv

help: ## Show this help message
	@echo "Voyager Coupon System - Available Commands:"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Examples:"
	@echo "  make install                    # Install dependencies"
	@echo "  make split                      # Split data into train/val/test"
	@echo "  make train                      # Train Q-Learning model"
	@echo "  make run                        # Start FastAPI server"
	@echo "  make verify                     # Verify system setup"

install: ## Install Python dependencies
	@echo "📦 Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✅ Dependencies installed successfully"

install-dev: install ## Install development dependencies
	@echo "🛠️  Installing development dependencies..."
	pip install -e ".[dev]"
	@echo "✅ Development dependencies installed"

clean: ## Clean up generated files
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	@echo "✅ Cleanup complete"

split: ## Split data into train/validation/test sets
	@echo "✂️  Splitting data..."
	@if [ ! -f "$(CSV_FILE)" ]; then \
		echo "❌ CSV file not found: $(CSV_FILE)"; \
		echo "Please ensure your data file exists at $(CSV_FILE)"; \
		exit 1; \
	fi
	$(PYTHON) $(SCRIPTS_DIR)/make_splits.py --csv $(CSV_FILE) --output-dir $(DATA_DIR) --validate
	@echo "✅ Data splitting complete"

train: ## Train Q-Learning model
	@echo "🧠 Training Q-Learning model..."
	@if [ ! -f "$(CSV_FILE)" ]; then \
		echo "❌ CSV file not found: $(CSV_FILE)"; \
		echo "Please run 'make split' first or ensure your data file exists"; \
		exit 1; \
	fi
	$(PYTHON) $(SCRIPTS_DIR)/train_qlearning.py --csv $(CSV_FILE) --epochs 30 --batch-size 1000
	@echo "✅ Training complete"

train-quick: ## Quick training with fewer epochs (for testing)
	@echo "⚡ Quick training..."
	$(PYTHON) $(SCRIPTS_DIR)/train_qlearning.py --csv $(CSV_FILE) --epochs 5 --batch-size 100
	@echo "✅ Quick training complete"

run: ## Start FastAPI server
	@echo "🚀 Starting Voyager Coupon System API..."
	uvicorn app.main:app --reload --port 8080 --host 0.0.0.0
	
run-prod: ## Start production server
	@echo "🏭 Starting production server..."
	uvicorn app.main:app --port 8080 --host 0.0.0.0 --workers 4

verify: ## Verify system setup and run tests
	@echo "🔍 Verifying system setup..."
	$(PYTHON) $(SCRIPTS_DIR)/verify_setup.py
	@echo "✅ Verification complete"

test-pipeline: ## Test the complete pipeline with sample data
	@echo "🧪 Testing pipeline..."
	$(PYTHON) $(SCRIPTS_DIR)/test_pipeline.py 10
	@echo "✅ Pipeline test complete"

lint: ## Run code linting
	@echo "🔍 Running linter..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	@echo "✅ Linting complete"

format: ## Format code with black
	@echo "🎨 Formatting code..."
	black --line-length 100 .
	@echo "✅ Code formatting complete"

type-check: ## Run type checking
	@echo "🔍 Running type checker..."
	mypy . --ignore-missing-imports
	@echo "✅ Type checking complete"

test: ## Run unit tests
	@echo "🧪 Running tests..."
	pytest tests/ -v
	@echo "✅ Tests complete"

# API testing commands
test-api: ## Test API endpoints (requires server to be running)
	@echo "🌐 Testing API endpoints..."
	@echo "Testing health endpoint..."
	curl -X GET http://localhost:8080/health || echo "❌ Health endpoint failed"
	@echo "\nTesting recommend endpoint..."
	curl -X POST http://localhost:8080/recommend \
		-H "Content-Type: application/json" \
		-d '{"user_id": 121}' || echo "❌ Recommend endpoint failed"
	@echo "✅ API tests complete"

# Development workflow
dev-setup: install-dev split ## Complete development setup
	@echo "🛠️  Development environment ready!"
	@echo "Next steps:"
	@echo "  1. Run 'make train' to train your model"
	@echo "  2. Run 'make run' to start the server"
	@echo "  3. Run 'make test-api' to test endpoints"

# Production workflow  
prod-setup: install split train ## Complete production setup
	@echo "🏭 Production environment ready!"
	@echo "Run 'make run-prod' to start the production server"

# Data validation
validate-data: ## Validate CSV data format
	@echo "📊 Validating data format..."
	$(PYTHON) -c "from adapters.external.data_analysis_client import DataAnalysisClient; \
		client = DataAnalysisClient('$(CSV_FILE)'); \
		result = client.validate_csv_format(); \
		print('✅ Data valid' if result.get('valid') else '❌ Data invalid: ' + str(result))"

# System information
info: ## Show system information
	@echo "📋 Voyager Coupon System Information:"
	@echo "  Python version: $(shell $(PYTHON) --version)"
	@echo "  Project directory: $(shell pwd)"
	@echo "  Data directory: $(DATA_DIR)"
	@echo "  CSV file: $(CSV_FILE)"
	@echo "  CSV exists: $(shell [ -f '$(CSV_FILE)' ] && echo 'Yes' || echo 'No')"
	@echo "  Model file: $(MODELS_DIR)/q_table.pkl"
	@echo "  Model exists: $(shell [ -f '$(MODELS_DIR)/q_table.pkl' ] && echo 'Yes' || echo 'No')"

# Docker commands (if needed)
docker-build: ## Build Docker image
	@echo "🐳 Building Docker image..."
	docker build -t voyager-coupon-system .
	@echo "✅ Docker image built"

docker-run: docker-build ## Run Docker container
	@echo "🐳 Running Docker container..."
	docker run -p 8080:8080 -v $(shell pwd)/$(DATA_DIR):/app/$(DATA_DIR) voyager-coupon-system
	
# Backup and restore
backup-model: ## Backup trained model
	@echo "💾 Backing up model..."
	@if [ -f "$(MODELS_DIR)/q_table.pkl" ]; then \
		cp $(MODELS_DIR)/q_table.pkl $(MODELS_DIR)/q_table_backup_$(shell date +%Y%m%d_%H%M%S).pkl; \
		echo "✅ Model backed up"; \
	else \
		echo "❌ No model found to backup"; \
	fi

# Show logs
logs: ## Show application logs (if running in background)
	@echo "📜 Application logs:"
	@tail -f logs/app.log 2>/dev/null || echo "No log file found"
