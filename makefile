# Makefile for VIT Network

.PHONY: help install dev test migrate docker-up docker-down clean lint format backtest monitor

help:
	@echo "VIT Sports Intelligence Network - Makefile Commands"
	@echo ""
	@echo "Development:"
	@echo "  make install     - Install dependencies"
	@echo "  make dev         - Run development server"
	@echo "  make test        - Run all tests"
	@echo "  make lint        - Run linters"
	@echo "  make format      - Format code"
	@echo ""
	@echo "Database:"
	@echo "  make migrate     - Run database migrations"
	@echo "  make seed        - Seed test data"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up   - Start all services"
	@echo "  make docker-down - Stop all services"
	@echo "  make docker-logs - View service logs"
	@echo "  make docker-clean - Remove all containers and volumes"
	@echo ""
	@echo "Utilities:"
	@echo "  make backtest    - Run backtesting"
	@echo "  make monitor     - Run system monitoring"
	@echo "  make clean       - Clean cache files"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

dev:
	uvicorn main:app --reload --port 8000 --host 0.0.0.0

test:
	pytest tests/ -v --cov=app --cov-report=html --cov-report=term

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-coverage:
	pytest tests/ --cov=app --cov-report=html --cov-report=xml

lint:
	ruff check app/ tests/ scripts/
	mypy app/ --ignore-missing-imports

format:
	black app/ tests/ scripts/
	isort app/ tests/ scripts/

migrate:
	python scripts/run_migrations.py

migrate-create:
	python scripts/run_migrations.py create "$(name)"

seed:
	python scripts/seed_data.py

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-clean:
	docker-compose down -v
	docker system prune -f

backtest:
	python scripts/backtest.py

monitor:
	python scripts/monitor.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

.PHONY: all
all: install lint test