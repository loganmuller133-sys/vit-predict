#!/bin/bash
# scripts/start_replit.sh - Startup script for Replit

set -e

echo "========================================="
echo "VIT Sports Intelligence Network"
echo "Starting on Replit..."
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Docker is not running. Attempting to start...${NC}"
        
        # Try to start Docker daemon
        if command -v dockerd > /dev/null; then
            sudo dockerd &
            sleep 5
        else
            echo -e "${RED}❌ Docker daemon not available.${NC}"
            echo "Falling back to direct Python execution..."
            return 1
        fi
    fi
    
    echo -e "${GREEN}✅ Docker is running${NC}"
    return 0
}

# Function to run without Docker (fallback)
run_native() {
    echo -e "${YELLOW}📦 Running in native mode (without Docker)...${NC}"
    
    # Install Python dependencies
    pip install -r requirements.txt
    
    # Set up SQLite (no PostgreSQL needed)
    export DATABASE_URL="sqlite+aiosqlite:///vit.db"
    export REDIS_URL="memory://"
    export AUTH_ENABLED="false"
    
    # Run migrations
    python scripts/run_migrations.py
    
    # Start services in background
    echo "Starting ML service..."
    uvicorn services.ml_service.main:app --host 0.0.0.0 --port 8001 &
    ML_PID=$!
    
    echo "Starting Celery worker..."
    celery -A app.worker.celery_app worker --loglevel=info --concurrency=2 &
    WORKER_PID=$!
    
    # Start main API
    echo "Starting API service..."
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
}

# Function to run with Docker Compose
run_docker() {
    echo -e "${GREEN}🐳 Running with Docker Compose...${NC}"
    
    # Create secrets directory and files
    mkdir -p secrets
    if [ ! -f secrets/db_password.txt ]; then
        echo "vit_password_$(date +%s)" > secrets/db_password.txt
    fi
    if [ ! -f secrets/api_key.txt ]; then
        echo "vit_api_key_$(date +%s)" > secrets/api_key.txt
    fi
    
    # Copy environment file
    if [ ! -f .env ]; then
        cp .env.example .env 2>/dev/null || echo "DATABASE_URL=postgresql+asyncpg://vit_user:vit_password@postgres:5432/vit_db" > .env
    fi
    
    # Build and start services
    docker-compose up --build -d
    
    echo ""
    echo -e "${GREEN}✅ Services started!${NC}"
    echo ""
    echo "📍 Access your services at:"
    echo "   API:        https://$REPL_SLUG.$REPL_OWNER.repl.co:8000"
    echo "   ML Service: https://$REPL_SLUG.$REPL_OWNER.repl.co:8001"
    echo "   Flower:     https://$REPL_SLUG.$REPL_OWNER.repl.co:5555"
    echo ""
    echo "📊 Check status: docker-compose ps"
    echo "📋 View logs:    docker-compose logs -f"
    echo "🛑 Stop all:     docker-compose down"
    
    # Keep the process running and show logs
    docker-compose logs -f
}

# Main execution
if [ -n "${REPL_SLUG:-}" ] || [ -n "${REPL_ID:-}" ] || ! tty -s; then
    echo "Auto-detected non-interactive/Replit environment."
    REPLY="1"
else
    echo ""
    echo "Select execution mode:"
    echo "1) Docker Compose (full stack)"
    echo "2) Native Python (lightweight)"
    echo ""
    read -p "Enter choice (1 or 2): " -n 1 -r
    echo ""
    REPLY=${REPLY:-1}
fi

if [[ $REPLY =~ ^[1]$ ]]; then
    if check_docker; then
        run_docker
    else
        echo -e "${YELLOW}⚠️  Docker not available, falling back to native mode...${NC}"
        run_native
    fi
else
    run_native
fi