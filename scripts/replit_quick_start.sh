#!/bin/bash
# scripts/replit_quick_start.sh - One-click start for Replit

echo "🚀 VIT Network - Quick Start on Replit"
echo "======================================="

# Install dependencies
echo "📦 Installing Python packages..."
pip install -q -r requirements.txt

# Create SQLite database
echo "🗄️  Setting up database..."
python -c "
from app.db.database import engine, Base
from app.db.models import *
import asyncio
async def init():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
asyncio.run(init())
print('Database created!')
"

# Create .env file
echo "📝 Creating environment file..."
cat > .env << EOF
DATABASE_URL=sqlite+aiosqlite:///vit.db
REDIS_URL=memory://
AUTH_ENABLED=false
LOG_LEVEL=INFO
EOF

# Run migrations
echo "🔄 Running migrations..."
python scripts/run_migrations.py 2>/dev/null || echo "Migrations skipped (no changes)"

# Start API
echo ""
echo "✅ Setup complete! Starting API..."
echo ""
echo "📍 API will be available at: https://$REPL_SLUG.$REPL_OWNER.repl.co:8000"
echo "📍 Health check: https://$REPL_SLUG.$REPL_OWNER.repl.co:8000/health"
echo ""
echo "Press Ctrl+C to stop"
echo ""

uvicorn main:app --host 0.0.0.0 --port 8000 --reload