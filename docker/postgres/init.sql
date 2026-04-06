-- docker/postgres/init.sql
-- VIT Sports Intelligence Network - PostgreSQL Initialization

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create read-only user for monitoring
CREATE USER IF NOT EXISTS vit_reader WITH PASSWORD 'readonly_password';
GRANT CONNECT ON DATABASE vit_db TO vit_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO vit_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO vit_reader;

-- Create performance monitoring schema
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Table size monitoring view
CREATE OR REPLACE VIEW monitoring.table_sizes AS
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
    pg_total_relation_size(schemaname||'.'||tablename) as total_size_bytes
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY total_size_bytes DESC;

-- Index usage monitoring
CREATE OR REPLACE VIEW monitoring.index_usage AS
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan as scans,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched
FROM pg_stat_user_indexes
ORDER BY scans ASC;

-- Slow query tracking
CREATE OR REPLACE VIEW monitoring.slow_queries AS
SELECT 
    query,
    calls,
    total_exec_time,
    mean_exec_time,
    max_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 50;

-- Create function for automatic CLV calculation trigger
CREATE OR REPLACE FUNCTION update_clv_on_result()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE clv_entries
    SET 
        closing_odds = CASE 
            WHEN bet_side = 'home' THEN NEW.closing_odds_home
            WHEN bet_side = 'draw' THEN NEW.closing_odds_draw
            WHEN bet_side = 'away' THEN NEW.closing_odds_away
        END,
        clv = (entry_odds - closing_odds) / closing_odds,
        bet_outcome = NEW.actual_outcome,
        profit = CASE 
            WHEN bet_side = NEW.actual_outcome THEN (SELECT recommended_stake FROM predictions WHERE id = clv_entries.prediction_id) * (entry_odds - 1)
            ELSE -(SELECT recommended_stake FROM predictions WHERE id = clv_entries.prediction_id)
        END
    WHERE match_id = NEW.id;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for automatic CLV updates (if not exists)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trigger_update_clv') THEN
        CREATE TRIGGER trigger_update_clv
            AFTER UPDATE OF actual_outcome ON matches
            FOR EACH ROW
            EXECUTE FUNCTION update_clv_on_result();
    END IF;
END $$;

-- Create materialized view for daily performance
DROP MATERIALIZED VIEW IF EXISTS mv_daily_performance;
CREATE MATERIALIZED VIEW mv_daily_performance AS
SELECT 
    DATE(p.timestamp) as date,
    COUNT(*) as total_predictions,
    AVG(p.vig_free_edge) as avg_edge,
    AVG(c.clv) as avg_clv,
    SUM(CASE WHEN c.bet_outcome = 'win' THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN c.bet_outcome = 'loss' THEN 1 ELSE 0 END) as losses,
    SUM(c.profit) as total_profit
FROM predictions p
LEFT JOIN clv_entries c ON p.id = c.prediction_id
GROUP BY DATE(p.timestamp);

CREATE UNIQUE INDEX idx_mv_daily_performance ON mv_daily_performance(date);

-- Refresh function for materialized view
CREATE OR REPLACE FUNCTION refresh_daily_performance()
RETURNS TRIGGER AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_daily_performance;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Autovacuum tuning for large tables
ALTER TABLE predictions SET (autovacuum_vacuum_scale_factor = 0.05, autovacuum_analyze_scale_factor = 0.02);
ALTER TABLE clv_entries SET (autovacuum_vacuum_scale_factor = 0.05, autovacuum_analyze_scale_factor = 0.02);

-- Create partitioned table for predictions (if not exists)
CREATE TABLE IF NOT EXISTS predictions_partitioned (LIKE predictions INCLUDING ALL) PARTITION BY RANGE (timestamp);

-- Create partition for current month
DO $$
DECLARE
    current_month TEXT;
    partition_name TEXT;
BEGIN
    current_month := TO_CHAR(CURRENT_DATE, 'YYYY_MM');
    partition_name := 'predictions_' || current_month;

    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I PARTITION OF predictions_partitioned
        FOR VALUES FROM (%L) TO (%L)',
        partition_name,
        DATE_TRUNC('month', CURRENT_DATE),
        DATE_TRUNC('month', CURRENT_DATE + INTERVAL '1 month')
    );
END $$;

-- Notify completion
DO $$
BEGIN
    RAISE NOTICE 'VIT Database initialization complete';
END $$;