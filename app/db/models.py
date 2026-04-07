# app/db/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean, JSON, Index, CheckConstraint
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .database import Base


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, index=True)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    league = Column(String, nullable=False)
    # ✅ FIXED: DateTime without timezone - stores naive UTC
    # This matches the TIMESTAMP WITHOUT TIME ZONE PostgreSQL column type
    kickoff_time = Column(DateTime, nullable=False)
    status = Column(String, default="scheduled")

    # Actual results (filled post-match)
    home_goals = Column(Integer, nullable=True)
    away_goals = Column(Integer, nullable=True)
    actual_outcome = Column(String, nullable=True)  # home/draw/away

    # Market data
    opening_odds_home = Column(Float, nullable=True)
    opening_odds_draw = Column(Float, nullable=True)
    opening_odds_away = Column(Float, nullable=True)
    closing_odds_home = Column(Float, nullable=True)
    closing_odds_draw = Column(Float, nullable=True)
    closing_odds_away = Column(Float, nullable=True)

    # Timestamps (with timezone - for system operations)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    predictions = relationship("Prediction", back_populates="match", uselist=False)
    clv_entries = relationship("CLVEntry", back_populates="match")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    request_hash = Column(String, unique=True, nullable=True, index=True)

    # Multi-market predictions
    home_prob = Column(Float, nullable=False)
    draw_prob = Column(Float, nullable=False)
    away_prob = Column(Float, nullable=False)
    over_25_prob = Column(Float, nullable=True)
    under_25_prob = Column(Float, nullable=True)
    btts_prob = Column(Float, nullable=True)
    no_btts_prob = Column(Float, nullable=True)

    # Metadata
    consensus_prob = Column(Float)
    final_ev = Column(Float)
    recommended_stake = Column(Float)
    model_weights = Column(JSON)
    confidence = Column(Float)

    # Market comparison
    bet_side = Column(String, nullable=True)  # home/draw/away - which side was bet
    entry_odds = Column(Float)  # Odds when prediction was made
    raw_edge = Column(Float)  # model_prob - market_prob (unadjusted)
    normalized_edge = Column(Float)  # After removing bookmaker margin
    vig_free_edge = Column(Float)  # True edge after vig removal

    # ✅ FIXED: timestamp with timezone for system operations
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    # Constraints
    __table_args__ = (
        CheckConstraint('home_prob >= 0 AND home_prob <= 1', name='check_home_prob'),
        CheckConstraint('draw_prob >= 0 AND draw_prob <= 1', name='check_draw_prob'),
        CheckConstraint('away_prob >= 0 AND away_prob <= 1', name='check_away_prob'),
        CheckConstraint('recommended_stake >= 0 AND recommended_stake <= 0.20', name='check_stake_limit'),
    )

    match = relationship("Match", back_populates="predictions")


class CLVEntry(Base):
    """Closing Line Value tracking - the truth metric"""
    __tablename__ = "clv_entries"

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"))
    prediction_id = Column(Integer, ForeignKey("predictions.id"))

    bet_side = Column(String, nullable=False)  # home/draw/away - CRITICAL for accurate CLV
    entry_odds = Column(Float, nullable=False)
    closing_odds = Column(Float, nullable=True)
    clv = Column(Float, nullable=True)  # (entry - closing) / closing

    bet_outcome = Column(String, nullable=True)  # win/loss/pending
    profit = Column(Float, nullable=True)

    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    match = relationship("Match", back_populates="clv_entries")


class Edge(Base):
    """Edge database - profitable patterns"""
    __tablename__ = "edges"

    id = Column(Integer, primary_key=True, index=True)
    edge_id = Column(String, unique=True, nullable=False)
    description = Column(String)

    # Performance metrics
    roi = Column(Float, default=0.0)
    sample_size = Column(Integer, default=0)
    confidence = Column(Float, default=0.0)
    avg_edge = Column(Float, default=0.0)

    # Filters that define this edge
    league = Column(String, nullable=True)
    home_condition = Column(String, nullable=True)
    away_condition = Column(String, nullable=True)
    market = Column(String, default="1x2")

    # Lifecycle
    status = Column(String, default="active")  # active, declining, dead, revived
    decay_rate = Column(Float, default=0.02)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_updated = Column(DateTime(timezone=True), onupdate=func.now())
    archived_at = Column(DateTime(timezone=True), nullable=True)


class ModelPerformance(Base):
    __tablename__ = "model_performances"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, unique=True, nullable=False)
    model_type = Column(String, nullable=False)
    version = Column(Integer, default=1)

    weight_decay_rate = Column(Float, default=0.05)
    min_weight_threshold = Column(Float, default=0.05)
    performance_window = Column(Integer, default=100)
    last_weight_update = Column(DateTime(timezone=True), nullable=True)
    consecutive_underperforming = Column(Integer, default=0)

    # Performance metrics
    accuracy_score = Column(Float)
    current_weight = Column(Float, default=1.0)
    calibration_error = Column(Float)
    expected_value = Column(Float)
    sharpe_ratio = Column(Float)
    positive_clv_rate = Column(Float, default=0.0)  # Track CLV by model

    # Certification
    certified = Column(Boolean, default=False)
    final_score = Column(Float, nullable=True)
    last_certified_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())


# Indexes for performance
Index('idx_matches_kickoff', Match.kickoff_time)
Index('idx_matches_status', Match.status)
Index('idx_predictions_timestamp', Prediction.timestamp.desc())
Index('idx_predictions_match_id', Prediction.match_id)
Index('idx_clv_match', CLVEntry.match_id)
Index('idx_clv_bet_side', CLVEntry.bet_side)
Index('idx_edges_status', Edge.status)
Index('idx_edges_roi', Edge.roi.desc())
Index('idx_model_perf_certified', ModelPerformance.certified)