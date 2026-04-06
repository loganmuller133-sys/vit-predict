# alembic/versions/001_initial_schema.py
"""Initial schema for VIT Network - Fixed with all constraints

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON

# revision identifiers
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create edge_status enum
    op.execute("CREATE TYPE edge_status AS ENUM ('active', 'declining', 'dead', 'archived')")

    # Create matches table
    op.create_table(
        'matches',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('home_team', sa.String(), nullable=False),
        sa.Column('away_team', sa.String(), nullable=False),
        sa.Column('league', sa.String(), nullable=False),
        sa.Column('kickoff_time', sa.DateTime(), nullable=False),
        sa.Column('status', sa.String(), server_default='scheduled'),
        sa.Column('home_goals', sa.Integer(), nullable=True),
        sa.Column('away_goals', sa.Integer(), nullable=True),
        sa.Column('actual_outcome', sa.String(), nullable=True),
        sa.Column('opening_odds_home', sa.Float(), nullable=True),
        sa.Column('opening_odds_draw', sa.Float(), nullable=True),
        sa.Column('opening_odds_away', sa.Float(), nullable=True),
        sa.Column('closing_odds_home', sa.Float(), nullable=True),
        sa.Column('closing_odds_draw', sa.Float(), nullable=True),
        sa.Column('closing_odds_away', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.text('now()')),
        sa.PrimaryKeyConstraint('id')
    )

    # Create predictions table with COMPLETE constraints
    op.create_table(
        'predictions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('match_id', sa.Integer(), nullable=False),
        sa.Column('home_prob', sa.Float(), nullable=False),
        sa.Column('draw_prob', sa.Float(), nullable=False),
        sa.Column('away_prob', sa.Float(), nullable=False),
        sa.Column('over_25_prob', sa.Float(), nullable=True),
        sa.Column('under_25_prob', sa.Float(), nullable=True),
        sa.Column('btts_prob', sa.Float(), nullable=True),
        sa.Column('no_btts_prob', sa.Float(), nullable=True),
        sa.Column('consensus_prob', sa.Float()),
        sa.Column('final_ev', sa.Float()),
        sa.Column('recommended_stake', sa.Float()),
        sa.Column('model_weights', sa.JSON()),  # Use sa.JSON() for portability
        sa.Column('confidence', sa.Float()),
        sa.Column('bet_side', sa.String(), nullable=True),
        sa.Column('entry_odds', sa.Float()),
        sa.Column('raw_edge', sa.Float()),
        sa.Column('normalized_edge', sa.Float()),
        sa.Column('vig_free_edge', sa.Float()),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['match_id'], ['matches.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        # Individual probability bounds
        sa.CheckConstraint('home_prob >= 0 AND home_prob <= 1', name='check_home_prob'),
        sa.CheckConstraint('draw_prob >= 0 AND draw_prob <= 1', name='check_draw_prob'),
        sa.CheckConstraint('away_prob >= 0 AND away_prob <= 1', name='check_away_prob'),
        # CRITICAL FIX: Sum to 1 constraint
        sa.CheckConstraint(
            'abs(home_prob + draw_prob + away_prob - 1.0) < 0.01',
            name='check_probabilities_sum_to_one'
        ),
        sa.CheckConstraint('recommended_stake >= 0 AND recommended_stake <= 0.20', name='check_stake_limit')
    )

    # Create clv_entries table with CASCADE
    op.create_table(
        'clv_entries',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('match_id', sa.Integer(), nullable=False),
        sa.Column('prediction_id', sa.Integer(), nullable=False),
        sa.Column('bet_side', sa.String(), nullable=False),
        sa.Column('entry_odds', sa.Float(), nullable=False),
        sa.Column('closing_odds', sa.Float(), nullable=True),
        sa.Column('clv', sa.Float(), nullable=True),
        sa.Column('bet_outcome', sa.String(), nullable=True),
        sa.Column('profit', sa.Float(), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['match_id'], ['matches.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['prediction_id'], ['predictions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # Create edges table with ENUM
    op.create_table(
        'edges',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('edge_id', sa.String(), nullable=False),
        sa.Column('description', sa.String()),
        sa.Column('roi', sa.Float(), server_default='0.0'),
        sa.Column('sample_size', sa.Integer(), server_default='0'),
        sa.Column('confidence', sa.Float(), server_default='0.0'),
        sa.Column('avg_edge', sa.Float(), server_default='0.0'),
        sa.Column('league', sa.String(), nullable=True),
        sa.Column('home_condition', sa.String(), nullable=True),
        sa.Column('away_condition', sa.String(), nullable=True),
        sa.Column('market', sa.String(), server_default='1x2'),
        sa.Column('status', sa.Enum('active', 'declining', 'dead', 'archived', name='edge_status'), 
                  server_default='active'),
        sa.Column('decay_rate', sa.Float(), server_default='0.02'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('last_updated', sa.DateTime(timezone=True), onupdate=sa.text('now()')),
        sa.Column('archived_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('edge_id')
    )

    # Create model_performances table
    op.create_table(
        'model_performances',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_name', sa.String(), nullable=False),
        sa.Column('model_type', sa.String(), nullable=False),
        sa.Column('version', sa.Integer(), server_default='1'),
        sa.Column('accuracy_score', sa.Float()),
        sa.Column('current_weight', sa.Float(), server_default='1.0'),
        sa.Column('calibration_error', sa.Float()),
        sa.Column('expected_value', sa.Float()),
        sa.Column('sharpe_ratio', sa.Float()),
        sa.Column('positive_clv_rate', sa.Float(), server_default='0.0'),
        sa.Column('certified', sa.Boolean(), server_default='false'),
        sa.Column('final_score', sa.Float(), nullable=True),
        sa.Column('last_certified_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('model_name')
    )

    # Create indexes
    op.create_index('idx_matches_kickoff', 'matches', ['kickoff_time'])
    op.create_index('idx_matches_status', 'matches', ['status'])
    op.create_index('idx_predictions_timestamp', 'predictions', ['timestamp'])
    op.create_index('idx_predictions_match_id', 'predictions', ['match_id'])
    op.create_index('idx_clv_match', 'clv_entries', ['match_id'])
    op.create_index('idx_clv_bet_side', 'clv_entries', ['bet_side'])
    op.create_index('idx_edges_status', 'edges', ['status'])
    op.create_index('idx_edges_roi', 'edges', ['roi'])
    op.create_index('idx_model_perf_certified', 'model_performances', ['certified'])


def downgrade() -> None:
    op.drop_index('idx_model_perf_certified')
    op.drop_index('idx_edges_roi')
    op.drop_index('idx_edges_status')
    op.drop_index('idx_clv_bet_side')
    op.drop_index('idx_clv_match')
    op.drop_index('idx_predictions_match_id')
    op.drop_index('idx_predictions_timestamp')
    op.drop_index('idx_matches_status')
    op.drop_index('idx_matches_kickoff')
    op.drop_table('model_performances')
    op.drop_table('edges')
    op.drop_table('clv_entries')
    op.drop_table('predictions')
    op.drop_table('matches')
    op.execute("DROP TYPE edge_status")