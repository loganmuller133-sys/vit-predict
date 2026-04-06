# alembic/versions/002_add_idempotency_constraint.py
"""Add idempotency constraints

Revision ID: 002
Revises: 001
Create Date: 2024-01-02 00:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add unique constraint to prevent duplicate predictions for same match
    op.create_unique_constraint(
        'uq_predictions_match_id',
        'predictions',
        ['match_id']
    )
    
    # Add request_hash column for API idempotency
    op.add_column('predictions', sa.Column('request_hash', sa.String(64), nullable=True))
    op.create_index('idx_predictions_request_hash', 'predictions', ['request_hash'])
    op.create_unique_constraint('uq_predictions_request_hash', 'predictions', ['request_hash'])


def downgrade() -> None:
    op.drop_constraint('uq_predictions_request_hash', 'predictions', type_='unique')
    op.drop_index('idx_predictions_request_hash')
    op.drop_column('predictions', 'request_hash')
    op.drop_constraint('uq_predictions_match_id', 'predictions', type_='unique')