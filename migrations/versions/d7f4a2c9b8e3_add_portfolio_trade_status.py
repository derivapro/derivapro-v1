"""add portfolio trade status

Revision ID: d7f4a2c9b8e3
Revises: c4d9a8e7b6f1
Create Date: 2026-08-11 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "d7f4a2c9b8e3"
down_revision = "c4d9a8e7b6f1"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("positions", sa.Column("trade_id", sa.String(length=100), nullable=True))
    op.add_column(
        "positions",
        sa.Column(
            "valuation_status",
            sa.String(length=50),
            nullable=False,
            server_default="unpriced",
        ),
    )


def downgrade():
    op.drop_column("positions", "valuation_status")
    op.drop_column("positions", "trade_id")
