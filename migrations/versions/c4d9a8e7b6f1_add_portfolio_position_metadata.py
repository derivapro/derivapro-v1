"""add portfolio position metadata

Revision ID: c4d9a8e7b6f1
Revises: ab34c9d7e201
Create Date: 2026-08-10 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "c4d9a8e7b6f1"
down_revision = "ab34c9d7e201"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "positions",
        sa.Column("side", sa.String(length=10), nullable=False, server_default="long"),
    )
    op.add_column(
        "positions",
        sa.Column("position_label", sa.String(length=150), nullable=True),
    )
    op.add_column(
        "positions",
        sa.Column("currency", sa.String(length=10), nullable=False, server_default="USD"),
    )
    op.add_column(
        "positions",
        sa.Column("asset_class", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "positions",
        sa.Column("product_category", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "positions",
        sa.Column("underlying", sa.String(length=100), nullable=True),
    )
    op.add_column("positions", sa.Column("notes", sa.Text(), nullable=True))


def downgrade():
    op.drop_column("positions", "notes")
    op.drop_column("positions", "underlying")
    op.drop_column("positions", "product_category")
    op.drop_column("positions", "asset_class")
    op.drop_column("positions", "currency")
    op.drop_column("positions", "position_label")
    op.drop_column("positions", "side")
