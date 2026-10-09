"""add user simulation config

Revision ID: ab34c9d7e201
Revises: 7d8e9f0a1b2c
Create Date: 2026-08-08 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "ab34c9d7e201"
down_revision = "7d8e9f0a1b2c"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "user_simulation_configs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("settings_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id"),
    )
    op.create_index(
        op.f("ix_user_simulation_configs_user_id"),
        "user_simulation_configs",
        ["user_id"],
        unique=True,
    )


def downgrade():
    op.drop_index(
        op.f("ix_user_simulation_configs_user_id"),
        table_name="user_simulation_configs",
    )
    op.drop_table("user_simulation_configs")
