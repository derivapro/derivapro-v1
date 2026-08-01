"""add registration profile fields

Revision ID: 7d8e9f0a1b2c
Revises: c1a2f3b4d5e6
Create Date: 2026-08-01 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa


revision = "7d8e9f0a1b2c"
down_revision = "c1a2f3b4d5e6"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.add_column(sa.Column("full_name", sa.String(length=150), nullable=True))
        batch_op.add_column(sa.Column("email", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("organization", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("intended_use", sa.String(length=255), nullable=True))
        batch_op.add_column(
            sa.Column("accepted_terms", sa.Boolean(), nullable=False, server_default=sa.false())
        )
        batch_op.add_column(sa.Column("accepted_terms_at", sa.DateTime(), nullable=True))
        batch_op.create_index(batch_op.f("ix_users_email"), ["email"], unique=True)

    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.alter_column("accepted_terms", server_default=None)


def downgrade():
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_users_email"))
        batch_op.drop_column("accepted_terms_at")
        batch_op.drop_column("accepted_terms")
        batch_op.drop_column("intended_use")
        batch_op.drop_column("organization")
        batch_op.drop_column("email")
        batch_op.drop_column("full_name")
