"""Add synthetic shadow prediction persistence.

Revision ID: 8e21c0b18a01
Revises: 16b1600164bd
"""

from alembic import op
import sqlalchemy as sa


revision = "8e21c0b18a01"
down_revision = "16b1600164bd"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "shadow_predictions",
        sa.Column(
            "id", sa.UUID(), primary_key=True
        ),
        sa.Column(
            "reference_id",
            sa.String(100),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "source_dataset",
            sa.String(16),
            nullable=False,
        ),
        sa.Column(
            "amount_unit",
            sa.String(32),
            nullable=False,
        ),
        sa.Column(
            "amount",
            sa.Numeric(28, 12),
            nullable=False,
        ),
        sa.Column(
            "transaction_type",
            sa.String(32),
            nullable=False,
        ),
        sa.Column(
            "calibrated_score",
            sa.Numeric(24, 20),
            nullable=False,
        ),
        sa.Column(
            "review_threshold",
            sa.Numeric(24, 20),
            nullable=False,
        ),
        sa.Column(
            "decision",
            sa.String(16),
            nullable=False,
        ),
        sa.Column(
            "model_version",
            sa.String(64),
            nullable=False,
        ),
        sa.Column(
            "model_run_id",
            sa.String(32),
            nullable=False,
        ),
        sa.Column(
            "deployment_mode",
            sa.String(16),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "amount > 0",
            name="ck_shadow_amount_positive",
        ),
        sa.CheckConstraint(
            "source_dataset = 'PAYSIM'",
            name="ck_shadow_source_dataset",
        ),
        sa.CheckConstraint(
            "amount_unit = 'PAYSIM_SYNTHETIC_UNITS'",
            name="ck_shadow_amount_unit",
        ),
        sa.CheckConstraint(
            "transaction_type IN "
            "('TRANSFER', 'CASH_OUT', 'CASH_IN', "
            "'PAYMENT', 'DEBIT')",
            name="ck_shadow_transaction_type",
        ),
        sa.CheckConstraint(
            "calibrated_score BETWEEN 0 AND 1",
            name="ck_shadow_score_range",
        ),
        sa.CheckConstraint(
            "review_threshold BETWEEN 0 AND 1",
            name="ck_shadow_threshold_range",
        ),
        sa.CheckConstraint(
            "decision IN ('REVIEW', 'NO_REVIEW')",
            name="ck_shadow_decision",
        ),
        sa.CheckConstraint(
            "deployment_mode = 'SHADOW_ONLY'",
            name="ck_shadow_deployment_mode",
        ),
    )

    op.create_index(
        "ix_shadow_predictions_created_at",
        "shadow_predictions",
        ["created_at"],
    )

    op.create_index(
        "ix_shadow_predictions_decision_created_at",
        "shadow_predictions",
        ["decision", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_shadow_predictions_decision_created_at",
        table_name="shadow_predictions",
    )
    op.drop_index(
        "ix_shadow_predictions_created_at",
        table_name="shadow_predictions",
    )
    op.drop_table("shadow_predictions")