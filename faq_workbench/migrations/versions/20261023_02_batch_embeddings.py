"""Add cached semantic embeddings and scheduled-analysis state."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20261023_02"
down_revision = "20261023_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("faq_questions", sa.Column("embedding", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("faq_questions", sa.Column("embedding_model", sa.String(length=160), nullable=True))
    op.add_column(
        "faq_candidates",
        sa.Column("source_questions", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
    )
    op.add_column(
        "faq_candidates",
        sa.Column("source_question_ids", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
    )
    op.create_table(
        "faq_analysis_state",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("last_analyzed_question_count", sa.Integer(), nullable=False),
        sa.Column("last_analyzed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lock_until", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("faq_analysis_state")
    op.drop_column("faq_candidates", "source_question_ids")
    op.drop_column("faq_candidates", "source_questions")
    op.drop_column("faq_questions", "embedding_model")
    op.drop_column("faq_questions", "embedding")
