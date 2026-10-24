"""Initial FAQ question, event, and candidate tables."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20261023_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "faq_questions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("normalized_text", sa.String(length=1000), nullable=False),
        sa.Column("question_text", sa.String(length=1000), nullable=False),
        sa.Column("occurrence_count", sa.Integer(), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("normalized_text", name="uq_faq_questions_normalized_text"),
    )
    op.create_table(
        "faq_question_events",
        sa.Column("event_id", sa.String(length=36), nullable=False),
        sa.Column("question_id", sa.String(length=36), nullable=False),
        sa.Column("conversation_id", sa.String(length=128), nullable=True),
        sa.Column("received_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["question_id"], ["faq_questions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("event_id"),
    )
    op.create_index("ix_faq_question_events_question_id", "faq_question_events", ["question_id"])
    op.create_table(
        "faq_candidates",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("category", sa.String(length=80), nullable=False),
        sa.Column("normalized_question", sa.String(length=1000), nullable=False),
        sa.Column("frequency", sa.Integer(), nullable=False),
        sa.Column("faq_score", sa.Integer(), nullable=False),
        sa.Column("priority", sa.String(length=24), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("source_indexes", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("is_published", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_faq_candidates_is_published", "faq_candidates", ["is_published"])


def downgrade() -> None:
    op.drop_index("ix_faq_candidates_is_published", table_name="faq_candidates")
    op.drop_table("faq_candidates")
    op.drop_index("ix_faq_question_events_question_id", table_name="faq_question_events")
    op.drop_table("faq_question_events")
    op.drop_table("faq_questions")
