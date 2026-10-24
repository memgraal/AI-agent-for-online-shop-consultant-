"""Pure scheduling rules for FAQ batch analysis."""

from datetime import datetime, timedelta


def should_rebuild_candidates(
    *,
    force: bool,
    new_questions: int,
    last_analyzed_at: datetime | None,
    now: datetime,
    min_new_questions: int,
    min_interval_hours: int,
) -> bool:
    """Return whether accumulated questions justify another expensive analysis."""
    if force:
        return True
    if new_questions >= min_new_questions:
        return True
    if new_questions <= 0 or last_analyzed_at is None:
        return False
    return now - last_analyzed_at >= timedelta(hours=min_interval_hours)
