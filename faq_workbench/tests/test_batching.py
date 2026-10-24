from datetime import datetime, timedelta, timezone

import pytest

from faq_workbench.batching import should_rebuild_candidates


NOW = datetime(2026, 10, 23, 12, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("force", "new_questions", "last_analyzed_at", "expected"),
    [
        (True, 0, None, True),
        (False, 49, NOW - timedelta(days=2), True),
        (False, 49, NOW - timedelta(hours=23), False),
        (False, 50, None, True),
        (False, 1, None, False),
        (False, 1, NOW - timedelta(hours=24), True),
        (False, 0, NOW - timedelta(days=3), False),
    ],
)
def test_should_rebuild_candidates(force, new_questions, last_analyzed_at, expected):
    assert should_rebuild_candidates(
        force=force,
        new_questions=new_questions,
        last_analyzed_at=last_analyzed_at,
        now=NOW,
        min_new_questions=50,
        min_interval_hours=24,
    ) is expected
