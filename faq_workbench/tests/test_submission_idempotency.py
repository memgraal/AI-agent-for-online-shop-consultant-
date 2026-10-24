from uuid import uuid4

import pytest
from fastapi import HTTPException

from faq_workbench.api import submit_question
from faq_workbench.schemas import QuestionSubmission


class FakeResult:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value

    def scalar_one(self):
        return self.value


class FakeSession:
    def __init__(self, values):
        self.values = iter(values)
        self.committed = False
        self.rolled_back = False

    async def execute(self, _statement):
        return FakeResult(next(self.values))

    async def commit(self):
        self.committed = True

    async def rollback(self):
        self.rolled_back = True


@pytest.mark.asyncio
async def test_reusing_event_id_for_different_question_rolls_back_new_aggregate():
    session = FakeSession(["new-question-id", None, "other-question-id"])
    payload = QuestionSubmission(question="Как оформить возврат?", event_id=uuid4())

    with pytest.raises(HTTPException) as error:
        await submit_question(payload, session=session)

    assert error.value.status_code == 409
    assert session.rolled_back is True
    assert session.committed is False


@pytest.mark.asyncio
async def test_retry_of_same_question_event_is_acknowledged_without_increment():
    session = FakeSession([None, "existing-question-id", None, "existing-question-id"])
    payload = QuestionSubmission(question="Как оформить возврат?", event_id=uuid4())

    result = await submit_question(payload, session=session)

    assert result["accepted"] is True
    assert session.committed is True
    assert session.rolled_back is False


@pytest.mark.asyncio
async def test_reusing_event_id_for_another_existing_question_returns_conflict():
    session = FakeSession([None, "current-question-id", None, "different-question-id"])
    payload = QuestionSubmission(question="Как изменить адрес доставки?", event_id=uuid4())

    with pytest.raises(HTTPException) as error:
        await submit_question(payload, session=session)

    assert error.value.status_code == 409
    assert session.rolled_back is True
    assert session.committed is False
