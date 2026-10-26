from datetime import UTC, datetime

import pytest

from app.schemas.content_controls import ContentControlsInput
from app.services.content_controls import ContentControlsStore
from app.storage.database import AssistantContentHistory, AssistantContentState


class _Result:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value

    def scalar_one(self):
        return self.value

    def scalars(self):
        return self

    def all(self):
        return self.value


class _Session:
    def __init__(self, state, history=()):
        self.state = state
        self.history = list(history)
        self.added = []
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def execute(self, statement):
        # The store uses one select for current state and one for history.
        if "ai_assistant_content_history" in str(statement):
            return _Result(self.history)
        return _Result(self.state)

    def add(self, value):
        self.added.append(value)

    async def commit(self):
        self.commits += 1
        if self.state is not None and self.state.updated_at is None:
            self.state.updated_at = datetime.now(UTC)


class _Factory:
    def __init__(self, session):
        self.session = session

    def __call__(self):
        return self.session


@pytest.mark.asyncio
async def test_initialize_creates_default_persistent_state():
    session = _Session(None)
    store = ContentControlsStore()

    await store.initialize(_Factory(session))

    assert session.commits == 1
    assert len(session.added) == 2
    assert store.get().revision == 1


@pytest.mark.asyncio
async def test_persistent_update_refreshes_cache_and_revision():
    now = datetime.now(UTC)
    state = AssistantContentState(
        id=1,
        revision=1,
        controls=ContentControlsInput().model_dump(mode="json"),
        updated_at=now,
    )
    history = [AssistantContentHistory(
        revision=1,
        controls=state.controls,
        updated_at=now,
    )]
    session = _Session(state, history)
    store = ContentControlsStore()
    await store.initialize(_Factory(session))

    updated = await store.update_persistent(
        ContentControlsInput(tone="concise"),
        _Factory(session),
    )

    assert updated.revision == 2
    assert updated.controls.tone == "concise"
    assert store.get().controls.tone == "concise"
    assert session.commits == 1
