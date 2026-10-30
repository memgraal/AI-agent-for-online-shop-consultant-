from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from app import config as config_module
from app.config import Settings
from app.storage import database
from app.storage.database import AssistantSettings


class _Result:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class _Session:
    def __init__(self, existing=None):
        self.existing = existing
        self.added = []
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def execute(self, _statement):
        return _Result(self.existing)

    def add(self, value):
        self.added.append(value)

    async def commit(self):
        self.commits += 1


def test_agent_settings_table_has_runtime_fields_and_json_prompts():
    columns = AssistantSettings.__table__.c

    assert {
        "llm_provider",
        "llm_model",
        "llm_temperature",
        "llm_max_retries",
        "chat_context_max_messages",
        "system_prompts",
    }.issubset(columns.keys())
    assert columns.system_prompts.type.__class__.__name__ == "JSONB"

    ddl = str(
        CreateTable(AssistantSettings.__table__).compile(
            dialect=postgresql.dialect()
        )
    )
    assert "CREATE TABLE ai_agent_settings" in ddl
    assert "JSONB" in ddl


async def test_seed_agent_settings_creates_one_initial_row(monkeypatch):
    session = _Session()

    class _SessionFactory:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return session

        async def __aexit__(self, *_args):
            return None

    monkeypatch.setattr(database, "AsyncSession", _SessionFactory)
    monkeypatch.setattr(
        config_module,
        "settings",
        Settings(
            llm_provider="openai-compatible",
            llm_model="local-test-model",
            llm_base_url="http://127.0.0.1:11434/v1",
        ),
    )

    await database._seed_agent_settings(object())

    assert session.commits == 1
    assert len(session.added) == 1
    row = session.added[0]
    assert row.llm_provider == "openai-compatible"
    assert row.llm_model == "local-test-model"
    assert {
        "customer",
        "admin",
        "faq_analysis",
        "product_seo",
        "attribute_normalization",
    } == set(row.system_prompts)
    assert row.system_prompts["customer"]
    assert not hasattr(row, "llm_api_key")


async def test_seed_agent_settings_does_not_overwrite_existing_row(monkeypatch):
    existing = AssistantSettings(
        id=1,
        system_prompts={
            "customer": "custom customer",
            "admin": "custom admin",
            "faq_analysis": "custom faq",
            "product_seo": "custom seo",
            "attribute_normalization": "custom normalization",
        },
    )
    session = _Session(existing=existing)

    class _SessionFactory:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return session

        async def __aexit__(self, *_args):
            return None

    monkeypatch.setattr(database, "AsyncSession", _SessionFactory)

    await database._seed_agent_settings(object())

    assert session.added == []
    assert session.commits == 0
