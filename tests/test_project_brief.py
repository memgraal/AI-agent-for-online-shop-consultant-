import json

import pytest

from app.services.project_brief import render_installation_brief
from app.tools.customer import create_installation_brief


def test_project_brief_reports_missing_requirements_and_completeness():
    raw = render_installation_brief(
        object_type="коттедж",
        area_m2=140,
        rooms_or_zones=["гараж", "первый этаж"],
        security_goals="охрана периметра",
        communication_channels=[],
        backup_autonomy_hours=None,
        constraints="без штробления стен",
    )
    brief = json.loads(raw)

    assert brief["Заполненность основных вводных, %"] == 100
    assert "Нужные каналы связи" in brief["Что уточнить для точного подбора"]
    assert "Требуемая автономность, ч" in brief["Что уточнить для точного подбора"]
    assert "не готовый проект" in brief["Примечание"]


async def test_project_brief_is_available_as_async_langchain_tool():
    raw = await create_installation_brief.ainvoke({
        "object_type": "офис",
        "area_m2": 85,
        "security_goals": "контроль входа",
    })
    brief = json.loads(raw)

    assert brief["Вводные от клиента"]["Тип объекта"] == "офис"
    assert brief["Заполненность основных вводных, %"] == 100


def test_project_brief_rejects_impossible_negative_area():
    with pytest.raises(ValueError, match="Площадь"):
        render_installation_brief(
            object_type="коттедж",
            area_m2=-30,
            rooms_or_zones=[],
            security_goals="охрана",
            communication_channels=[],
            backup_autonomy_hours=None,
            constraints="",
        )
