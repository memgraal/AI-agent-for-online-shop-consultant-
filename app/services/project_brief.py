"""Pure formatting helpers for a customer-ready, factual project intake brief."""

import json


def render_installation_brief(
    *,
    object_type: str,
    area_m2: float | None,
    rooms_or_zones: list[str],
    security_goals: str,
    communication_channels: list[str],
    backup_autonomy_hours: float | None,
    constraints: str,
) -> str:
    if area_m2 is not None and area_m2 <= 0:
        raise ValueError("Площадь должна быть больше нуля.")
    if backup_autonomy_hours is not None and backup_autonomy_hours < 0:
        raise ValueError("Автономность не может быть отрицательной.")

    supplied = {
        "Тип объекта": object_type.strip(),
        "Площадь, м²": area_m2,
        "Зоны или помещения": rooms_or_zones,
        "Задачи охраны/оповещения": security_goals.strip(),
        "Нужные каналы связи": communication_channels,
        "Требуемая автономность, ч": backup_autonomy_hours,
        "Ограничения и пожелания": constraints.strip(),
    }
    missing = [
        name for name, value in supplied.items()
        if value is None or value == "" or value == []
    ]
    required = ("Тип объекта", "Площадь, м²", "Задачи охраны/оповещения")
    complete = sum(bool(supplied[name]) for name in required)
    completeness = round(complete / len(required) * 100)
    payload = {
        "Название": "Предварительное ТЗ на систему безопасности",
        "Заполненность основных вводных, %": completeness,
        "Вводные от клиента": supplied,
        "Что уточнить для точного подбора": missing,
        "Следующий шаг": (
            "Передать вводные специалисту для подбора совместимого оборудования и проверки проекта."
        ),
        "Примечание": "Это сбор исходных требований, а не готовый проект или инженерный расчёт.",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
