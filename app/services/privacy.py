import re

_PERSONAL_DATA_PATTERNS = (
    re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-zА-Яа-я]{2,}\b"),
    re.compile(r"(?<!\w)(?:\+?\d[\s().-]*){10,15}(?!\w)"),
    re.compile(r"\b\d{10,12}\b"),
)


def redact_personal_data(value: str) -> str:
    """Mask common email, phone and tax/long numeric identifiers before model calls."""
    for pattern in _PERSONAL_DATA_PATTERNS:
        value = pattern.sub("[СКРЫТО]", value)
    return value
