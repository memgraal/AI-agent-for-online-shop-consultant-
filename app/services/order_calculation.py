"""Pure, database-free validation and calculation for explicitly supplied order data."""

from decimal import Decimal, ROUND_HALF_UP

from app.schemas.orders import (
    AppliedDiscount,
    CompanyDetailsInput,
    CompanyDetailsValidation,
    IdentifierCheck,
    OrderCalculationInput,
    OrderCalculationResult,
)

_CENT = Decimal("0.01")
_INN_10_WEIGHTS = (2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN_11_WEIGHTS = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN_12_WEIGHTS = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)


def validate_company_details(data: CompanyDetailsInput) -> CompanyDetailsValidation:
    checks = [_check_inn(data.inn)]
    if data.ogrn:
        checks.append(_check_ogrn(data.ogrn))
    if data.kpp:
        is_valid = len(data.kpp) == 9 and data.kpp.isdigit()
        checks.append(IdentifierCheck(
            identifier="kpp",
            value=data.kpp,
            format_valid=is_valid,
            checksum_valid=None,
            message="КПП должен содержать 9 цифр." if not is_valid else "Формат КПП корректен.",
        ))
    return CompanyDetailsValidation(
        is_valid=all(check.format_valid and check.checksum_valid is not False for check in checks),
        checks=checks,
    )


def _check_inn(value: str) -> IdentifierCheck:
    if not value.isdigit() or len(value) not in {10, 12}:
        return IdentifierCheck(
            identifier="inn", value=value, format_valid=False, checksum_valid=None,
            message="ИНН должен содержать 10 цифр для организации или 12 цифр для ИП.",
        )
    digits = [int(char) for char in value]
    if len(digits) == 10:
        expected = _checksum(digits[:9], _INN_10_WEIGHTS)
        valid = digits[9] == expected
    else:
        first_check = _checksum(digits[:10], _INN_11_WEIGHTS)
        second_check = _checksum(digits[:11], _INN_12_WEIGHTS)
        valid = digits[10] == first_check and digits[11] == second_check
    return IdentifierCheck(
        identifier="inn", value=value, format_valid=True, checksum_valid=valid,
        message="Контрольная сумма ИНН корректна." if valid else "Контрольная сумма ИНН не совпадает.",
    )


def _check_ogrn(value: str) -> IdentifierCheck:
    if not value.isdigit() or len(value) not in {13, 15}:
        return IdentifierCheck(
            identifier="ogrn", value=value, format_valid=False, checksum_valid=None,
            message="ОГРН должен содержать 13 цифр, ОГРНИП — 15 цифр.",
        )
    divisor = 11 if len(value) == 13 else 13
    expected = (int(value[:-1]) % divisor) % 10
    valid = int(value[-1]) == expected
    return IdentifierCheck(
        identifier="ogrn", value=value, format_valid=True, checksum_valid=valid,
        message="Контрольная сумма ОГРН корректна." if valid else "Контрольная сумма ОГРН не совпадает.",
    )


def _checksum(digits: list[int], weights: tuple[int, ...]) -> int:
    return sum(digit * weight for digit, weight in zip(digits, weights, strict=True)) % 11 % 10


def calculate_order_total(data: OrderCalculationInput) -> OrderCalculationResult:
    subtotal = _money(sum(
        (item.unit_price * item.quantity for item in data.items),
        start=Decimal("0"),
    ))
    qualifying = []
    for kind in ("volume", "loyalty"):
        base = subtotal if kind == "volume" else data.prior_purchase_total
        matching = [
            rule for rule in data.discount_rules
            if rule.kind == kind and base >= rule.threshold
        ]
        if matching:
            qualifying.append(max(matching, key=lambda rule: (rule.percent, rule.threshold)))

    if data.stacking == "best" and qualifying:
        qualifying = [max(qualifying, key=lambda rule: (rule.percent, rule.threshold))]

    current_total = subtotal
    applied: list[AppliedDiscount] = []
    for rule in qualifying:
        amount = _money(current_total * rule.percent / Decimal("100"))
        current_total = _money(current_total - amount)
        applied.append(AppliedDiscount(
            kind=rule.kind,
            label=rule.label,
            percent=rule.percent,
            amount=amount,
        ))

    return OrderCalculationResult(
        subtotal=subtotal,
        applied_discounts=applied,
        total_discount=_money(subtotal - current_total),
        total=current_total,
        stacking=data.stacking,
    )


def _money(value: Decimal) -> Decimal:
    return value.quantize(_CENT, rounding=ROUND_HALF_UP)
