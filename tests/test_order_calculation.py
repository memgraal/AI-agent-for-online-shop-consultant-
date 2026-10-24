from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.schemas.orders import CompanyDetailsInput, OrderCalculationInput
from app.services.order_calculation import calculate_order_total, validate_company_details


def test_company_identifier_checksums_and_hyphenated_input():
    result = validate_company_details(CompanyDetailsInput(
        inn="770-708-3893",
        ogrn="1027700132195",
        kpp="770701001",
    ))

    assert result.is_valid is True
    assert result.existence_verified is False
    assert [check.identifier for check in result.checks] == ["inn", "ogrn", "kpp"]
    assert result.checks[0].value == "7707083893"


def test_company_identifier_bad_checksum_is_reported_without_external_lookup():
    result = validate_company_details(CompanyDetailsInput(inn="7707083894"))

    assert result.is_valid is False
    assert result.checks[0].format_valid is True
    assert result.checks[0].checksum_valid is False


def test_order_total_uses_supplied_discount_rules_and_decimal_rounding():
    request = OrderCalculationInput.model_validate({
        "items": [
            {"sku": "A-1", "name": "Датчик", "quantity": 2, "unit_price": "100.10"},
            {"sku": "B-2", "name": "Контроллер", "quantity": 1, "unit_price": "299.90"},
        ],
        "prior_purchase_total": "50000",
        "discount_rules": [
            {"kind": "volume", "threshold": "500", "percent": "5", "label": "Корзина от 500"},
            {"kind": "loyalty", "threshold": "40000", "percent": "3", "label": "Накопительная"},
        ],
        "stacking": "best",
    })

    result = calculate_order_total(request)

    assert result.subtotal == Decimal("500.10")
    assert [discount.kind for discount in result.applied_discounts] == ["volume"]
    assert result.total_discount == Decimal("25.01")
    assert result.total == Decimal("475.09")


def test_sequential_discounts_apply_in_order_to_remaining_subtotal():
    request = OrderCalculationInput.model_validate({
        "items": [{"sku": "A", "name": "Датчик", "quantity": 1, "unit_price": "1000"}],
        "prior_purchase_total": "10000",
        "discount_rules": [
            {"kind": "volume", "threshold": "500", "percent": "10", "label": "Объём"},
            {"kind": "loyalty", "threshold": "5000", "percent": "5", "label": "Лояльность"},
        ],
        "stacking": "sequential",
    })

    result = calculate_order_total(request)

    assert [discount.amount for discount in result.applied_discounts] == [
        Decimal("100.00"), Decimal("45.00"),
    ]
    assert result.total == Decimal("855.00")


def test_order_input_rejects_negative_prices_and_unknown_fields():
    with pytest.raises(ValidationError):
        OrderCalculationInput.model_validate({
            "items": [{"sku": "A", "name": "Датчик", "quantity": 1, "unit_price": "-1"}],
            "stacking": "best",
        })

    with pytest.raises(ValidationError):
        OrderCalculationInput.model_validate({
            "items": [{"sku": "A", "name": "Датчик", "quantity": 1, "unit_price": "10"}],
            "stacking": "best",
            "unknown_rule": "ignored no longer",
        })
