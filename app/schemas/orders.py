"""Validated request and response models for database-free checkout helpers."""

from decimal import Decimal
import re
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.enums import Currency, DiscountKind, DiscountStacking, IdentifierKind


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CompanyDetailsInput(StrictModel):
    inn: str = Field(min_length=1, max_length=20)
    ogrn: str | None = Field(default=None, max_length=20)
    kpp: str | None = Field(default=None, max_length=20)
    company_name: str = Field(default="", max_length=500)

    @field_validator("inn", "ogrn", "kpp", mode="before")
    @classmethod
    def normalize_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return re.sub(r"[\s-]", "", str(value))


class IdentifierCheck(StrictModel):
    identifier: IdentifierKind
    value: str
    format_valid: bool
    checksum_valid: bool | None
    message: str


class CompanyDetailsValidation(StrictModel):
    is_valid: bool
    existence_verified: Literal[False] = False
    checks: list[IdentifierCheck]
    note: str = "Проверены формат и контрольная сумма; существование организации не проверялось."


class OrderItem(StrictModel):
    sku: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=500)
    quantity: int = Field(ge=1, le=100_000)
    unit_price: Decimal = Field(gt=Decimal("0"), max_digits=12, decimal_places=2)


class DiscountRule(StrictModel):
    kind: DiscountKind
    threshold: Decimal = Field(gt=Decimal("0"), max_digits=14, decimal_places=2)
    percent: Decimal = Field(gt=Decimal("0"), le=Decimal("100"), max_digits=5, decimal_places=2)
    label: str = Field(min_length=1, max_length=200)


class OrderCalculationInput(StrictModel):
    items: list[OrderItem] = Field(min_length=1, max_length=500)
    prior_purchase_total: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), max_digits=14, decimal_places=2)
    discount_rules: list[DiscountRule] = Field(default_factory=list, max_length=50)
    stacking: DiscountStacking


class AppliedDiscount(StrictModel):
    kind: DiscountKind
    label: str
    percent: Decimal
    amount: Decimal


class OrderCalculationResult(StrictModel):
    currency: Currency = Currency.RUB
    subtotal: Decimal
    applied_discounts: list[AppliedDiscount]
    total_discount: Decimal
    total: Decimal
    stacking: DiscountStacking
    note: str = "Расчёт выполнен по переданным ценам и правилам; это не проверка цены или наличия в каталоге."
