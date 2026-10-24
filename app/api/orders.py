"""HTTP endpoints for local checkout validations that don't call an LLM."""

from fastapi import APIRouter

from app.schemas.orders import (
    CompanyDetailsInput,
    CompanyDetailsValidation,
    OrderCalculationInput,
    OrderCalculationResult,
)
from app.services.order_calculation import calculate_order_total, validate_company_details

router = APIRouter(prefix="/api/orders", tags=["checkout helpers"])


@router.post("/validate-company-details", response_model=CompanyDetailsValidation)
async def validate_company_details_api(
    request: CompanyDetailsInput,
) -> CompanyDetailsValidation:
    """Check Russian business identifier formats/checksums without external lookup."""
    return validate_company_details(request)


@router.post("/calculate-total", response_model=OrderCalculationResult)
async def calculate_order_total_api(
    request: OrderCalculationInput,
) -> OrderCalculationResult:
    """Calculate a provisional total only from client-supplied prices and rules."""
    return calculate_order_total(request)
