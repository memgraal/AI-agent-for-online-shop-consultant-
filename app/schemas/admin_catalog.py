"""Validated contracts for catalog preparation without a connected CMS."""

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProductAttribute(StrictModel):
    name: str = Field(min_length=1, max_length=200)
    value: str = Field(min_length=1, max_length=1000)


class AttributeNormalizationRequest(StrictModel):
    attributes: list[ProductAttribute] = Field(min_length=1, max_length=100)


class NormalizedAttribute(StrictModel):
    source_name: str = Field(min_length=1, max_length=200)
    source_value: str = Field(min_length=1, max_length=1000)
    normalized_name: str = Field(min_length=1, max_length=200)
    normalized_value: str = Field(min_length=1, max_length=1000)
    explanation: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0.0, le=1.0)


class AttributeNormalizationResponse(StrictModel):
    attributes: list[NormalizedAttribute] = Field(min_length=1, max_length=100)


class DuplicateProduct(StrictModel):
    sku: str = Field(default="", max_length=200)
    name: str = Field(min_length=1, max_length=500)
    manufacturer: str = Field(default="", max_length=300)
    attributes: dict[str, str] = Field(default_factory=dict, max_length=100)


class DuplicateScanRequest(StrictModel):
    # Exhaustive pairwise matching is O(n²); keep an in-memory batch bounded.
    products: list[DuplicateProduct] = Field(min_length=2, max_length=100)


class DuplicateCandidate(StrictModel):
    left_index: int = Field(ge=0)
    right_index: int = Field(ge=0)
    left_sku: str
    right_sku: str
    left_name: str
    right_name: str
    confidence: int = Field(ge=0, le=100)
    matched_fields: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1, max_length=1000)


class DuplicateScanResponse(StrictModel):
    product_count: int = Field(ge=2)
    candidates: list[DuplicateCandidate]
    note: str = "Кандидаты требуют проверки администратором; записи не изменялись."
