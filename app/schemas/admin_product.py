"""Structured inputs and drafts for AI-assisted product administration."""

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ProductContentInput(StrictModel):
    name: str = Field(min_length=1, max_length=500)
    manufacturer: str = Field(min_length=1, max_length=300)
    attributes: str = Field(min_length=1, max_length=12000)


class ProductClassificationDraft(StrictModel):
    category: str = Field(min_length=1, max_length=200)
    product_type: str = Field(min_length=1, max_length=200)
    purpose: str = Field(min_length=1, max_length=500)
    tags: list[str] = Field(max_length=30)


class ProductSEODraft(StrictModel):
    description: str = Field(
        min_length=1,
        max_length=5000,
        validation_alias=AliasChoices("description", "description_draft"),
    )
    h1: str = Field(min_length=1, max_length=250)
    meta_title: str = Field(min_length=1, max_length=250)
    meta_description: str = Field(min_length=1, max_length=500)
    category_text: str = Field(
        min_length=1,
        max_length=5000,
        validation_alias=AliasChoices("category_text", "category_page_text"),
    )


class ProductContentDraft(StrictModel):
    classification: ProductClassificationDraft
    seo: ProductSEODraft
    related_product_criteria: dict[str, str]
