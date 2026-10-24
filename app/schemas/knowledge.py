"""Knowledge document API input schema."""

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeDocumentInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    document_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1, max_length=200000)
    source_url: str = Field(default="", max_length=2000)
    product_sku: str = Field(default="", max_length=200)
    section: str = Field(default="", max_length=500)
    page: int | None = Field(default=None, ge=1)


class KnowledgeDocumentStored(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1, max_length=200)
    chunks_added: int = Field(ge=0)
    storage: str = "process_memory"
