"""Small-batch semantic grouping of aggregated questions by cosine similarity."""

from dataclasses import dataclass, field

from faq_workbench.database import FAQQuestion
from faq_workbench.embeddings import cosine_similarity


@dataclass
class SemanticQuestionGroup:
    representative: FAQQuestion
    embedding: list[float]
    members: list[FAQQuestion] = field(default_factory=list)

    @property
    def frequency(self) -> int:
        return sum(question.occurrence_count for question in self.members)

    def analysis_input(self) -> dict[str, object]:
        return {
            "text": self.representative.question_text,
            "frequency": self.frequency,
            "variants": [
                question.question_text
                for question in self.members
                if question.id != self.representative.id
            ][:5],
        }


def cluster_questions(
    questions: list[FAQQuestion],
    *,
    similarity_threshold: float,
) -> list[SemanticQuestionGroup]:
    """Group each aggregate against a representative to avoid transitive chaining."""
    groups: list[SemanticQuestionGroup] = []
    for question in questions:
        if question.embedding is None:
            continue
        closest_group: SemanticQuestionGroup | None = None
        closest_similarity = similarity_threshold
        for group in groups:
            similarity = cosine_similarity(question.embedding, group.embedding)
            if similarity >= closest_similarity:
                closest_group = group
                closest_similarity = similarity
        if closest_group is None:
            groups.append(SemanticQuestionGroup(question, question.embedding, [question]))
        else:
            closest_group.members.append(question)
    return groups
