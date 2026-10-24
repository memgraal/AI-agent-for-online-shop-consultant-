from types import SimpleNamespace

from faq_workbench.clustering import cluster_questions


def question(identifier, text, frequency, embedding):
    return SimpleNamespace(
        id=identifier,
        question_text=text,
        occurrence_count=frequency,
        embedding=embedding,
    )


def test_groups_similar_questions_and_sums_frequency():
    first = question("1", "Как оплатить заказ?", 4, [1.0, 0.0])
    similar = question("2", "Какие способы оплаты?", 3, [0.99, 0.1])
    unrelated = question("3", "Как вернуть товар?", 2, [0.0, 1.0])

    groups = cluster_questions([first, similar, unrelated], similarity_threshold=0.9)

    assert len(groups) == 2
    assert groups[0].representative is first
    assert groups[0].members == [first, similar]
    assert groups[0].frequency == 7
    assert groups[0].analysis_input() == {
        "text": "Как оплатить заказ?",
        "frequency": 7,
        "variants": ["Какие способы оплаты?"],
    }


def test_skips_missing_embeddings_and_does_not_chain_through_members():
    first = question("1", "first", 1, [1.0, 0.0])
    bridge = question("2", "bridge", 1, [0.8, 0.6])
    chain = question("3", "chain", 1, [0.6, 0.8])
    missing = question("4", "missing", 1, None)

    groups = cluster_questions([first, bridge, chain, missing], similarity_threshold=0.75)

    assert [group.members for group in groups] == [[first, bridge], [chain]]
