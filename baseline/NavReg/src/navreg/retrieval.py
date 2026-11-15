"""Top-K chunks select context; parent@100 remains an independent evaluation rank."""


def retrieve(question, embedder, index, chunk_k):
    hits = index.search(embedder.encode([question], query=True))
    selected, ranking, seen = [], [], set()
    for position, (chunk, score) in enumerate(hits):
        rid = chunk['source_record_id']
        if position < chunk_k and rid not in selected:
            selected.append(rid)
        if rid not in seen:
            seen.add(rid)
            if len(ranking) < 100:
                ranking.append(rid)
    return {'context_record_ids': selected, 'retrieved_record_ids': ranking,
            'top_chunks': [{'chunk_id': chunk['chunk_id'], 'source_record_id': chunk['source_record_id'], 'score': score}
                           for chunk, score in hits[:chunk_k]], 'search_scope': 'all_chunks_exact'}
