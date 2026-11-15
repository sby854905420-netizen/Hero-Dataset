"""Score the six Hero Dataset quality metrics against the final, repaired gold."""
import argparse
import json
import re
import sys
from pathlib import Path

from answer_matcher import normalize_prediction, pair_decision

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_DATA = ROOT / 'Data'
GOLD_VERSION = 'quality-readiness-audit-v1'
VERSION = 'hero-quality-scorer-v4'
METRICS = ('Sufficiency@5', 'Sufficiency@10', 'MRR@100', 'Hit@5', 'Hit@10',
           'Answer Set F1', 'Answer Set EM', 'Grounded EM')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def nonnegative_index(value):
    return type(value) is int and value >= 0


def resolve_citation(citation, records):
    """Validate citation structure using public corpus only, without consulting gold."""
    if not isinstance(citation, dict) or set(citation) != {'source_record_id', 'selector'}:
        raise ValueError('citation requires exactly source_record_id and selector')
    rid = citation['source_record_id']
    if not isinstance(rid, str) or rid not in records:
        raise ValueError('unknown citation source_record_id')
    selector = citation['selector']
    if not isinstance(selector, dict):
        raise ValueError('selector must be an object')
    source, parent = records[rid]
    unit = selector.get('unit')
    if source == 'kb' and unit == 'statement_target':
        if set(selector) - {'unit', 'target_index', 'statement_guid'}:
            raise ValueError('unknown statement selector field')
        index = selector.get('target_index')
        targets = parent.get('targets', [parent])
        if not nonnegative_index(index) or index >= len(targets):
            raise ValueError('target_index outside record')
        guid = targets[index].get('statement_guid', parent.get('statement_guid'))
        if 'statement_guid' in selector and selector['statement_guid'] != guid:
            raise ValueError('statement_guid conflicts with target_index')
        return rid + '#target=' + str(index)
    if source == 'table' and unit == 'table_row':
        if set(selector) - {'unit', 'row_index', 'whole_table'}:
            raise ValueError('unknown table-row selector field')
        index = selector.get('row_index')
        if not nonnegative_index(index) or index >= len(parent['rows']):
            raise ValueError('row_index outside table')
        if 'whole_table' in selector and selector['whole_table'] is not False:
            raise ValueError('table_row cannot select whole_table')
        return rid + '#row=' + str(index)
    if source == 'table' and unit == 'whole_table':
        if set(selector) - {'unit', 'whole_table'} or selector.get('whole_table', True) is not True:
            raise ValueError('invalid whole-table selector')
        return rid + '#table=all'
    if source == 'info' and unit == 'infobox_field':
        if set(selector) - {'unit', 'field', 'field_occurrence', 'row_index'}:
            raise ValueError('unknown InfoBox selector field')
        field, occurrence = selector.get('field'), selector.get('field_occurrence')
        if not isinstance(field, str) or not nonnegative_index(occurrence):
            raise ValueError('InfoBox requires field and field_occurrence')
        rows = [row for row in parent['rows'] if row.get('field') == field]
        if occurrence >= len(rows):
            raise ValueError('InfoBox field occurrence does not exist')
        if 'row_index' in selector and (not nonnegative_index(selector['row_index']) or
                                      selector['row_index'] != rows[occurrence]['row_idx']):
            raise ValueError('row_index conflicts with InfoBox field occurrence')
        # A JSON key avoids ambiguity from punctuation in field names.
        return canonical([rid, field, occurrence])
    raise ValueError('selector unit does not match the source type')


def supports_all(gold, refs):
    covered = set()
    for branch in gold['proof_branches']:
        if all(any(all(ref in refs for ref in option) for option in step['or_options'])
               for step in branch['steps']):
            covered.update(branch['answer_indices'])
    return covered == set(range(len(gold['answer_items'])))


def load_assets(data_dir=DEFAULT_DATA, policies_path=HERE / 'answer_policies.json',
                judge_path=HERE / 'judge_config.json'):
    """Load the self-contained published QA, complete corpus and scoring gold."""
    data_dir = Path(data_dir).resolve()
    annotations_path = data_dir / 'annotations/evaluation_annotations.jsonl'
    qa_path = data_dir / 'qa/qa_pairs.jsonl'
    rows = read_jsonl(annotations_path)
    gold = {row['question_id']: row for row in rows}
    if len(gold) != len(rows) or len(gold) != 500:
        raise ValueError('expected 500 unique gold questions')
    qa_rows = read_jsonl(qa_path)
    samples = {sample['question_id']: sample for sample in qa_rows}
    if len(samples) != len(qa_rows) or any(not isinstance(qid, str) for qid in samples):
        raise ValueError('QA requires unique string question IDs')
    policy_document = read_json(Path(policies_path))
    policies = policy_document['questions']
    if set(policies) != set(gold) or set(samples) != set(gold):
        raise ValueError('answer policies, questions and annotations must cover the same 500 IDs')
    records = {}
    for source, relative, id_key, expected in (
            ('kb', 'data_repository/kg/kb.jsonl', 'record_uid', 747),
            ('table', 'data_repository/tables/tables.jsonl', 'evidence_id', 528),
            ('info', 'data_repository/infoboxes/infoboxes.jsonl', 'evidence_id', 358)):
        path = data_dir / relative
        parents = read_jsonl(path)
        if len(parents) != expected:
            raise ValueError(f'expected {expected} complete {source} source records')
        for parent in parents:
            rid = parent[id_key]
            if not isinstance(rid, str) or rid in records:
                raise ValueError('duplicate or non-string source record ID')
            records[rid] = (source, parent)
    citation_maps = {}
    for qid, row in gold.items():
        if samples[qid]['answers'] != row['answers']:
            raise ValueError('QA and annotations disagree on standard answers: ' + qid)
        mapping, refs = {}, set()
        for evidence in row['evidence']:
            key = resolve_citation({k: evidence[k] for k in ('source_record_id', 'selector')}, records)
            mapping.setdefault(key, set()).add(evidence['ref'])
            refs.add(evidence['ref'])
        citation_maps[qid] = mapping
        item_ids = [answer['gold_item_id'] for answer in row['answer_items']]
        if not item_ids or len(set(item_ids)) != len(item_ids):
            raise ValueError('gold answers require distinct stable item IDs: ' + qid)
        registered = policies[qid].get('gold_items', {})
        if set(registered) != set(item_ids) or policies[qid].get('question') != samples[qid]['question']:
            raise ValueError('answer policy belongs to different gold: ' + qid)
        for item in row['answer_items']:
            entry = registered[item['gold_item_id']]
            if any(entry.get(k) != item.get(k) for k in ('canonical_id', 'label')):
                raise ValueError('answer policy changes a gold answer: ' + qid)
        if set(row['relevant_record_ids']) != {e['source_record_id'] for e in row['evidence']
                                              if e['retrieval_relevant']}:
            raise ValueError('inconsistent retrieval relevance: ' + qid)
        for branch in row['proof_branches']:
            if not branch['answer_indices'] or not branch['steps']:
                raise ValueError('empty gold proof branch: ' + qid)
            if any(type(i) is not int or not 0 <= i < len(item_ids) for i in branch['answer_indices']):
                raise ValueError('gold answer index out of range: ' + qid)
            for step in branch['steps']:
                if not step['or_options'] or any(not option or not set(option) <= refs
                                                for option in step['or_options']):
                    raise ValueError('gold proof references missing evidence: ' + qid)
        if not supports_all(row, refs):
            raise ValueError('gold cannot prove every answer: ' + qid)
    judge = read_json(Path(judge_path))
    prompt_path = HERE / 'judge_prompt.txt'
    context = {
        'evaluator_version': VERSION, 'gold_version': GOLD_VERSION,
        'answer_policy_version': policy_document['version'],
        'judge_prompt': prompt_path.read_text(encoding='utf-8'),
        'judge_config': judge,
    }
    return {'gold': gold, 'samples': samples, 'policies': policies, 'records': records,
            'citation_maps': citation_maps, 'context': context}


def validate_submissions(rows, assets):
    submissions, errors = {}, []
    for line, row in enumerate(rows, start=1):
        try:
            if not isinstance(row, dict) or set(row) - {'question_id', 'status', 'retrieved_record_ids', 'answers', 'citations'}:
                raise ValueError('unknown submission fields or non-object row')
            qid = row.get('question_id')
            if not isinstance(qid, str) or qid not in assets['gold']:
                raise ValueError('unknown question_id; use a string ID')
            if qid in submissions:
                raise ValueError('duplicate question_id')
            submissions[qid] = row
            if row.get('status', 'completed') not in {'completed', 'failed', 'timeout', 'refused'}:
                raise ValueError('unknown status')
            ranking = row.get('retrieved_record_ids', [])
            if not isinstance(ranking, list) or len(ranking) > 100:
                raise ValueError('retrieved_record_ids must be a list of at most 100 IDs')
            if any(not isinstance(r, str) or r not in assets['records'] for r in ranking):
                raise ValueError('ranking contains an unknown source_record_id')
            if len(set(ranking)) != len(ranking):
                raise ValueError('ranking contains duplicate records; ranks were not changed')
            answers = row.get('answers', [])
            if not isinstance(answers, list):
                raise ValueError('answers must be an array of independent answer items')
            for answer in answers:
                if not isinstance(answer, (str, dict)):
                    raise ValueError('answer must be text or an object with a label/canonical ID')
                if isinstance(answer, dict) and (set(answer) - {'label', 'text', 'id', 'canonical_id'} or
                        any(not isinstance(v, str) for v in answer.values())):
                    raise ValueError('answer object has unknown fields or non-string values')
                if isinstance(answer, dict):
                    cid = answer.get('canonical_id')
                    if cid and not re.fullmatch(r'Q[1-9]\d*', cid):
                        raise ValueError('canonical_id must be a Wikidata QID')
                    if cid and re.fullmatch(r'Q[1-9]\d*', answer.get('id', '')) and answer['id'] != cid:
                        raise ValueError('answer has conflicting entity IDs')
                    if 'text' in answer and 'label' in answer and answer['text'] != answer['label']:
                        raise ValueError('answer has conflicting text and label fields')
                normalized = normalize_prediction(answer, assets['policies'][qid])
                if not normalized['text'] and not normalized['canonical_id']:
                    raise ValueError('empty answer item; use an empty answers array for abstention')
            citations = row.get('citations', [])
            if not isinstance(citations, list):
                raise ValueError('citations must be an array')
            for citation in citations:
                resolve_citation(citation, assets['records'])
        except (ValueError, TypeError, KeyError) as exc:
            errors.append({'line': line, 'question_id': row.get('question_id') if isinstance(row, dict) else None,
                           'error': str(exc)})
    return submissions, errors


def maximum_matching(edges):
    matched = {}

    def augment(prediction, seen):
        for answer in edges[prediction]:
            if answer in seen:
                continue
            seen.add(answer)
            if answer not in matched or augment(matched[answer], seen):
                matched[answer] = prediction
                return True
        return False

    for prediction in range(len(edges)):
        augment(prediction, set())
    return sorted((prediction, answer) for answer, prediction in matched.items())


def answer_scores(qid, answers, assets, judgments):
    gold = assets['gold'][qid]['answer_items']
    policy = assets['policies'][qid]
    unique = {}
    for raw in answers:
        normalized = normalize_prediction(raw, policy)
        unique.setdefault(normalized['key'], (raw, normalized))
    edges, pending, seen_singletons = [], [], set()
    for raw, normalized in unique.values():
        decisions = [pair_decision(raw, item, policy) for item in gold]
        confirmed = [i for i, decision in enumerate(decisions) if decision == 'equivalent']
        # Once identity is known, distinct gold items cannot be additional aliases of it.
        if not confirmed:
            unresolved = []
            for i, decision in enumerate(decisions):
                if decision != 'uncertain':
                    continue
                pair = {
                    'question_id': qid, 'question': assets['samples'][qid]['question'],
                    'normalized_prediction': normalized, 'gold_item': gold[i], 'answer_policy': policy,
                }
                key = canonical({'context': assets['context'], 'pair': pair})
                cached = judgments.get(key)
                if cached is None or cached['decision'] == 'uncertain':
                    unresolved.append({'cache_key': key, **pair})
                elif cached['decision'] == 'equivalent':
                    confirmed.append(i)
            if not confirmed:
                pending.extend(unresolved)
        identity = confirmed[0] if len(confirmed) == 1 else None
        if identity is not None and identity in seen_singletons:
            continue
        if identity is not None:
            seen_singletons.add(identity)
        edges.append(confirmed)
    if pending:
        return {'Answer Set F1': None, 'Answer Set EM': None, 'matches': None,
                'prediction_count': None, 'gold_count': len(gold)}, pending
    matches = maximum_matching(edges)
    n, g, m = len(edges), len(gold), len(matches)
    return {'Answer Set F1': 2 * m / (n + g), 'Answer Set EM': int(m == n == g),
            'matches': [{'prediction_index': p, 'gold_item_id': gold[i]['gold_item_id']} for p, i in matches],
            'prediction_count': n, 'gold_count': g}, []


def load_judgments(path, context):
    if path is None:
        return {}
    document = read_json(Path(path))
    if document.get('context') != context:
        raise ValueError('judgments belong to a different gold/matcher/judge version; regenerate the queue')
    judge = context['judge_config']
    if not judge.get('model_snapshot') or not judge.get('judge_version') or judge.get('status') != 'frozen':
        raise ValueError('freeze the judge model snapshot and version before importing judgments')
    result = {}
    for row in document['decisions']:
        if row['decision'] not in {'equivalent', 'different', 'uncertain'} or not row.get('reason'):
            raise ValueError('every judgment needs a decision and reason')
        if row['cache_key'] in result:
            raise ValueError('duplicate judgment cache key')
        result[row['cache_key']] = row
    return result


def evaluate(submissions, assets, judgments=None):
    judgments = judgments or {}
    results, pending = [], []
    for qid, gold in assets['gold'].items():
        row = submissions.get(qid)
        status = row.get('status', 'completed') if row is not None else 'missing'
        result = {'question_id': qid, 'status': status, **{metric: 0 for metric in METRICS}}
        if status == 'completed':
            ranking = row.get('retrieved_record_ids', [])
            positives = set(gold['relevant_record_ids'])
            for k in (5, 10):
                selected_records = set(ranking[:k])
                refs = {e['ref'] for e in gold['evidence'] if e['source_record_id'] in selected_records}
                result['Sufficiency@' + str(k)] = int(supports_all(gold, refs))
                result['Hit@' + str(k)] = int(bool(selected_records & positives))
            result['MRR@100'] = next((1 / i for i, rid in enumerate(ranking[:100], start=1)
                                     if rid in positives), 0)
            qa, unresolved = answer_scores(qid, row.get('answers', []), assets, judgments)
            result.update(qa)
            pending.extend(unresolved)
            selected_keys = {resolve_citation(c, assets['records'])
                             for c in row.get('citations', [])}
            refs = set().union(*(assets['citation_maps'][qid].get(key, set()) for key in selected_keys))
            result['citations_sufficient'] = supports_all(gold, refs)
            result['unconfirmed_citation_keys'] = sorted(selected_keys - set(assets['citation_maps'][qid]))
            em = result['Answer Set EM']
            result['Grounded EM'] = None if em is None else int(em and result['citations_sufficient'])
        results.append(result)
    metrics = {}
    for name in METRICS:
        values = [row[name] for row in results]
        metrics[name] = None if any(value is None for value in values) else 100 * sum(values) / len(results)
    summary = {'status': 'needs_adjudication' if pending else 'scored', 'questions': len(results),
               'received_questions': len(submissions),
               'missing_questions': sum(row['status'] == 'missing' for row in results),
               'unsuccessful_questions': sum(row['status'] in {'failed', 'timeout', 'refused'} for row in results),
               'pending_answer_pairs': len(pending), 'unit': 'percent', 'metrics': metrics,
               'gold_version': assets['context']['gold_version'], 'evaluator_version': VERSION,
               'scope': 'Fixed confirmed gold; unregistered valid evidence requires versioned review.',
               'efficiency': 'Not computed here; record actual baseline request and wall-clock logs separately.'}
    return summary, results, pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--answer-policies', type=Path, default=HERE / 'answer_policies.json')
    parser.add_argument('--judge-config', type=Path, default=HERE / 'judge_config.json')
    parser.add_argument('--judgments', type=Path)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('output directory is nonempty; use a new directory to preserve previous runs')
    try:
        assets = load_assets(args.data_dir, args.answer_policies, args.judge_config)
        rows = read_jsonl(args.predictions)
        submissions, errors = validate_submissions(rows, assets)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if errors:
            write_json(args.output_dir / 'validation_errors.json', errors)
            print('Submission rejected; see validation_errors.json. No ranks were changed.', file=sys.stderr)
            return 2
        judgments = load_judgments(args.judgments, assets['context'])
        summary, results, pending = evaluate(submissions, assets, judgments)
        write_json(args.output_dir / 'run_manifest.json', {**assets['context'],
                   'predictions': str(args.predictions.resolve()),
                   'judgments': str(args.judgments.resolve()) if args.judgments else None})
        if args.judgments:
            write_json(args.output_dir / 'judgments.used.json', read_json(args.judgments))
        write_json(args.output_dir / 'summary.json', summary)
        write_jsonl(args.output_dir / 'per_question.jsonl', results)
        write_jsonl(args.output_dir / 'pending_answer_pairs.jsonl', pending)
        if pending:
            write_json(args.output_dir / 'judgments.template.json', {
                'context': assets['context'],
                'decisions': [{'cache_key': pair['cache_key'], 'decision': 'uncertain',
                               'reason': 'Pending fixed-judge adjudication or official review'} for pair in pending]})
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 3 if pending else 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print('Cannot score: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
