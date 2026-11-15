"""Serialize complete public corpus records without reading QA or gold labels."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import tempfile
from urllib.parse import quote

from .serialization.kg import serialize_kg
from .serialization.table import serialize_table
from .serialization.infobox import serialize_infobox

SERIALIZATION_VERSION = 'navreg-fixed-serialization-v1'
CORPUS_FILES = (
    ('kb', 'data_repository/kg/kb.jsonl', 'record_uid'),
    ('table', 'data_repository/tables/tables.jsonl', 'evidence_id'),
    ('info', 'data_repository/infoboxes/infoboxes.jsonl', 'evidence_id'),
)
RENDERERS = {'kb': serialize_kg, 'table': serialize_table, 'info': serialize_infobox}


def load_sources(data_root):
    """Only these three complete corpus files are inputs; QA is never opened."""
    data_root = Path(data_root).resolve()
    sources, seen = [], set()
    for source, relative, id_field in CORPUS_FILES:
        count = 0
        with (data_root / relative).open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    parent = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{relative}:{line_number}: invalid JSON') from exc
                if not isinstance(parent, dict):
                    raise ValueError(f'{relative}:{line_number}: expected a JSON object')
                rid = parent.get(id_field)
                if not isinstance(rid, str) or not rid:
                    raise ValueError(f'{relative}:{line_number}: missing official {id_field}')
                if rid in seen:
                    raise ValueError('duplicate official source_record_id: ' + rid)
                seen.add(rid)
                record = {'record_id': rid, 'source': source, 'source_path': relative,
                          'line_number': line_number}
                sources.append((record, parent))
                count += 1
        if not count:
            raise ValueError('empty corpus file: ' + relative)
    return sources


def build_labels(sources):
    """Use existing source names; the two fixed additions name measurement units."""
    labels = {'entities': {}, 'properties': {}}

    def remember(bucket, identifier, label):
        pattern = r'Q[1-9]\d*' if bucket == 'entities' else r'P[1-9]\d*'
        if isinstance(identifier, str) and re.fullmatch(pattern, identifier) and isinstance(label, str) and label:
            labels[bucket].setdefault(identifier, label)

    def links(value):
        if isinstance(value, list):
            for child in value:
                links(child)
        elif isinstance(value, dict):
            if value.get('qid') and value.get('title'):
                remember('entities', value['qid'], value['title'])
            for key in ('links', 'cells', 'rows', 'header_links'):
                if key in value:
                    links(value[key])

    # Registry order is stable; KG names take precedence over link titles.
    for record, parent in sources:
        if record['source'] != 'kb':
            continue
        for target in parent.get('targets', [parent]):
            remember('entities', target.get('subject_qid'), target.get('subject_label'))
            remember('properties', target.get('pid'), target.get('property_label'))
            value = target.get('value', {})
            if value.get('type') == 'entity':
                remember('entities', value.get('id') or value.get('qid'), value.get('label'))
    for record, parent in sources:
        if record['source'] != 'kb':
            links(parent)
    labels['entities'].setdefault('Q4917', 'United States dollar')
    labels['entities'].setdefault('Q7727', 'minute')
    return labels

def validate_document(document, record, parent):
    rid, source = record['record_id'], record['source']
    if document.get('source_record_id') != rid or document.get('source_type') != source:
        raise ValueError('renderer changed source identity: ' + rid)
    if not isinstance(document.get('context_text'), str):
        raise ValueError('missing context text: ' + rid)
    originals = parent.get('targets', [parent]) if source == 'kb' else parent['rows']
    units = document['units']
    if len(units) != len(originals) or len({u['unit_id'] for u in units}) != len(units):
        raise ValueError('renderer dropped or merged source units: ' + rid)
    occurrences = Counter()
    for index, (unit, original) in enumerate(zip(units, originals)):
        if not isinstance(unit.get('text'), str) or not isinstance(unit.get('metadata'), dict):
            raise ValueError('invalid unit payload: ' + rid)
        if source == 'kb':
            expected_id = rid + '#target=' + str(index)
            expected_selector = {'unit': 'statement_target', 'target_index': index,
                                 'statement_guid': original.get('statement_guid', parent.get('statement_guid'))}
            raw_qualifiers = original.get('wikidata_statement', {}).get('qualifiers', {})
            if raw_qualifiers:
                metadata = unit['metadata']
                if metadata.get('qualifiers_raw') != raw_qualifiers:
                    raise ValueError('renderer lost original qualifier snaks: ' + expected_id)
                normalized = metadata.get('qualifiers', [])
                if Counter(q['pid'] for q in normalized) != Counter({pid: len(snaks) for pid, snaks in raw_qualifiers.items()}):
                    raise ValueError('renderer dropped qualifier values: ' + expected_id)
                if any(pid not in unit['text'] for pid in raw_qualifiers):
                    raise ValueError('qualifiers missing from retrieval text: ' + expected_id)
            if unit['metadata'].get('value') != original['value']:
                raise ValueError('renderer changed a typed KG value: ' + expected_id)
        elif source == 'table':
            expected_id = rid + '#row=' + str(index)
            expected_selector = {'unit': 'table_row', 'row_index': index, 'whole_table': False}
            if unit['metadata'].get('cells') != original or unit['metadata'].get('header') != parent['header']:
                raise ValueError('renderer lost original table cells/header: ' + expected_id)
        else:
            row_index = original['row_idx']
            if 'field' in original:
                field = original['field']
                occurrence = occurrences[field]
                occurrences[field] += 1
                expected_id = rid + '#field=' + quote(field, safe='') + '&occurrence=' + str(occurrence)
                expected_selector = {'unit': 'infobox_field', 'field': field,
                                     'field_occurrence': occurrence, 'row_index': row_index}
            else:
                expected_id = rid + '#context-row=' + str(row_index)
                expected_selector = None
        if unit['unit_id'] != expected_id or unit.get('selector') != expected_selector:
            raise ValueError('unit changed original coordinates: ' + rid)
    if any(not u['text'].strip() for u in units) and source != 'info':
        raise ValueError('empty KG/table serialization: ' + rid)


def render_source(source, labels):
    record, parent = source
    renderer = RENDERERS[record['source']]
    document = renderer(record['record_id'], parent, labels) if record['source'] == 'kb' else renderer(record['record_id'], parent)
    validate_document(document, record, parent)
    document['serialization_version'] = SERIALIZATION_VERSION
    document['source_provenance'] = {'file': record['source_path'], 'line_number': record['line_number']}
    return document


def serialize_repository(data_root, output_dir, workers=4):
    """Build deterministic record documents in a new directory, without chunking."""
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError('output already exists; choose a new output directory')
    sources = load_sources(data_root)
    labels = build_labels(sources)
    output.parent.mkdir(parents=True, exist_ok=True)
    record_counts, unit_counts = Counter(), Counter()
    with tempfile.TemporaryDirectory(prefix='.' + output.name + '-build-', dir=output.parent) as tmp:
        staging = Path(tmp) / 'result'
        staging.mkdir()
        with ThreadPoolExecutor(max_workers=workers) as executor, (staging / 'records.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
            # map keeps corpus file/line order independent of worker scheduling.
            for document in executor.map(lambda source: render_source(source, labels), sources):
                record_counts[document['source_type']] += 1
                unit_counts[document['source_type']] += len(document['units'])
                stream.write(json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')
        summary = {
            'serialization_version': SERIALIZATION_VERSION,
            'source_records': sum(record_counts.values()),
            'source_records_by_type': dict(sorted(record_counts.items())),
            'units': sum(unit_counts.values()),
            'units_by_type': dict(sorted(unit_counts.items())),
            'gold_used_for_serialization': False,
            'chunking_complete': False,
            'embeddings_computed': False,
            'vector_index_built': False,
        }
        for name, value in [('labels.json', labels), ('summary.json', summary)]:
            (staging / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        staging.rename(output)
    return summary
