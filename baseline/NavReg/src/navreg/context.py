"""Restore full serialized parents and declare only visible citation selectors."""
import json


def build_context(record_ids, documents):
    blocks, labels = [], {}
    def declare(rid, selector, text):
        label = 'E' + str(len(labels) + 1)
        labels[label] = {'source_record_id': rid, 'selector': selector}
        return f'[{label}] Citable: true\n' + text
    for rid in record_ids:
        document = documents[rid]
        lines = ['Source record: ' + rid, document['context_text']]
        if document['source_type'] == 'table':
            lines.append(declare(rid, {'unit': 'whole_table', 'whole_table': True}, 'Complete table follows; row labels identify specific rows.'))
        for unit in document['units']:
            citable = unit['selector'] is not None and unit.get('metadata', {}).get('row_type') != 'header'
            lines.append(declare(rid, unit['selector'], unit['text']) if citable else unit['text'])
        blocks.append('\n'.join(lines))
    return '\n\n'.join(blocks), labels


def parse_answer(text, labels):
    answer = json.loads(text)
    if not isinstance(answer, dict) or set(answer) != {'answers', 'citations'}:
        raise ValueError('response must contain exactly answers and citations')
    for name in ('answers', 'citations'):
        items = answer[name]
        if not isinstance(items, list) or any(not isinstance(item, str) or not item.strip() for item in items):
            raise ValueError('answers and citations must be arrays of nonempty strings')
        if len(items) != len(set(items)):
            raise ValueError('duplicate answer or citation')
    if bool(answer['answers']) != bool(answer['citations']):
        raise ValueError('answers and citations must both be empty or both be nonempty')
    if any(label not in labels for label in answer['citations']):
        raise ValueError('citation label is not visible in this question')
    return answer['answers'], [labels[label] for label in answer['citations']]
