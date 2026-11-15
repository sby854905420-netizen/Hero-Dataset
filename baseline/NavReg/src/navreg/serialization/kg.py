"""Deterministic KG statement text and selectors, using only the supplied record."""
import copy
import json
import re


def _name(identifier, label, names):
    label = label or names.get(identifier)
    return f'{label} ({identifier})' if label and label != identifier else identifier


def _value_text(value, labels):
    kind = value['type']
    if kind == 'entity':
        return _name(value.get('qid') or value['id'], value.get('label'), labels['entities'])
    if kind == 'string':
        return json.dumps(value['text'], ensure_ascii=False)
    if kind == 'novalue':
        return 'no value'
    if kind == 'time':
        match = re.fullmatch(r'([+-]?\d+)-(\d{2})-(\d{2})(?:T.*)?', value['time'])
        if not match:
            raise ValueError('Invalid KG time: ' + value['time'])
        year, month, day = match.groups()
        year = year.lstrip('+')
        precision = value['precision']
        if precision == 9:
            return year
        if precision == 10:
            return f'{year}-{month}'
        if precision == 11:
            return f'{year}-{month}-{day}'
        raise ValueError('Unsupported KG time precision: ' + str(precision))
    if kind == 'quantity':
        text = value['amount']
        unit = value['unit']
        if unit != '1':
            unit_id = unit.rsplit('/', 1)[-1]
            text += ' ' + _name(unit_id, None, labels['entities'])
        return text
    raise ValueError('Unsupported KG value type: ' + str(kind))


def _snak_value(snak, entity_names):
    if snak['snaktype'] == 'novalue':
        return {'type': 'novalue'}
    if snak['snaktype'] != 'value':
        raise ValueError('Unsupported KG qualifier snaktype: ' + str(snak['snaktype']))
    data = snak['datavalue']
    value = data['value']
    if data['type'] == 'wikibase-entityid':
        qid = value['id']
        result = {'type': 'entity', 'id': qid, 'qid': qid}
        if entity_names.get(qid):
            result['label'] = entity_names[qid]
        return result
    if data['type'] == 'time':
        result = {'type': 'time', **copy.deepcopy(value)}
        if 'calendarmodel' in result:
            result['calendar'] = result.pop('calendarmodel')
        return result
    if data['type'] == 'quantity':
        return {'type': 'quantity', **copy.deepcopy(value)}
    if data['type'] == 'string':
        return {'type': 'string', 'text': value}
    raise ValueError('Unsupported KG qualifier datavalue: ' + str(data['type']))


def _qualifiers(target, labels):
    statement = target.get('wikidata_statement', {})
    if 'qualifiers' not in statement:
        qualifiers = copy.deepcopy(target.get('qualifiers', []))
        return qualifiers, {}, list(dict.fromkeys(item['pid'] for item in qualifiers))
    raw = statement['qualifiers']
    order = statement.get('qualifiers-order', list(raw))
    properties = dict(labels['properties'])
    entities = dict(labels['entities'])
    # The normalized copy supplies names only; raw snaks determine every value.
    for item in target.get('qualifiers', []):
        if item.get('label'):
            properties.setdefault(item['pid'], item['label'])
        value = item['value']
        if value['type'] == 'entity' and value.get('label'):
            entities.setdefault(value.get('qid') or value['id'], value['label'])
    qualifiers = []
    for pid in dict.fromkeys([*order, *raw]):
        for snak in raw.get(pid, []):
            if snak['property'] != pid:
                raise ValueError('KG qualifier property conflicts with its PID group')
            item = {'pid': pid, 'value': _snak_value(snak, entities)}
            if properties.get(pid):
                item['label'] = properties[pid]
            qualifiers.append(item)
    return qualifiers, copy.deepcopy(raw), list(order)


def serialize_kg(source_record_id: str, parent: dict, labels: dict | None = None) -> dict:
    """Render each target separately without changing its official parent identity."""
    if not isinstance(source_record_id, str) or not source_record_id:
        raise ValueError('source_record_id must be a nonempty string')
    labels = labels or {}
    names = {'entities': labels.get('entities', {}), 'properties': labels.get('properties', {})}
    targets = parent.get('targets')
    if targets is None:
        targets = [parent]
    if not isinstance(targets, list) or not targets:
        raise ValueError('KG targets must be a nonempty list when present')
    units = []
    for index, target in enumerate(targets):
        guid = target['statement_guid']
        if not isinstance(guid, str) or not guid:
            raise ValueError('KG statement_guid must be a nonempty string')
        subject = _name(target['subject_qid'], target.get('subject_label'), names['entities'])
        relation = _name(target['pid'], target.get('property_label'), names['properties'])
        qualifiers, qualifiers_raw, qualifiers_order = _qualifiers(target, names)
        qualifier_text = '; '.join(
            _name(item['pid'], item.get('label'), names['properties']) + ': '
            + _value_text(item['value'], names) for item in qualifiers) or 'none'
        rank = target.get('rank', 'unspecified')
        text = '\n'.join([
            'Subject: ' + subject,
            'Relation: ' + relation,
            'Value: ' + _value_text(target['value'], names),
            'Qualifiers: ' + qualifier_text,
            'Rank: ' + rank,
        ])
        metadata = copy.deepcopy({key: target[key] for key in (
            'statement_guid', 'subject_qid', 'subject_label', 'pid', 'property_label',
            'value', 'rank', 'rev_id', 'rev_timestamp', 'kb_snapshot') if key in target})
        metadata['qualifiers'] = copy.deepcopy(qualifiers)
        metadata['qualifiers_raw'] = qualifiers_raw
        metadata['qualifiers_order'] = qualifiers_order
        metadata['references'] = copy.deepcopy(target.get(
            'references', target.get('wikidata_statement', {}).get('references', [])))
        units.append({
            'unit_id': source_record_id + '#target=' + str(index),
            'selector': {'unit': 'statement_target', 'target_index': index, 'statement_guid': guid},
            'text': text, 'metadata': metadata,
        })
    return {'source_record_id': source_record_id, 'source_type': 'kb',
            'context_text': 'Type: KG', 'units': units}
