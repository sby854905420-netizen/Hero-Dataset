"""Native-tokenizer packing; only retrieval cell values lose their explicit tail."""
from .serialization.table import _cell_text, _markdown
from .serialization.infobox import _value_text


def chunk_documents(documents, tokenizer, max_tokens, cell_limit=100):
    def count(text):
        return len(tokenizer.encode(text, add_special_tokens=True, verbose=False))

    def prefix(value):
        ids = tokenizer.encode(value, add_special_tokens=False, verbose=False)
        if len(ids) <= cell_limit:
            return value, False
        kept = ids[:cell_limit]
        text = tokenizer.decode(kept, skip_special_tokens=False)
        while len(tokenizer.encode(text, add_special_tokens=False, verbose=False)) > cell_limit:
            kept = kept[:-1]
            text = tokenizer.decode(kept, skip_special_tokens=False)
        return text, True

    for document in documents:
        rid, kind = document['source_record_id'], document['source_type']
        number = 0

        def emit(text, coverage):
            nonlocal number
            tokens = count(text)
            if tokens > max_tokens:
                raise ValueError(f'chunk exceeds {max_tokens} tokens: {rid}, {coverage}')
            chunk = {'chunk_id': f'{rid}#chunk={number}', 'source_record_id': rid,
                     'source_type': kind, 'text': text, 'token_count': tokens, 'coverage': coverage}
            number += 1
            return chunk

        context = document['context_text']
        if kind == 'table':
            # Use original structured cells, never split Markdown on escaped pipes.
            heading = context.rsplit('\n\n', 1)[0]
            rows, coverage = [], []
            def header(columns, meta):
                names = ['source_row_index (locator only)']
                for column in columns:
                    parts = [_markdown(layer[column]) for layer in meta['header'] if layer[column].strip()]
                    names.append(f'C{column}' + (': ' + ' > '.join(parts) if parts else ''))
                return heading + '\n\n| ' + ' | '.join(names) + ' |\n| ' + ' | '.join(['---'] * len(names)) + ' |'
            base = None
            for unit in document['units']:
                meta = unit['metadata']
                cells, truncated = [], []
                for i, cell in enumerate(meta['cells']):
                    value, cut = prefix(_cell_text(cell))
                    cells.append(value)
                    if cut:
                        truncated.append(i)
                def part(columns):
                    return '| ' + ' | '.join([str(meta['row_index'])] + [cells[i] for i in columns]) + ' |'
                def cover(columns):
                    return {'unit_id': unit['unit_id'], 'selector': unit['selector'],
                            'column_indices': columns, 'truncated_columns': [i for i in columns if i in truncated]}
                columns = list(range(len(cells)))
                base = header(columns, meta)
                row = part(columns)
                if count(base + '\n' + row) <= max_tokens:
                    if rows and count(base + '\n' + '\n'.join(rows + [row])) > max_tokens:
                        yield emit(base + '\n' + '\n'.join(rows), coverage)
                        rows, coverage = [], []
                    rows.append(row)
                    coverage.append(cover(columns))
                    continue
                if rows:
                    yield emit(base + '\n' + '\n'.join(rows), coverage)
                    rows, coverage = [], []
                group = []
                for column in columns:
                    candidate = group + [column]
                    if count(header(candidate, meta) + '\n' + part(candidate)) <= max_tokens:
                        group = candidate
                    else:
                        if not group:
                            raise ValueError(f'atomic table cell/header exceeds budget: {rid} row {meta["row_index"]} column {column}')
                        yield emit(header(group, meta) + '\n' + part(group), [cover(group)])
                        group = [column]
                        if count(header(group, meta) + '\n' + part(group)) > max_tokens:
                            raise ValueError(f'atomic table cell/header exceeds budget: {rid} row {meta["row_index"]} column {column}')
                if group:
                    yield emit(header(group, meta) + '\n' + part(group), [cover(group)])
            if rows:
                yield emit(base + '\n' + '\n'.join(rows), coverage)
        else:
            bodies, coverage, active_header = [], [], ''
            for unit in document['units']:
                meta = unit.get('metadata', {})
                if kind == 'info' and meta.get('row_type') == 'header':
                    active_header = unit['text']
                    # Delay the context header until there is a data field below it.
                    continue
                text, cut = unit['text'], False
                if kind == 'info':
                    value, cut = prefix(_value_text(meta, bool(meta.get('field'))))
                    text = (meta['field'] + ':' + (' ' + value if value else '')) if meta.get('field') else value
                if not text:
                    continue
                body = (('Internal section: ' + active_header + '\n') if active_header else '') + text
                item = {'unit_id': unit['unit_id'], 'selector': unit['selector'], 'value_truncated': cut}
                candidate = context + '\n\n' + '\n\n'.join(bodies + [body])
                if bodies and count(candidate) > max_tokens:
                    yield emit(context + '\n\n' + '\n\n'.join(bodies), coverage)
                    bodies, coverage = [], []
                if count(context + '\n\n' + body) > max_tokens:
                    raise ValueError(f'atomic unit/context exceeds budget: {rid}, {unit["unit_id"]}')
                bodies.append(body)
                coverage.append(item)
            if bodies:
                yield emit(context + '\n\n' + '\n\n'.join(bodies), coverage)
