"""Render official table records as Markdown without changing source coordinates."""
import copy
import html
import re


def _markdown(text):
    if not isinstance(text, str):
        raise ValueError('table text must be a string')
    text = html.escape(text, quote=False).replace('\\', '\\\\').replace('|', '\\|')
    return text.replace('\r\n', '\n').replace('\r', '\n').replace('\n', '<br>')


def _cell_text(cell):
    if not isinstance(cell, dict) or not isinstance(cell.get('text'), str):
        raise ValueError('each table cell requires a text string')
    raw = cell['text']
    items = cell.get('text_items', [])
    if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
        raise ValueError('text_items must be an array of strings')
    if not items:
        return _markdown(raw)
    rendered = '<br>'.join(_markdown(item) for item in items)
    # Keep the full source text if a future record's list extraction loses content.
    if re.sub(r'\s+', '', ''.join(items)) != re.sub(r'\s+', '', raw):
        rendered += '<br>raw_text=' + _markdown(raw)
    return rendered


def serialize_table(source_record_id, parent):
    """Return one context and one unit per original row; metadata preserves cells."""
    if not isinstance(source_record_id, str) or not source_record_id:
        raise ValueError('source_record_id must be a nonempty official ID')
    if not isinstance(parent, dict):
        raise ValueError('table parent must be an object')
    header, rows = parent.get('header'), parent.get('rows')
    if not isinstance(header, list) or any(not isinstance(layer, list) for layer in header):
        raise ValueError('header must be an array of header rows')
    if not isinstance(rows, list) or any(not isinstance(row, list) for row in rows):
        raise ValueError('data rows must be arrays')
    width = len(header[0]) if header else (len(rows[0]) if rows else 0)
    if not width or any(len(layer) != width for layer in header):
        raise ValueError('header rows must have the same nonzero width')
    if any(not isinstance(value, str) for layer in header for value in layer):
        raise ValueError('header values must be strings')
    if not isinstance(rows, list) or any(not isinstance(row, list) or len(row) != width for row in rows):
        raise ValueError('data rows must retain the complete header width')
    sections = parent.get('section_path', [])
    if not isinstance(sections, list) or any(not isinstance(section, str) for section in sections):
        raise ValueError('section_path must be an array of strings')

    labels = ['source_row_index (locator only)']
    for column in range(width):
        path = [layer[column] for layer in header if layer[column].strip()]
        label = f'C{column}'
        if path:
            label += ': ' + ' > '.join(_markdown(part) for part in path)
        labels.append(label)
    context = ['Type: Table', 'Page: ' + _markdown(parent.get('page_title', '')),
               'Section: ' + ' > '.join(_markdown(section) for section in sections)]
    if not header:
        context.append('Headers: not supplied in source; Cn labels are column coordinates only')
    if 'caption' in parent:
        context.append('Caption: ' + _markdown(parent['caption']))
    context.extend(['', '| ' + ' | '.join(labels) + ' |',
                    '| ' + ' | '.join(['---'] * len(labels)) + ' |'])

    units = []
    for row_index, row in enumerate(rows):
        metadata = {'row_index': row_index, 'column_indices': list(range(width)),
                    'header': copy.deepcopy(header), 'cells': copy.deepcopy(row)}
        for key in ('header_text_items', 'header_item_boundaries', 'header_links'):
            if key in parent:
                metadata[key] = copy.deepcopy(parent[key])
        units.append({
            'unit_id': f'{source_record_id}#row={row_index}',
            'selector': {'unit': 'table_row', 'row_index': row_index, 'whole_table': False},
            'text': '| ' + ' | '.join([str(row_index)] + [_cell_text(cell) for cell in row]) + ' |',
            'metadata': metadata,
        })
    return {'source_record_id': source_record_id, 'source_type': 'table',
            'context_text': '\n'.join(context), 'units': units}
