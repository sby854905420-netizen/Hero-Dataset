"""Render complete frozen InfoBoxes without using question or gold annotations."""

from collections import Counter
from copy import deepcopy
from urllib.parse import quote


def _cell_text(cell):
    items = cell.get("text_items", [])
    return "\n".join(items) if any(item.strip() for item in items) else cell.get("text", "")


def _value_text(row, has_field):
    items = row.get("value_items", [])
    if any(item.strip() for item in items):
        return "\n".join(items)
    value = row.get("value", "")
    if value.strip():
        return value
    cells = row.get("cells", [])
    # The first th is the field label, not a second copy of the value.
    if has_field and cells and cells[0].get("tag") == "th":
        cells = cells[1:]
    if not any(_cell_text(cell).strip() for cell in cells):
        return ""
    return " | ".join(_cell_text(cell) for cell in cells)


def serialize_infobox(source_record_id, parent):
    """Return context plus one ordered, addressable unit for every original row.

    Field occurrences count all original rows, including headers and empty rows.
    Original coordinates and field strings are never reconstructed from display
    cells. The caller supplies the official record ID, including any @occ suffix.
    """
    if not isinstance(source_record_id, str) or not source_record_id:
        raise ValueError("source_record_id must be a nonempty official ID")
    context = "\n".join((
        "Type: InfoBox",
        "Page: " + parent.get("page_title", ""),
        "Section: " + " > ".join(parent.get("section_path", [])),
    ))
    units = []
    occurrences = Counter()
    seen_rows = set()
    for position, row in enumerate(parent["rows"]):
        index = row["row_idx"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in seen_rows:
            raise ValueError("row_idx must be a unique nonnegative integer")
        seen_rows.add(index)
        field = row.get("field")
        if field is not None and not isinstance(field, str):
            raise ValueError("field must be an original string or absent")
        has_field = bool(field)
        occurrence = occurrences[field] if has_field else None
        if has_field:
            occurrences[field] += 1
            unit_id = source_record_id + "#field=" + quote(field, safe="") + "&occurrence=" + str(occurrence)
            selector = {"unit": "infobox_field", "field": field,
                        "field_occurrence": occurrence, "row_index": index}
        elif row["row_type"] == "image_caption":
            unit_id = source_record_id + "#row=" + str(position)
            selector = {"unit": "infobox_row", "row_index": position}
        else:
            unit_id = source_record_id + "#context-row=" + str(index)
            selector = None
        value = _value_text(row, has_field)
        if row["row_type"] == "header":
            text = value or field or ""
        elif has_field:
            text = field + ":" + (" " + value if value else "")
        else:
            text = value
        cells = row.get("cells", [])
        links = row.get("links")
        if links is None:
            links = [link for cell in cells for link in cell.get("links", [])]
        metadata = {
            "row_idx": index,
            "row_type": row["row_type"],
            "field": field,
            "field_occurrence": occurrence,
            "value": row.get("value", ""),
            "value_items": deepcopy(row.get("value_items", [])),
            "value_item_boundaries": deepcopy(row.get("value_item_boundaries", [])),
            # Links have one representation; cell text and ordering stay intact.
            "cells": [{key: deepcopy(value) for key, value in cell.items() if key != "links"}
                      for cell in cells],
            "links": deepcopy(links),
        }
        units.append({"unit_id": unit_id, "selector": selector,
                      "text": text, "metadata": metadata})
    return {"source_record_id": source_record_id, "source_type": "info",
            "context_text": context, "units": units}
