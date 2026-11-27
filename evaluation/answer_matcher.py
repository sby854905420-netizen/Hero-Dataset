"""Deterministic answer matching; unresolved semantics require a cached adjudication.

Runtime policies register gold items from QA. This module never invents
aliases, infers entity subclasses, or calls an external judge.
"""

import json
import re
import unicodedata
from datetime import date, time
from decimal import Decimal, InvalidOperation
from fractions import Fraction


def normalize_text(s):
    return " ".join(unicodedata.normalize("NFKC", str(s)).casefold().split())


def _qid(value):
    text = str(value or "").strip()
    return text.upper() if re.fullmatch(r"[Qq][1-9]\d*", text) else None


def _parts(answer):
    if isinstance(answer, dict):
        text = answer.get("text", answer.get("label", ""))
        cid = _qid(answer.get("canonical_id")) or _qid(answer.get("id"))
        if not text:
            text = cid or answer.get("id", "") or ""
    else:
        text = "true" if answer is True else "false" if answer is False else str(answer)
        cid = _qid(text)
    return normalize_text(text), cid or _qid(text)


_NUMBER = r"[+-]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?"


def _decimal(text):
    if "," in text:
        integer = text.lstrip("+-").split(".")[0]
        if not (re.fullmatch(r"\d{1,3}(?:,\d{3})+", integer)
                or re.fullmatch(r"\d{1,2}(?:,\d{2})*,\d{3}", integer)):
            return None
    if not re.fullmatch(_NUMBER, text):
        return None
    try:
        value = Decimal(text.replace(",", ""))
        return value if value.is_finite() else None
    except InvalidOperation:
        return None


def _number_string(value):
    # Decimal.normalize uses the active precision and can round long inputs.
    if value == 0:
        return "0"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


_MONTHS = {name: i for i, name in enumerate(
    ("january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"), 1)}
_MONTHS.update({name[:3]: i for name, i in list(_MONTHS.items())})
_MONTHS["sept"] = 9


def _date(text):
    text = re.sub(r"\b(\d{1,2})(?:st|nd|rd|th)\b", r"\1", text)
    match = re.fullmatch(r"\+?(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?(?:t\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:z|[+-]\d{2}:?\d{2})?)?", text)
    if match:
        if "t" in text:
            try:
                time.fromisoformat(text.split("t", 1)[1].replace("z", "+00:00"))
            except ValueError:
                return None
        year, month, day = (int(v) if v else 0 for v in match.groups())
    else:
        tokens = text.replace(",", " ").split()
        if len(tokens) != 3:
            return None
        try:
            if tokens[0].rstrip(".") in _MONTHS:
                month, day, year = _MONTHS[tokens[0].rstrip(".")], int(tokens[1]), int(tokens[2])
            else:
                day, month, year = int(tokens[0]), _MONTHS[tokens[1].rstrip(".")], int(tokens[2])
        except (KeyError, ValueError):
            return None
    try:
        date(year, month or 1, day or 1)
    except ValueError:
        return None
    if day and not month:
        return None
    precision = "day" if day else "month" if month else "year"
    value = f"{year:04d}"
    if month:
        value += f"-{month:02d}"
    if day:
        value += f"-{day:02d}"
    return value, precision


def _quantity(text, policy):
    qualifier = None
    qualifiers = set(policy.get("qualifiers", {}).values())
    for candidate in qualifiers:
        suffix = " (" + normalize_text(candidate) + ")"
        if text.endswith(suffix):
            qualifier, text = normalize_text(candidate), text[:-len(suffix)].strip()
            break
    if policy["unit"] == "minute":
        pattern = rf"({_NUMBER})\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b"
        matches = list(re.finditer(pattern, text))
        remainder = re.sub(pattern, "", text)
        if not matches or re.sub(r"\band\b", "", remainder).strip():
            return None, qualifier
        value = Decimal(0)
        for match in matches:
            number = _decimal(match[1])
            if number is None:
                return None, qualifier
            unit = match[2]
            factor = Decimal(60) if unit.startswith("h") else Decimal(1) / 60 if unit.startswith("s") else Decimal(1)
            value += number * factor
        return {"amount": _number_string(value), "unit": "minute"}, qualifier
    currency = r"₹|inr|rs\.?|indian rupees?|rupees?|\$|usd|€|eur|£|gbp"
    match = re.fullmatch(rf"(?:(?P<before>{currency})\s*)?(?P<number>{_NUMBER})\s*(?P<scale>crores?|cr|lakhs?|lacs?|million|billion)?\s*(?P<after>{currency})?", text)
    if not match or not (match["before"] or match["after"]):
        return None, qualifier
    symbols = {"$": "USD", "usd": "USD", "€": "EUR", "eur": "EUR", "£": "GBP", "gbp": "GBP"}
    units = [symbols.get(match[k], "INR") for k in ("before", "after") if match[k]]
    number = _decimal(match["number"])
    if number is None or len(set(units)) != 1:
        return None, qualifier
    scale = match["scale"] or ""
    factor = 10000000 if scale.startswith(("crore", "cr")) else 100000 if scale.startswith(("lakh", "lac")) else 1000000 if scale == "million" else 1000000000 if scale == "billion" else 1
    return {"amount": _number_string(number * factor), "unit": units[0]}, qualifier


def normalize_prediction(answer, policy):
    """Return JSON-safe values and a stable key. Safe to normalize this result again."""
    text, cid = _parts(answer)
    kind = policy["type"]
    result = {"text": text, "canonical_id": cid, "type": kind, "value": None,
              "resolved_gold_item_id": None}
    items = policy.get("gold_items", {})
    alias_matches = [gid for gid, item in items.items()
                     if text in {normalize_text(a) for a in item.get("aliases", [item["label"]])}]
    id_matches = [gid for gid, item in items.items() if cid and cid == item.get("canonical_id")]
    value_text = text.rstrip(".")
    if kind in ("entity", "string"):
        if cid and len(alias_matches) == 1:
            alias_id = items[alias_matches[0]].get("canonical_id")
            if alias_id not in (None, cid) or (len(id_matches) == 1 and id_matches != alias_matches):
                result["id_text_conflict"] = True
        resolved = id_matches if cid else alias_matches
        if len(resolved) == 1:
            result["resolved_gold_item_id"] = resolved[0]
        result["value"] = cid or (resolved[0] if len(resolved) == 1 else None)
    elif kind in ("date", "year"):
        parsed = _date(value_text)
        if parsed:
            value, precision = parsed
            result["value"] = value[:4] if kind == "year" else value
            result["precision"] = "year" if kind == "year" else precision
    elif kind == "number":
        for unit in policy.get("count_units", []):
            if value_text.endswith(" " + unit):
                value_text = value_text[:-len(unit)].strip()
                break
        value = _decimal(value_text)
        if value is not None:
            result["value"] = _number_string(value)
    elif kind == "boolean":
        if value_text in ("yes", "true", "no", "false"):
            result["value"] = value_text in ("yes", "true")
    elif kind == "quantity":
        result["value"], result["qualifier"] = _quantity(value_text, policy)
    elif kind == "ratio":
        match = re.fullmatch(rf"(?:w\s*:\s*h\s*=\s*)?({_NUMBER})(?:\s*[:/]\s*({_NUMBER}))?", value_text)
        if match:
            numerator, denominator = _decimal(match[1]), _decimal(match[2] or "1")
            if numerator is not None and denominator is not None and denominator and numerator > 0 and denominator > 0:
                value = Fraction(numerator) / Fraction(denominator)
                result["value"] = f"{value.numerator}/{value.denominator}"
    # An exact QA label or explicit alias remains valid for structured answers too.
    if kind not in ("entity", "string") and not cid and len(alias_matches) == 1:
        result["resolved_gold_item_id"] = alias_matches[0]
    key = [kind, result["value"], result.get("qualifier")]
    if result["resolved_gold_item_id"]:
        key = [kind, "gold", result["resolved_gold_item_id"]]
    if result.get("id_text_conflict") or result["value"] is None:
        key = [kind, text, cid]
    result["key"] = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
    return result


def pair_decision(pred, gold_item, policy):
    """Return equivalent/different/uncertain; never treat an unknown alias as false."""
    p = normalize_prediction(pred, policy)
    gold_id = gold_item.get("gold_item_id")
    registered = policy.get("gold_items", {}).get(gold_id, gold_item)
    g = normalize_prediction({"canonical_id": registered.get("canonical_id"),
                              "text": registered["label"]}, policy)
    if not p["text"] or p.get("id_text_conflict"):
        return "different"
    if p["canonical_id"]:
        if g["canonical_id"]:
            return "equivalent" if p["canonical_id"] == g["canonical_id"] else "different"
        return "different" if p["resolved_gold_item_id"] else "uncertain"
    if p["resolved_gold_item_id"]:
        return "equivalent" if p["resolved_gold_item_id"] == gold_id else "different"
    kind = policy["type"]
    if kind in ("entity", "string"):
        return "equivalent" if p["text"] == g["text"] else "uncertain"
    if p["value"] is None or g["value"] is None:
        return "uncertain"
    if kind == "date":
        ranks = {"year": 1, "month": 2, "day": 3}
        if ranks[p["precision"]] < ranks[policy.get("precision", "day")]:
            return "different"
    if kind in ("number", "quantity"):
        pv, gv = p["value"], g["value"]
        if kind == "quantity":
            if pv["unit"] != gv["unit"]:
                return "different"
            if g.get("qualifier") and not p.get("qualifier"):
                return "uncertain"
            if p.get("qualifier") != g.get("qualifier"):
                return "different"
            pv, gv = pv["amount"], gv["amount"]
        pv, gv = Decimal(pv), Decimal(gv)
        tolerance = max(Decimal(policy.get("absolute_tolerance", "0")),
                        abs(gv) * Decimal(policy.get("relative_tolerance", "0")))
        return "equivalent" if abs(pv - gv) <= tolerance else "different"
    return "equivalent" if p["value"] == g["value"] else "different"
