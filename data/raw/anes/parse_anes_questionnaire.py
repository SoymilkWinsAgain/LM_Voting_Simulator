#!/usr/bin/env python3
"""
Parse the ANES 2024 questionnaire PDF into a machine-readable schema.

Usage:
    python parse_anes_questionnaire.py \
        --pdf anes_timeseries_2024_questionnaire_20240808.pdf \
        --out anes_2024_questionnaire_schema.json
"""
import argparse
import json
import re
from pathlib import Path

import fitz  # PyMuPDF


LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
}


def normalize_text(text: str) -> str:
    for old, new in LIGATURES.items():
        text = text.replace(old, new)
    return text


FIELD_MARKERS = {
    "Label", "Survey Question", "Universe", "Logic", "Interviewer", "Instruction",
    "Display Spec", "Web Spec", "Response Order", "Misc Spec",
    "Release Variable(s)", "Randomi-", "Randomization", "zation"
}

SPLIT_FIELD_MARKERS = {
    ("Survey", "Question"): "Survey Question",
    ("Release", "Variable(s)"): "Release Variable(s)",
    ("Response", "Order"): "Response Order",
    ("Display", "Spec"): "Display Spec",
    ("Web", "Spec"): "Web Spec",
    ("Misc", "Spec"): "Misc Spec",
    ("Interviewer", "Instruction"): "Interviewer Instruction",
}

SECTION_HEADERS = {
    "PRE-ELECTION SURVEY QUESTIONNAIRE",
    "POST-ELECTION SURVEY QUESTIONNAIRE",
}

HEADER_RE = re.compile(r"^[A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+$")
LEGACY_HEADER_RE = re.compile(r"^(PRE|POST)\s+([A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+)\s*$")
MODE_RE = re.compile(r"^(CAPI|CASI|Web|CAPI\+Web|CASI\+Web).*")
VAR_RE = re.compile(r"\bV\d{6}[a-z]?\b")


def extract_lines(pdf_path: str):
    doc = fitz.open(pdf_path)
    records = []
    for page_num, page in enumerate(doc, start=1):
        text = normalize_text(page.get_text("text"))
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line == str(page_num):
                continue
            if line in SECTION_HEADERS:
                continue
            records.append((page_num, line))
    return records


def is_item_header(records, idx: int) -> bool:
    if idx + 2 >= len(records):
        return False
    item = records[idx][1]
    return (
        HEADER_RE.match(item) is not None
        and records[idx + 1][1] in {"PRE", "POST"}
        and MODE_RE.match(records[idx + 2][1]) is not None
    )


def combine_split_markers(lines):
    out = []
    i = 0
    while i < len(lines):
        if i + 1 < len(lines):
            pair = (lines[i], lines[i + 1])
            if pair in SPLIT_FIELD_MARKERS:
                out.append(SPLIT_FIELD_MARKERS[pair])
                i += 2
                continue
        out.append(lines[i])
        i += 1
    return out


def previous_release_marker(records, idx: int) -> int:
    for j in range(idx - 1, max(-1, idx - 30), -1):
        line = records[j][1]
        prev = records[j - 1][1] if j - 1 >= 0 else ""
        if line == "Release Variable(s)" or (prev == "Release" and line == "Variable(s)"):
            return j - 1 if prev == "Release" else j
    return idx


def split_legacy_entries(records):
    starts = [i for i, (_, line) in enumerate(records) if LEGACY_HEADER_RE.match(line)]
    entries = []
    for k, header_idx in enumerate(starts):
        release_idx = previous_release_marker(records, header_idx)
        end = previous_release_marker(records, starts[k + 1]) if k + 1 < len(starts) else len(records)
        page = records[header_idx][0]
        header = records[header_idx][1]
        m = LEGACY_HEADER_RE.match(header)
        if not m:
            continue
        label = records[header_idx + 1][1] if header_idx + 1 < len(records) else ""
        lines = [line for _, line in records[release_idx:end]]
        entries.append(
            {
                "item": m.group(2),
                "wave": m.group(1),
                "mode": "unknown",
                "page": page,
                "lines": lines,
                "legacy_label": label,
            }
        )
    return entries


def split_entries(records):
    starts = [i for i in range(len(records) - 2) if is_item_header(records, i)]
    if not starts:
        return split_legacy_entries(records)
    entries = []
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(records)
        page = records[start][0]
        lines = [line for _, line in records[start:end]]
        entries.append(
            {
                "item": lines[0],
                "wave": lines[1],
                "mode": lines[2],
                "page": page,
                "lines": lines,
            }
        )
    return entries


def next_marker_idx(lines, start_idx):
    for j in range(start_idx, len(lines)):
        if lines[j] in FIELD_MARKERS:
            return j
    return len(lines)


def extract_field(lines, field):
    try:
        i = lines.index(field)
    except ValueError:
        return ""
    j = next_marker_idx(lines, i + 1)
    return "\n".join(lines[i + 1 : j]).strip()


def extract_release_vars(lines):
    vars_out = []
    for i, line in enumerate(lines):
        if line == "Release Variable(s)":
            release_lines = []
            for candidate in lines[i + 1 :]:
                if LEGACY_HEADER_RE.match(candidate) or HEADER_RE.match(candidate) or candidate in FIELD_MARKERS:
                    break
                release_lines.append(candidate)
            seg = "\n".join(release_lines)
            vars_out.extend(VAR_RE.findall(seg))
    seen = set()
    deduped = []
    for v in vars_out:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped


def extract_options(question_text):
    lines = question_text.splitlines()
    options = {}
    current_code = None
    current_text = []
    opt_re = re.compile(r"^(\d{1,2})\.\s*(.*)$")
    for raw in lines:
        line = raw.strip()
        m = opt_re.match(line)
        if m:
            if current_code is not None:
                options[current_code] = " ".join(current_text).strip()
            current_code = m.group(1)
            current_text = [m.group(2).strip()]
        elif current_code is not None:
            current_text.append(line)
    if current_code is not None:
        options[current_code] = " ".join(current_text).strip()
    return options


def compact_text(text):
    return re.sub(r"\s+", " ", text).strip()


def parse_pdf(pdf_path: str):
    records = extract_lines(pdf_path)
    entries = split_entries(records)
    items = []
    variables = {}
    for e in entries:
        lines = combine_split_markers(e["lines"])
        label = compact_text(extract_field(lines, "Label") or e.get("legacy_label", ""))
        question_raw = extract_field(lines, "Survey Question")
        question = compact_text(question_raw)
        options = extract_options(question_raw)
        release_vars = extract_release_vars(lines)
        item = {
            "item": e["item"],
            "wave": e["wave"],
            "mode": e["mode"],
            "page": e["page"],
            "label": label,
            "question": question,
            "options": options,
            "release_vars": release_vars,
        }
        items.append(item)
        for var in release_vars:
            variables[var] = item
    return {
        "source_pdf": str(pdf_path),
        "n_items": len(items),
        "n_release_variables": len(variables),
        "items": items,
        "variables": variables,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    schema = parse_pdf(args.pdf)
    Path(args.out).write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Parsed {schema['n_items']} questionnaire items and {schema['n_release_variables']} release variables.")


if __name__ == "__main__":
    main()
