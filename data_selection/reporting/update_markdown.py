"""Append a row to a marked table. Existing rows are left in place."""

from __future__ import annotations

from pathlib import Path


def append_row(path: Path, table_id: str, cells: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    marker = f"<!-- TABLE_ID: {table_id} -->"
    if marker not in text:
        raise ValueError(f"missing table marker {table_id}")
    if any("|" in cell or "\n" in cell for cell in cells):
        raise ValueError("cells must not contain table delimiters or newlines")
    row = "| " + " | ".join(cells) + " |"
    version = cells[0]
    start = text.index(marker)
    rest = text[start:]
    if f"|{version}|" in rest or f"| {version} |" in rest:
        return
    lines = text.splitlines()
    insert_at = None
    seen_marker = False
    seen_header = False
    for index, line in enumerate(lines):
        if line.strip() == marker:
            seen_marker = True
            continue
        if seen_marker and line.startswith("|") and not seen_header:
            seen_header = True
            continue
        if seen_header and set(line.replace("|", "").strip()) <= {"-"} and line.startswith("|"):
            insert_at = index + 1
            break
    if insert_at is None:
        raise ValueError(f"table {table_id} has no header")
    # Insert after the separator, before later rows, so new runs stay next to the header
    # only when the table has no body yet. If a body exists, append after the last
    # contiguous table row.
    last = insert_at
    while last < len(lines) and lines[last].startswith("|"):
        last += 1
    lines.insert(last, row)
    path.write_text("\n".join(lines) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")
