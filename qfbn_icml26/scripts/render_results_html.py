import csv
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INP  = ROOT / "artifacts" / "results.csv"
OUT  = ROOT / "artifacts" / "results.html"

def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

def main():
    if not INP.exists():
        raise FileNotFoundError(f"missing: {INP}")

    with INP.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = reader.fieldnames or []

    # build html
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # prepare header with data-type hints
    col_types = {}
    for c in cols:
        # infer numeric if most non-empty values are numeric
        vals = [r.get(c, "") for r in rows]
        non_empty = [v for v in vals if v not in (None, "", "nan", "NaN")]
        if not non_empty:
            col_types[c] = "str"
            continue
        numeric_cnt = sum(1 for v in non_empty[:200] if is_number(str(v)))
        col_types[c] = "num" if numeric_cnt >= max(1, int(0.8 * min(200, len(non_empty)))) else "str"

    def cell(v):
        if v is None:
            v = ""
        s = str(v)
        return html.escape(s)

    # make run_dir clickable if looks like a path
    def maybe_link(col, v):
        s = "" if v is None else str(v)
        if col == "run_dir" and s.strip():
            href = s.strip()
            # windows local path: make it file:///
            if "://" not in href:
                href = "file:///" + href.replace("\\", "/")
            return f'<a href="{html.escape(href)}" target="_blank">{cell(s)}</a>'
        return cell(s)

    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>QFBN Experiments — results.csv ({len(rows)} runs)</title>
<style>
  body {{ font-family: Arial, Helvetica, sans-serif; margin: 16px; }}
  .meta {{ color:#444; margin-bottom: 10px; }}
  input {{ padding: 8px; width: min(900px, 98%); }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 12px; }}
  th {{ position: sticky; top: 0; background: #f7f7f7; cursor: pointer; user-select:none; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .hint {{ font-size: 12px; color:#666; margin-top: 6px; }}
  .right {{ text-align: right; }}
</style>
</head>
<body>
  <h2>QFBN Experiments — Full Results</h2>
  <div class="meta">
    Source: <code>{INP.as_posix()}</code> &nbsp; | &nbsp; Rows: <b>{len(rows)}</b> &nbsp; | &nbsp; Columns: <b>{len(cols)}</b>
  </div>

  <input id="q" placeholder="Filter (supports substring across all columns)..." oninput="filterRows()" />
  <div class="hint">Click a column header to sort. Hold Shift to sort descending on first click.</div>

  <table id="tbl">
    <thead>
      <tr>
        {"".join([f'<th data-type="{col_types[c]}" onclick="sortBy({i}, event)">{html.escape(c)}</th>' for i, c in enumerate(cols)])}
      </tr>
    </thead>
    <tbody>
      {"".join([
        "<tr>" + "".join([
          f'<td class="{ "right" if col_types[c]=="num" else "" }">{maybe_link(c, r.get(c, ""))}</td>'
          for c in cols
        ]) + "</tr>"
        for r in rows
      ])}
    </tbody>
  </table>

<script>
let sortState = {{ col: -1, asc: true }};

function filterRows() {{
  const q = document.getElementById('q').value.toLowerCase();
  const tbody = document.querySelector('#tbl tbody');
  for (const tr of tbody.rows) {{
    const text = tr.innerText.toLowerCase();
    tr.style.display = text.includes(q) ? '' : 'none';
  }}
}}

function sortBy(colIdx, ev) {{
  const table = document.getElementById('tbl');
  const tbody = table.tBodies[0];
  const rows = Array.from(tbody.rows);

  const th = table.tHead.rows[0].cells[colIdx];
  const typ = th.getAttribute('data-type') || 'str';

  let asc;
  if (sortState.col === colIdx) {{
    asc = !sortState.asc;
  }} else {{
    // default: ascending; if user holds shift, start descending
    asc = !(ev && ev.shiftKey);
  }}
  sortState = {{ col: colIdx, asc }};

  const getVal = (tr) => tr.cells[colIdx].innerText.trim();

  rows.sort((a, b) => {{
    const va = getVal(a);
    const vb = getVal(b);
    if (typ === 'num') {{
      const na = parseFloat(va);
      const nb = parseFloat(vb);
      const aOK = Number.isFinite(na);
      const bOK = Number.isFinite(nb);
      if (aOK && bOK) return asc ? (na - nb) : (nb - na);
      if (aOK && !bOK) return -1;
      if (!aOK && bOK) return 1;
      return asc ? va.localeCompare(vb) : vb.localeCompare(va);
    }} else {{
      return asc ? va.localeCompare(vb) : vb.localeCompare(va);
    }}
  }});

  for (const r of rows) tbody.appendChild(r);
}}
</script>
</body>
</html>
"""

    OUT.write_text(html_text, encoding="utf-8")
    print(f"[OK] wrote {OUT}  (open in browser)")

if __name__ == "__main__":
    main()
