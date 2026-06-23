from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


DEMO_REPORT: dict[str, Any] = {
    "summary": {
        "cases": 75,
        "answerable_cases": 65,
        "unanswerable_cases": 10,
        "recall_at_k": 0.9538,
        "mrr": 0.8651,
        "ndcg_at_k": 0.8860,
        "faithfulness": 0.8267,
        "abstain_rate": 0.0,
        "false_answer_rate": 1.0,
    },
    "root_causes": {
        "passed": 62,
        "retrieval_failure": 3,
        "unanswerable_answered": 10,
    },
    "failing_cases": [
        {
            "query": "日语动词连接形规则",
            "root_cause": "retrieval_failure",
            "suggested_action": "improve_index_chunking_embedding_or_query_rewrite",
        },
        {
            "query": "被别人做了某事怎么表达",
            "root_cause": "retrieval_failure",
            "suggested_action": "improve_index_chunking_embedding_or_query_rewrite",
        },
        {
            "query": "日本的人口大概是多少",
            "root_cause": "unanswerable_answered",
            "suggested_action": "tighten_abstention_guardrail",
        },
    ],
}


def _metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def demo_report() -> dict[str, Any]:
    return DEMO_REPORT


def render_index() -> str:
    summary = DEMO_REPORT["summary"]
    root_causes = DEMO_REPORT["root_causes"]
    failing_cases = DEMO_REPORT["failing_cases"]
    metric_cards = [
        ("Recall@K", summary["recall_at_k"], "Correct evidence found"),
        ("MRR", summary["mrr"], "Correct evidence rank quality"),
        ("Faithfulness", summary["faithfulness"], "Answer grounded in context"),
        ("False Answer", summary["false_answer_rate"], "Out-of-domain guardrail gap"),
    ]
    metric_markup = "\n".join(
        f"""
        <article class="metric-card">
          <span>{label}</span>
          <strong>{_metric(value)}</strong>
          <small>{caption}</small>
        </article>
        """
        for label, value, caption in metric_cards
    )
    root_markup = "\n".join(
        f"""
        <div class="root-row">
          <span>{name}</span>
          <strong>{count}</strong>
        </div>
        """
        for name, count in root_causes.items()
    )
    case_markup = "\n".join(
        f"""
        <article class="case-card">
          <p>{case["query"]}</p>
          <div>
            <code>{case["root_cause"]}</code>
            <span>{case["suggested_action"]}</span>
          </div>
        </article>
        """
        for case in failing_cases
    )
    command = """arklab eval-jvm \\
  --base-url http://localhost:3456 \\
  --mode search \\
  --eval-set benchmarks/japanese_verb_master/golden_eval.jsonl \\
  --output data/reports/japanese-verb-master-search.json"""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ArkLab</title>
  <meta name="description" content="ArkLab is a local-first RAG evaluation and diagnosis workbench." />
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f8fafc;
      --panel: #ffffff;
      --panel-soft: #f1f5f9;
      --text: #0f172a;
      --muted: #64748b;
      --border: #dbe3ef;
      --accent: #2563eb;
      --accent-soft: #dbeafe;
      --danger: #b91c1c;
      --shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #0a0f1a;
        --panel: #111827;
        --panel-soft: #0f172a;
        --text: #e5e7eb;
        --muted: #94a3b8;
        --border: #1f2937;
        --accent: #60a5fa;
        --accent-soft: rgba(96, 165, 250, 0.16);
        --danger: #fca5a5;
        --shadow: none;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      letter-spacing: 0;
    }}
    main {{
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 56px 0;
    }}
    .hero {{
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(340px, 0.95fr);
      gap: 28px;
      align-items: stretch;
    }}
    .hero-copy {{
      padding: 28px 0;
    }}
    .eyebrow {{
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    h1 {{
      margin: 14px 0 18px;
      font-size: clamp(42px, 8vw, 86px);
      line-height: 0.93;
      letter-spacing: 0;
      max-width: 680px;
    }}
    .lead {{
      max-width: 660px;
      color: var(--muted);
      font-size: 18px;
      line-height: 1.65;
    }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 28px;
    }}
    a.button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 42px;
      padding: 0 16px;
      border-radius: 8px;
      text-decoration: none;
      font-weight: 700;
      border: 1px solid var(--border);
      color: var(--text);
      background: var(--panel);
    }}
    a.button.primary {{
      color: white;
      border-color: var(--accent);
      background: #2563eb;
    }}
    .dashboard {{
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .dashboard-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid var(--border);
      padding: 18px 20px;
    }}
    .dashboard-head strong {{ font-size: 15px; }}
    .status-pill {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 700;
    }}
    .status-pill::before {{
      content: "";
      width: 7px;
      height: 7px;
      border-radius: 999px;
      background: currentColor;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 1px;
      background: var(--border);
    }}
    .metric-card {{
      min-height: 132px;
      background: var(--panel);
      padding: 18px;
    }}
    .metric-card span, .metric-card small {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    .metric-card strong {{
      display: block;
      margin: 10px 0 8px;
      font-size: 32px;
      line-height: 1;
      letter-spacing: 0;
    }}
    .section {{
      margin-top: 34px;
      padding-top: 28px;
      border-top: 1px solid var(--border);
    }}
    h2 {{
      margin: 0 0 16px;
      font-size: 24px;
      letter-spacing: 0;
    }}
    .two-col {{
      display: grid;
      grid-template-columns: minmax(0, 0.85fr) minmax(0, 1.15fr);
      gap: 20px;
    }}
    .panel {{
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel);
      padding: 18px;
    }}
    .root-row {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid var(--border);
      padding: 11px 0;
      color: var(--muted);
      font-size: 14px;
    }}
    .root-row:last-child {{ border-bottom: 0; }}
    .root-row strong {{ color: var(--text); }}
    .case-list {{
      display: grid;
      gap: 10px;
    }}
    .case-card {{
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px;
    }}
    .case-card p {{
      margin: 0 0 10px;
      font-size: 14px;
      color: var(--text);
    }}
    .case-card div {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}
    code {{
      border-radius: 6px;
      background: var(--panel-soft);
      color: var(--danger);
      padding: 3px 6px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
    }}
    pre {{
      overflow-x: auto;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel-soft);
      padding: 16px;
      color: var(--text);
      font-size: 13px;
      line-height: 1.6;
    }}
    .note {{
      color: var(--muted);
      font-size: 14px;
      line-height: 1.65;
    }}
    @media (max-width: 860px) {{
      main {{ padding: 32px 0; }}
      .hero, .two-col {{ grid-template-columns: 1fr; }}
      .hero-copy {{ padding: 0; }}
    }}
    @media (max-width: 520px) {{
      .metric-grid {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 46px; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="hero-copy">
        <div class="eyebrow">RAG evaluation orchestration</div>
        <h1>ArkLab</h1>
        <p class="lead">
          A local-first workbench for turning RAG failures into reproducible experiments:
          retrieval metrics, answer grounding, root-cause diagnosis, trace files, and regression sets.
        </p>
        <div class="actions">
          <a class="button primary" href="https://github.com/yuaiccc/ArkLab" target="_blank" rel="noreferrer">GitHub</a>
          <a class="button" href="/api/demo-report">Demo JSON</a>
        </div>
      </div>
      <aside class="dashboard" aria-label="Demo experiment dashboard">
        <div class="dashboard-head">
          <strong>japanese-verb-master search eval</strong>
          <span class="status-pill">75 cases</span>
        </div>
        <div class="metric-grid">{metric_markup}</div>
      </aside>
    </section>

    <section class="section two-col">
      <div class="panel">
        <h2>Root Causes</h2>
        {root_markup}
      </div>
      <div>
        <h2>Failure Drilldown</h2>
        <div class="case-list">{case_markup}</div>
      </div>
    </section>

    <section class="section">
      <h2>Run Locally</h2>
      <p class="note">
        The hosted UI is intentionally read-only. Real evaluations stay in your local CLI so private docs,
        traces, API keys, and benchmark data do not leave your machine.
      </p>
      <pre><code>{command}</code></pre>
    </section>
  </main>
</body>
</html>"""


class ArkLabHandler(BaseHTTPRequestHandler):
    server_version = "ArkLabWeb/0.1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/index.html"}:
            self._send_text(render_index(), content_type="text/html; charset=utf-8")
            return
        if self.path == "/api/demo-report":
            self._send_json(demo_report())
            return
        if self.path == "/healthz":
            self._send_json({"ok": True})
            return
        self._send_text("Not Found", status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_text(
        self,
        text: str,
        *,
        status: HTTPStatus = HTTPStatus.OK,
        content_type: str = "text/plain; charset=utf-8",
    ) -> None:
        body = text.encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run(host: str = "0.0.0.0", port: int | None = None) -> None:
    selected_port = port or int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer((host, selected_port), ArkLabHandler)
    print(f"ArkLab web listening on http://{host}:{selected_port}", flush=True)
    server.serve_forever()


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

