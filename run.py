"""Entrypoint for running the feature-agent graph."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.agent import build_agent_graph, build_initial_state


DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def build_runtime_metadata() -> dict[str, object]:
    return {
        "data_dir": _env_str("DATA_DIR", str(DATA_DIR)),
        "output_dir": _env_str("OUTPUT_DIR", str(OUTPUT_DIR)),
        "evaluator_sample_rows": _env_int("EVALUATOR_SAMPLE_ROWS", 50000),
        "evaluator_iterations": _env_int("EVALUATOR_ITERATIONS", 300),
        "evaluator_thread_count": _env_int("EVALUATOR_THREAD_COUNT", 4),
        "random_state": _env_int("RANDOM_STATE", 42),
    }


def save_graph_diagram(app: object, output_dir: Path) -> None:
    graph_dir = output_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    graph = app.get_graph()
    mermaid = graph.draw_mermaid()

    (graph_dir / "graph_latest.mmd").write_text(mermaid, encoding="utf-8")

    try:
        png = graph.draw_mermaid_png()
        (graph_dir / "graph_latest.png").write_bytes(png)
    except Exception:
        pass


def main() -> None:
    load_dotenv()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    app = build_agent_graph()
    state = build_initial_state(
        max_retries=_env_int("MAX_RETRIES", 2),
        max_iterations=_env_int("MAX_ITERATIONS", 15),
    )
    state["metadata"] = build_runtime_metadata()
    save_graph_diagram(app, Path(str(state["metadata"].get("output_dir", OUTPUT_DIR))))

    result = app.invoke(state)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  dataset:           {result.get('metadata', {}).get('data_dir')}")
    print(f"  status:            {result.get('status')}")
    print(f"  iterations:        {result.get('iteration')}/{result.get('max_iterations')}")
    print(f"  base_score:        {result.get('base_score', 0):.4f}")
    print(f"  final_score:       {result.get('current_score', 0):.4f}")
    print(f"  improvement:       {result.get('current_score', 0) - result.get('base_score', 0):+.4f}")
    print(f"  selected_features: {result.get('selected_features')}")
    print(f"  accepted_count:    {len(result.get('accepted_features', []))}")
    print(f"  rejected_count:    {len(result.get('rejected_features', []))}")
    print(f"  output_train:      {result.get('metadata', {}).get('output_train_path')}")
    print(f"  output_test:       {result.get('metadata', {}).get('output_test_path')}")
    if result.get("execution_error"):
        print(f"  last_error:        {result['execution_error'][:300]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
