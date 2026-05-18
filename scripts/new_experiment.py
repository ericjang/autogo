#!/usr/bin/env python3
"""Create a timestamped experiment folder with the standard AutoGo layout."""

from __future__ import annotations

import argparse
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    if not slug:
        raise ValueError("experiment topic must contain at least one letter or number")
    return slug


def timestamp() -> str:
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H-%M")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def write_file(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def readme_content(exp_name: str, topic: str, commit: str) -> str:
    return f"""
# {topic}

## Goal

Describe the research question this experiment answers.

## Hypothesis

State the expected result before running the experiment.

## Metric

- Primary metric:
- Secondary metrics:

## Setup

- Experiment: `{exp_name}`
- Created from commit: `{commit}`
- Data source:
- Compute target: local / cluster

## Run

```bash
uv run experiments/{exp_name}/train.py
uv run experiments/{exp_name}/analyze.py
```

## Files

- `train.py` - experiment runner or training entrypoint
- `analyze.py` - reads saved results and generates figures/report updates
- `results.tsv` - machine-readable run ledger
- `report.md` - final findings
- `figures/` - generated plots
"""


def train_content(exp_name: str) -> str:
    return f'''
"""Training or experiment entrypoint for {exp_name}.

Replace this scaffold with the smallest controlled experiment that tests the
hypothesis in README.md. Keep outputs machine-readable so analyze.py can
summarize them.
"""

from __future__ import annotations

import json
from pathlib import Path


EXP_NAME = "{exp_name}"
EXP_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EXP_DIR / "results.tsv"


def main() -> None:
    result = {{
        "run_id": "run-001",
        "metric_name": "todo",
        "metric_value": "nan",
        "status": "todo",
        "notes": "replace train.py scaffold with real experiment",
    }}
    print("===RESULT===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
'''


def analyze_content(exp_name: str) -> str:
    return f'''
"""Analyze saved results for {exp_name}.

Analysis should read from disk only. Do not depend on live jobs or notebook
state. Write figures into figures/ and summarize findings in report.md.
"""

from __future__ import annotations

import csv
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EXP_DIR / "results.tsv"
REPORT_PATH = EXP_DIR / "report.md"
FIGURES_DIR = EXP_DIR / "figures"


def load_results() -> list[dict[str, str]]:
    if not RESULTS_PATH.exists():
        return []
    with RESULTS_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\\t"))


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    rows = load_results()
    completed = [r for r in rows if r.get("status") not in {{"todo", ""}}]

    summary = [
        f"# {{EXP_DIR.name}} report",
        "",
        "## Results",
        "",
        f"- Logged runs: {{len(rows)}}",
        f"- Completed runs: {{len(completed)}}",
        "",
        "## Findings",
        "",
        "- TODO: replace this with the experiment interpretation.",
        "",
        "## Next Steps",
        "",
        "- TODO: decide what to keep, discard, or test next.",
    ]
    REPORT_PATH.write_text("\\n".join(summary) + "\\n", encoding="utf-8")
    print(f"Wrote {{REPORT_PATH}}")


if __name__ == "__main__":
    main()
'''


def results_content(commit: str) -> str:
    return (
        "run_id\tgit_commit\thypothesis\tcommand\tmetric_name\tmetric_value\t"
        "status\tnotes\n"
        f"run-001\t{commit}\tTODO\tTODO\tTODO\tTODO\ttodo\tcreated from template\n"
    )


def report_content(topic: str) -> str:
    return f"""
# {topic} Report

## Goal

TODO

## Setup

TODO

## Results

TODO

## Key Findings

- TODO

## Next Steps

- TODO
"""


def create_experiment(topic: str, *, date_prefix: str | None, dry_run: bool) -> Path:
    exp_name = f"{date_prefix or timestamp()}-{slugify(topic)}"
    exp_dir = EXPERIMENTS_DIR / exp_name
    commit = git_commit()

    paths = [
        exp_dir,
        exp_dir / "figures",
        exp_dir / "README.md",
        exp_dir / "train.py",
        exp_dir / "analyze.py",
        exp_dir / "results.tsv",
        exp_dir / "report.md",
    ]

    if dry_run:
        print(f"Would create experiment: {exp_dir}")
        for path in paths[1:]:
            print(path.relative_to(ROOT))
        return exp_dir

    if exp_dir.exists():
        raise FileExistsError(f"{exp_dir} already exists")

    (exp_dir / "figures").mkdir(parents=True)
    write_file(exp_dir / "README.md", readme_content(exp_name, topic, commit))
    write_file(exp_dir / "train.py", train_content(exp_name))
    write_file(exp_dir / "analyze.py", analyze_content(exp_name))
    write_file(exp_dir / "results.tsv", results_content(commit))
    write_file(exp_dir / "report.md", report_content(topic))
    return exp_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a standard timestamped AutoGo experiment folder."
    )
    parser.add_argument("topic", help="short experiment topic, used in README and slug")
    parser.add_argument(
        "--date-prefix",
        help="override timestamp prefix, e.g. 2026-04-28_00-38",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the files that would be created without writing anything",
    )
    args = parser.parse_args()

    exp_dir = create_experiment(args.topic, date_prefix=args.date_prefix, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"Created {exp_dir.relative_to(ROOT)}")
        print(f"Next: edit {exp_dir.relative_to(ROOT) / 'README.md'}")


if __name__ == "__main__":
    main()
