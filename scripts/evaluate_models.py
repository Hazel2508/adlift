"""Prepare baselines or evaluate frozen test predictions; do not retune uplift models."""

import argparse
from adlift import evaluation
from adlift.artifacts import read_json
from adlift.paths import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["prepare", "evaluate", "all"], nargs="?", default="evaluate"
    )
    parser.add_argument("--outcome", choices=["visit", "conversion", "both"], default="both")
    args = parser.parse_args()
    config = read_json(PROJECT_ROOT / "config/evaluation.json")
    outcomes = config["outcomes"] if args.outcome == "both" else [args.outcome]
    if args.stage in ["prepare", "all"]:
        evaluation.prepare(config, outcomes)
    if args.stage in ["evaluate", "all"]:
        evaluation.evaluate(config, outcomes)


if __name__ == "__main__":
    main()
