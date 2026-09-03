"""Train uplift learners using train/validation data, freeze and score test features."""

import argparse
from adlift import models
from adlift.paths import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome", choices=["visit", "conversion", "both"], default="both")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fit a small train/validation subset without freezing or test scoring.",
    )
    args = parser.parse_args()
    config = models.read_json(PROJECT_ROOT / "config/models.json")
    outcomes = config["outcomes"] if args.outcome == "both" else [args.outcome]
    models.run_development(
        config, outcomes, "smoke" if args.smoke else "full", freeze=not args.smoke
    )
    if not args.smoke:
        models.run_test_scoring(config, outcomes)


if __name__ == "__main__":
    main()
