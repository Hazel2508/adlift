"""Run AdLift's raw-data-to-results pipeline without opening a notebook."""

import argparse
from adlift import data, experiments, models, evaluation
from adlift.paths import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["all", "data", "experiment", "models", "evaluation"], default="all"
    )
    args = parser.parse_args()
    if args.stage in ["all", "data"]:
        data.prepare_dataset()
    if args.stage in ["all", "experiment"]:
        experiments.run_incrementality()
    if args.stage in ["all", "models"]:
        config = models.read_json(PROJECT_ROOT / "config/models.json")
        models.run_development(config, config["outcomes"], "full", freeze=True)
        models.run_test_scoring(config, config["outcomes"])
    if args.stage in ["all", "evaluation"]:
        config = evaluation.read_json(PROJECT_ROOT / "config/evaluation.json")
        evaluation.prepare(config, config["outcomes"])
        evaluation.evaluate(config, config["outcomes"])


if __name__ == "__main__":
    main()
