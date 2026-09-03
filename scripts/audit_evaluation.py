"""Reconcile saved scores to evaluation tables and optionally rerun sensitivity."""

import argparse
from adlift.audit import audit_saved_evaluation, bootstrap_sensitivity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity", action="store_true")
    args = parser.parse_args()
    audit_saved_evaluation()
    if args.sensitivity:
        bootstrap_sensitivity()
