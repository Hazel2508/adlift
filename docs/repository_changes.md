# Repository restructuring record

The September 3 restructuring follows the instructor's final-session requirements:

| Requirement | Implementation |
|---|---|
| Review evaluation before packaging | Numerical audit and sensitivity check retained in `reports/audit/` |
| Scripts execute the complete analysis | `scripts/main.py` and individual preparation, estimation, training and evaluation entry points |
| Reusable functionality lives in an importable module | `src/adlift/` with data, experiments, models, metrics, evaluation and plots |
| Notebooks explain rather than generate the analysis | Three executed walkthroughs under `notebooks/` |
| Modern dependency management | `pyproject.toml` and `uv.lock` |
| README covers overview, installation, execution, exploration and limitations | Root README with links to evidence and detailed rationale |
| Separate results from narrative report | `results/tables`, `results/figures`, and `reports/final_report.md` |
| Remove teaching files and one-off generators | Removed from the publication tree; original local reference copies retained separately |
| Explain meaningful test scope | `docs/reproducibility.md` |
| Review excess complexity and bugs | Removed notebook-generation scripts, obsolete pre-boundary model diagnostics, stale cross-fitting preparation, notebook-only rendering calls and ad hoc module imports |

The internal `phase2`, `phase3`, and `phase4` result identifiers are intentionally retained to preserve the original frozen evaluation lineage. Reader-facing structure and narrative follow the analytical workflow. No raw data, trained models, local editor state, or old OneDrive Git metadata is published.

HTML presentation, GitHub Pages and profile redesign are separate later deliverables and are not part of this repository audit/refactor.
