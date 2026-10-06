# S5-CPAG-39 source-closure audit

Date: 2026-10-06

## Scope and method

This audit started from the five runnable S5-CPAG-39 entry points and traced
static imports, `importlib` usage, `sys.path` manipulation, and shell/external
Python invocations recursively.  Historical result Markdown/CSV files are
retained as provenance but are not executable dependencies.

| Entry point | Included implementation closure |
|---|---|
| `scripts/run_raw_selector.py` | INIT segmentation wrapper; ring processing; V2 geometry; affine scale; sharpness/time-bin selection; B1 aggregation |
| `base38/scripts/run_post_capture_guard.py` | raw candidate tables; C/P area support; post-capture relaxed guard; capture replacement and group fusion |
| `ip88_extension/scripts/run_reacquisition.py` | raw-video decoding; INIT segmentation; ring fit; V2; centerline inner-ring reconstruction; B1 extension result |
| `mainline/scripts/build_mainline.py` | immutable base38 + Ip8.8 assembly; group fusion; metrics |
| `base38/scripts/verify_mainline.py` | SHA/integrity checks for the base38 evidence bundle |

Custom source copied into this repository:

- `src/polypseg/`: model loading helpers, calibration, ring fitting,
  reconstruction dataset utilities, and the frozen single-view measurement
  geometry.
- `src/dxy_s5cpag/`: V2 `process_view` / `choose` / `choose_geometry`, frozen
  legacy measurement helpers, rectified scale, C/P support, sharpness/bin
  selection, B1 aggregation, centerline reconstruction, segmentation wrapper,
  and metrics.
- `model_code/init_segmentation/`: the local UNet3+ configuration/model and
  its recursive `models/` dependency closure.  Weights are intentionally not
  included.

## Path and dynamic-import findings

The scan for the former machine-specific home-directory prefix returned no
matches after packaging.  Runtime paths are read from command-line arguments or the `DXY_*` environment variables in
`configs/runtime.example.env`.

The remaining `sys.path` additions are deliberately repository-local:

- entry scripts insert `<repo>/src` when launched directly;
- `polypseg.common.load_pretrained_model` temporarily imports the configured
  model-code directory.  The required INIT model code is included under
  `model_code/init_segmentation`; this lets a user point `DXY_MODEL_DIR` at a
  separately installed/weight-containing model directory without hard-coding a
  machine path;
- the upstream UNet3+ file temporarily inserts its own directory to import its
  included `models/` package.

No entry point dynamically imports, executes, or shells out to a custom Python
file outside this repository.  External dependencies are data, videos, and
model weights—not untracked custom Python source—and are listed in README.md
and `configs/data_layout.md`.

## Intentional exclusions

No raw endoscopy video, patient-derived frames/masks/caches, trained weights,
conda environment, or credentials are tracked.  The frozen numerical CSVs,
reports, and SHA evidence remain.  Path-valued fields in a small number of
Ip8.8 provenance CSV/JSON files were changed to portable `${DXY_*}` placeholders;
their historical hashes refer to the pre-sanitization private artifact and are
documented in `ip88_extension/SANITIZATION.md`.

## Verification performed

With the project Python and `PYTHONPATH=src`:

1. `python -m unittest discover -s tests -v` — 4/4 passed.
2. Each of the five entry points above accepted `--help`.
3. `mainline/scripts/build_mainline.py --output-root <temporary directory>`
   rebuilt the committed no-video final table and metrics: 39/39, median
   absolute error 4.576786%, MAE 0.613364 mm, p95 19.263320%, max 23.685499%.
4. A post-packaging repository scan found no machine-specific source path and
   no common model-weight files (`*.safetensors`, `*.pth`, `*.pt`, `*.ckpt`).

The raw-selector, post-capture and Ip8.8 reacquisition runs require the
intentionally excluded videos/intermediate data/weights, so only their import
and argument parsing are exercised in this public package.
