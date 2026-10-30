# SnakeWatch Live — modified dashboard package

## Included
- `snakewatch_live_app.py`: simplified Operator view plus Advanced / analysis view.
- `evaluate_sensitivity_presets.py`: threshold sweep evaluation for YOLO-format image/label datasets.
- `requirements_live.txt`: runtime dependencies (pin `supervision==0.25.1` for the ByteTrack API used here).
- `RUN_LIVE_APP_COLAB.md`: Colab setup and run instructions.

## Important research caveats
- The dashboard's `0.055` threshold is labelled as the validation-selected setting based on the project notes. `0.03` and `0.08` are candidate settings and must be measured; they are not validated preset performance claims.
- The app displays detector score bands, not calibrated probabilities. The `Higher detector score` / `Lower detector score` labels do not assert that the object is or is not a snake.
- Current app states are `Tentative` and `Tracked`. A track can be wrong; this live app does not yet connect the full zone/risk/alert stages, so the panel is explicitly provisional.
- The offline replay `app.py` is not overwritten. Keep its saved JSON/video results for demo fallback if the Colab GPU session or tunnel is unavailable.
- The evaluation script assumes one-class YOLO normalized labels, where class `0` is snake. Verify the dataset paths and annotation convention before running. It reports object-level precision/recall/F1 and false positives per image; optional group CSV enables category-specific summaries.
- Candidate presets use detector thresholds 0.03, 0.055, and 0.08. Choose settings using validation data; use the held-out test set only for final reporting.
