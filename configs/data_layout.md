# External data layout

The code accepts paths through environment variables so that no machine-local
path is embedded in source.  The original raw datasets are deliberately not
part of this repository.

`DXY_DATA_ROOT` should contain (or the matching variables should override):

- `models/init_segmentation/`: the preserved INIT multiclass segmentation
  model directory, including its Transformers config/custom code and either
  `model.safetensors` or `pytorch_model.bin`.
- `scale_video_audit/inputs/immutable_scale_manifest.csv`: formal view
  metadata and references used by scale/fusion stages.
- `scale_video_audit/inputs/dense_frame_manifest.csv`: video inventory used
  by `scripts/run_raw_selector.py extract`.
- `raw_selector/`, `area_guard/`, `prior_selector/`: upstream blind CSVs when
  reproducing the frozen base38 post-capture selection exactly.

`DXY_VIDEO_DIR` must contain the source mp4 files for a fresh raw-video or
Ip8 re-acquisition.  `DXY_MODEL_DIR` overrides the model location.

The submitted CSVs in `base38/tables/`, `ip88_extension/tables/`, and
`mainline/tables/` are sufficient to validate the final aggregation and error
ranking without any raw video or model weight.
