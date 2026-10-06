# Published audit-path sanitization

The committed `frozen_protocol.json`, `reacquisition_manifest_blind.csv`, and
`selected_frames_blind.csv` replace machine-local source paths with
`${DXY_VIDEO_DIR}` and `${DXY_OUTPUT_ROOT}` placeholders.  Numerical results,
frame indices, source-video SHA256 values, selection decisions, and the
historical SHA evidence are unchanged.

Consequently, `inputs/blind_sha256.txt` remains a record of the original blind
run rather than a digest of these path-sanitized public copies.  This avoids
publishing workstation layout while retaining the original provenance evidence.
