#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy Smart Endoscope still images and extract sharp multi-view keyframes from videos."
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Directory containing flat jpg/mp4 captures.")
    parser.add_argument("--output-root", type=Path, required=True, help="Prepared dataset root. Images are written to output-root/images.")
    parser.add_argument("--keyframes-per-video", type=int, default=3)
    parser.add_argument("--max-candidates-per-segment", type=int, default=24)
    parser.add_argument("--min-margin-ratio", type=float, default=0.05)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_output_stem(stem: str) -> str:
    return stem.replace(" ", "_")


def _copy_stills(input_root: Path, images_dir: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for image_path in sorted([*input_root.glob("*.jpg"), *input_root.glob("*.jpeg"), *input_root.glob("*.png")]):
        out_stem = _normalize_output_stem(image_path.stem)
        out_path = images_dir / f"{out_stem}.jpg"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        cv2.imwrite(str(out_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        records.append(
            {
                "stem": out_stem,
                "output_path": str(out_path),
                "source_type": "still",
                "source_path": str(image_path),
            }
        )
    return records


def _candidate_indices(frame_count: int, keyframes_per_video: int, min_margin_ratio: float) -> List[List[int]]:
    margin = int(frame_count * min_margin_ratio)
    start = max(0, margin)
    end = max(start + 1, frame_count - margin)
    usable = max(1, end - start)
    segment_len = max(1, usable // max(keyframes_per_video, 1))

    segments: List[List[int]] = []
    for idx in range(keyframes_per_video):
        seg_start = start + idx * segment_len
        seg_end = end if idx == keyframes_per_video - 1 else min(end, seg_start + segment_len)
        if seg_end <= seg_start:
            seg_end = min(frame_count, seg_start + 1)
        segments.append(list(range(seg_start, seg_end)))
    return segments


def _sample_indices(indices: List[int], max_candidates: int) -> List[int]:
    if len(indices) <= max_candidates:
        return indices
    step = len(indices) / float(max_candidates)
    chosen = []
    for i in range(max_candidates):
        pos = min(len(indices) - 1, int(round(i * step)))
        chosen.append(indices[pos])
    return sorted(set(chosen))


def _frame_sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_video_keyframes(
    video_path: Path,
    images_dir: Path,
    keyframes_per_video: int,
    max_candidates_per_segment: int,
    min_margin_ratio: float,
    jpeg_quality: int,
) -> List[Dict[str, object]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_count <= 0:
        capture.release()
        raise RuntimeError(f"Video reports no frames: {video_path}")

    segments = _candidate_indices(frame_count, keyframes_per_video, min_margin_ratio)
    chosen_frames = []
    for seg_idx, segment in enumerate(segments, start=1):
        candidates = _sample_indices(segment, max_candidates_per_segment)
        best = None
        best_score = -math.inf
        for frame_idx in candidates:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            score = _frame_sharpness(frame)
            if score > best_score:
                best_score = score
                best = (frame_idx, frame)
        if best is not None:
            chosen_frames.append((seg_idx, best[0], best[1], best_score))
    capture.release()

    records: List[Dict[str, object]] = []
    base_stem = _normalize_output_stem(video_path.stem)
    for seg_idx, frame_idx, frame, sharpness in chosen_frames:
        stem = f"{base_stem}-kf{seg_idx:02d}"
        out_path = images_dir / f"{stem}.jpg"
        cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        records.append(
            {
                "stem": stem,
                "output_path": str(out_path),
                "source_type": "video_keyframe",
                "source_path": str(video_path),
                "frame_index": int(frame_idx),
                "fps": fps,
                "time_sec": float(frame_idx / fps) if fps > 1e-6 else None,
                "sharpness": float(sharpness),
            }
        )
    return records


def main() -> None:
    args = build_argparser().parse_args()
    input_root = args.input_root
    output_root = args.output_root
    images_dir = ensure_dir(output_root / "images")

    still_records = _copy_stills(input_root, images_dir)
    keyframe_records: List[Dict[str, object]] = []
    for video_path in sorted(input_root.glob("*.mp4")):
        keyframe_records.extend(
            extract_video_keyframes(
                video_path=video_path,
                images_dir=images_dir,
                keyframes_per_video=args.keyframes_per_video,
                max_candidates_per_segment=args.max_candidates_per_segment,
                min_margin_ratio=args.min_margin_ratio,
                jpeg_quality=args.jpeg_quality,
            )
        )

    all_records = still_records + keyframe_records
    stems = sorted(record["stem"] for record in all_records)
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "num_stills_copied": len(still_records),
        "num_videos_processed": len(list(input_root.glob("*.mp4"))),
        "num_keyframes_extracted": len(keyframe_records),
        "num_output_images": len(all_records),
        "keyframes_per_video": int(args.keyframes_per_video),
    }

    (output_root / "all_stems.txt").write_text("\n".join(stems) + "\n", encoding="utf-8")
    (output_root / "extraction_summary.json").write_text(
        json.dumps({"summary": summary, "records": all_records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_path = output_root / "extraction_summary.csv"
    fieldnames = [
        "stem",
        "output_path",
        "source_type",
        "source_path",
        "frame_index",
        "fps",
        "time_sec",
        "sharpness",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in all_records:
            writer.writerow({key: record.get(key) for key in fieldnames})

    print(f"num_stills_copied: {summary['num_stills_copied']}")
    print(f"num_videos_processed: {summary['num_videos_processed']}")
    print(f"num_keyframes_extracted: {summary['num_keyframes_extracted']}")
    print(f"num_output_images: {summary['num_output_images']}")
    print(f"images_dir: {images_dir}")
    print(f"stems_file: {output_root / 'all_stems.txt'}")
    print(f"summary_json: {output_root / 'extraction_summary.json'}")


if __name__ == "__main__":
    main()
