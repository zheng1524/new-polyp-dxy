"""Frozen raw candidate indices, before C_joint image-quality selection."""
from typing import List

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
