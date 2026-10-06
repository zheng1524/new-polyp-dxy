import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CalibrationResult:
    image_width: int
    image_height: int
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    new_camera_matrix: np.ndarray
    roi: Tuple[int, int, int, int]
    reprojection_error: float
    used_images: List[str]
    rejected_images: List[str]
    board_shape: Tuple[int, int]
    square_size_mm: float

    def to_dict(self) -> Dict:
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.reshape(-1).tolist(),
            "new_camera_matrix": self.new_camera_matrix.tolist(),
            "roi": list(self.roi),
            "reprojection_error": float(self.reprojection_error),
            "used_images": self.used_images,
            "rejected_images": self.rejected_images,
            "board_shape": list(self.board_shape),
            "square_size_mm": float(self.square_size_mm),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CalibrationResult":
        return cls(
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            camera_matrix=np.asarray(data["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1),
            new_camera_matrix=np.asarray(data["new_camera_matrix"], dtype=np.float64),
            roi=tuple(int(v) for v in data["roi"]),
            reprojection_error=float(data["reprojection_error"]),
            used_images=list(data["used_images"]),
            rejected_images=list(data["rejected_images"]),
            board_shape=tuple(int(v) for v in data["board_shape"]),
            square_size_mm=float(data["square_size_mm"]),
        )


def save_calibration(result: CalibrationResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_calibration(path: Path) -> CalibrationResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    return CalibrationResult.from_dict(data)


def build_object_points(board_shape: Tuple[int, int], square_size_mm: float) -> np.ndarray:
    cols, rows = board_shape
    grid = np.zeros((rows * cols, 3), np.float32)
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
    grid[:, 0] = xs.reshape(-1) * square_size_mm
    grid[:, 1] = ys.reshape(-1) * square_size_mm
    return grid


def detect_chessboard_corners(gray: np.ndarray, board_shape: Tuple[int, int]):
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
    ok, corners = cv2.findChessboardCornersSB(gray, board_shape, flags=flags)
    return ok, corners


def calibrate_from_chessboard_images(
    image_paths: List[Path],
    board_shape: Tuple[int, int],
    square_size_mm: float,
) -> CalibrationResult:
    object_template = build_object_points(board_shape, square_size_mm)
    object_points = []
    image_points = []
    used_images: List[str] = []
    rejected_images: List[str] = []
    image_size: Optional[Tuple[int, int]] = None

    for image_path in image_paths:
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            rejected_images.append(image_path.name)
            continue
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        ok, corners = detect_chessboard_corners(gray, board_shape)
        if not ok or corners is None:
            rejected_images.append(image_path.name)
            continue
        object_points.append(object_template.copy())
        image_points.append(corners.astype(np.float32))
        used_images.append(image_path.name)

    if image_size is None:
        raise RuntimeError("No readable calibration images found.")
    if len(image_points) < 3:
        raise RuntimeError(f"Not enough valid calibration images: {len(image_points)}")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        image_size,
        0,
        image_size,
    )

    total_error = 0.0
    total_points = 0
    for objp, imgp, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        err = cv2.norm(imgp, projected, cv2.NORM_L2)
        total_error += err ** 2
        total_points += len(objp)
    reprojection_error = float(np.sqrt(total_error / max(total_points, 1)))

    return CalibrationResult(
        image_width=image_size[0],
        image_height=image_size[1],
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        new_camera_matrix=new_camera_matrix,
        roi=tuple(int(v) for v in roi),
        reprojection_error=reprojection_error,
        used_images=used_images,
        rejected_images=rejected_images,
        board_shape=board_shape,
        square_size_mm=square_size_mm,
    )


def undistort_image(image: np.ndarray, calib: CalibrationResult, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    height, width = image.shape[:2]
    map_x, map_y = cv2.initUndistortRectifyMap(
        calib.camera_matrix,
        calib.dist_coeffs,
        None,
        calib.new_camera_matrix,
        (width, height),
        cv2.CV_32FC1,
    )
    border_value = 0 if image.ndim == 2 else (0, 0, 0)
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def undistort_mask(mask: np.ndarray, calib: CalibrationResult) -> np.ndarray:
    height, width = mask.shape[:2]
    map_x, map_y = cv2.initUndistortRectifyMap(
        calib.camera_matrix,
        calib.dist_coeffs,
        None,
        calib.new_camera_matrix,
        (width, height),
        cv2.CV_32FC1,
    )
    return cv2.remap(
        mask.astype(np.uint8),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
