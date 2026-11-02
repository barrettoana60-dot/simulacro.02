"""
camera_acquisition.py
=====================
Camera status checking and initialization verification utilities.

This module provides lightweight camera probing via OpenCV's ``cv2.VideoCapture``
for use **before** ``GazeFollower`` is instantiated.  GazeFollower owns the
camera during an experiment; the helpers here must be called while GazeFollower
is *not* running so that the device is free to open temporarily.

Typical usage::

    from eye_tracking_exp.modules.camera_acquisition import CameraAcquisition

    cam = CameraAcquisition(camera_index=0)

    # 1. Confirm the desired index is accessible.
    info = cam.verify_camera_available()
    if not info["available"]:
        raise RuntimeError(f"Camera not available: {info['error']}")

    # 2. Optionally inspect image quality before starting GazeFollower.
    quality = cam.check_image_quality()
    if quality["is_dark"]:
        print("Warning: scene is very dark — check lighting conditions.")

    # --- Now it is safe to create GazeFollower ---
    gf = GazeFollower(...)
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
_BLURRY_LAPLACIAN_THRESHOLD: float = 100.0   # below → blurry
_DARK_BRIGHTNESS_THRESHOLD: float = 30.0     # below → too dark  (0-255 scale)
_SATURATED_BRIGHTNESS_THRESHOLD: float = 220.0  # above → saturated


class CameraAcquisition:
    """Camera availability and image-quality checker.

    This class **never** keeps a ``cv2.VideoCapture`` open between method
    calls.  Each public method opens the capture device, performs its checks,
    and releases immediately — ensuring the device is free for GazeFollower.

    Parameters
    ----------
    camera_index:
        Default camera device index used when methods are called without an
        explicit ``camera_index`` argument.
    """

    def __init__(self, camera_index: int = 0) -> None:
        self._default_index: int = camera_index

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_index(self, camera_index: Optional[int]) -> int:
        """Return *camera_index* if provided, otherwise the stored default."""
        return camera_index if camera_index is not None else self._default_index

    @staticmethod
    def _open_capture(index: int) -> tuple[cv2.VideoCapture, Optional[str]]:
        """Open a ``cv2.VideoCapture`` for *index*.

        Returns
        -------
        cap:
            The capture object (may not be opened successfully).
        error:
            Human-readable error string, or ``None`` on success.
        """
        try:
            cap = cv2.VideoCapture(index)
        except Exception as exc:  # pragma: no cover — defensive
            dummy = cv2.VideoCapture()
            return dummy, str(exc)

        if not cap.isOpened():
            return cap, f"cv2.VideoCapture could not open device index {index}"
        return cap, None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_camera_available(
        self,
        camera_index: Optional[int] = None,
    ) -> dict:
        """Test whether a camera device is accessible.

        Opens the device briefly, reads basic properties, then releases it.

        Parameters
        ----------
        camera_index:
            Device index to test.  Defaults to the index supplied at
            construction time.

        Returns
        -------
        dict with keys:

        * ``available`` (*bool*) — ``True`` if the device opened successfully.
        * ``index`` (*int*) — the tested device index.
        * ``resolution`` (*tuple[int, int] | None*) — ``(width, height)`` in
          pixels, or ``None`` when unavailable.
        * ``fps`` (*float | None*) — reported frame rate, or ``None``.
        * ``error`` (*str | None*) — error description, or ``None`` on
          success.
        """
        idx = self._resolve_index(camera_index)
        result: dict = {
            "available": False,
            "index": idx,
            "resolution": None,
            "fps": None,
            "error": None,
        }

        cap, error = self._open_capture(idx)
        try:
            if error:
                result["error"] = error
                logger.warning("verify_camera_available(%d): %s", idx, error)
                return result

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            result["available"] = True
            result["resolution"] = (width, height)
            result["fps"] = fps if fps > 0 else None
            logger.debug(
                "verify_camera_available(%d): OK — %dx%d @ %.1f fps",
                idx, width, height, fps,
            )
        finally:
            cap.release()

        return result

    def list_available_cameras(self, max_test: int = 5) -> list[int]:
        """Enumerate reachable camera device indices.

        Probes indices ``0`` through ``max_test - 1`` sequentially and
        returns those that open successfully.

        Parameters
        ----------
        max_test:
            Number of consecutive indices to test (default ``5``).

        Returns
        -------
        list[int]
            Sorted list of available device indices.
        """
        available: list[int] = []
        for idx in range(max_test):
            cap, error = self._open_capture(idx)
            try:
                if error:
                    logger.debug("list_available_cameras: index %d — %s", idx, error)
                    continue
                available.append(idx)
                logger.debug("list_available_cameras: index %d — available", idx)
            finally:
                cap.release()

        return available

    def get_camera_info(self, camera_index: Optional[int] = None) -> dict:
        """Return detailed information about a camera device.

        Parameters
        ----------
        camera_index:
            Device index to query.  Defaults to the stored default index.

        Returns
        -------
        dict with keys:

        * ``index`` (*int*) — device index.
        * ``resolution`` (*tuple[int, int] | None*) — ``(width, height)``.
        * ``fps`` (*float | None*) — reported frame rate.
        * ``backend`` (*str | None*) — OpenCV backend name (e.g. ``"AVFOUNDATION"``).
        * ``available`` (*bool*) — whether the device could be opened.
        * ``error`` (*str | None*) — error string when unavailable.
        """
        idx = self._resolve_index(camera_index)
        info: dict = {
            "index": idx,
            "resolution": None,
            "fps": None,
            "backend": None,
            "available": False,
            "error": None,
        }

        cap, error = self._open_capture(idx)
        try:
            if error:
                info["error"] = error
                logger.warning("get_camera_info(%d): %s", idx, error)
                return info

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            backend_id = int(cap.get(cv2.CAP_PROP_BACKEND))

            # cv2.videoio_registry is available in OpenCV ≥ 4.x
            try:
                backend_name: Optional[str] = cv2.videoio_registry.getBackendName(
                    backend_id
                )
            except (AttributeError, cv2.error):
                backend_name = str(backend_id) if backend_id else None

            info["resolution"] = (width, height)
            info["fps"] = fps if fps > 0 else None
            info["backend"] = backend_name
            info["available"] = True
        finally:
            cap.release()

        return info

    def check_image_quality(
        self,
        camera_index: Optional[int] = None,
        n_frames: int = 5,
    ) -> dict:
        """Capture a short burst of frames and evaluate image quality.

        Metrics are computed from the grayscale representation of each frame.
        The device is released immediately after capture.

        Parameters
        ----------
        camera_index:
            Device index to sample from.  Defaults to the stored default.
        n_frames:
            Number of frames to capture and average (default ``5``).

        Returns
        -------
        dict with keys:

        * ``mean_brightness`` (*float | None*) — mean pixel intensity
          (0–255) across all captured frames.
        * ``brightness_std`` (*float | None*) — standard deviation of
          per-frame mean brightnesses.
        * ``is_dark`` (*bool*) — ``True`` when ``mean_brightness`` is below
          ``30``.
        * ``is_saturated`` (*bool*) — ``True`` when ``mean_brightness``
          exceeds ``220``.
        * ``is_blurry`` (*bool*) — ``True`` when Laplacian variance is
          below ``100.0``.
        * ``laplacian_variance`` (*float | None*) — mean variance of the
          Laplacian operator applied to each frame's grayscale image.
        * ``frames_captured`` (*int*) — number of frames successfully read.
        * ``available`` (*bool*) — whether the device could be opened.
        * ``error`` (*str | None*) — error string on failure.

        Notes
        -----
        Blurriness is estimated via the variance of the Laplacian of a
        grayscale frame.  A variance below ``100.0`` is considered blurry.
        """
        idx = self._resolve_index(camera_index)
        result: dict = {
            "mean_brightness": None,
            "brightness_std": None,
            "is_dark": False,
            "is_saturated": False,
            "is_blurry": False,
            "laplacian_variance": None,
            "frames_captured": 0,
            "available": False,
            "error": None,
        }

        cap, error = self._open_capture(idx)
        try:
            if error:
                result["error"] = error
                logger.warning("check_image_quality(%d): %s", idx, error)
                return result

            result["available"] = True

            per_frame_brightness: list[float] = []
            per_frame_laplacian: list[float] = []

            for frame_num in range(n_frames):
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.debug(
                        "check_image_quality(%d): failed to read frame %d",
                        idx, frame_num,
                    )
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                per_frame_brightness.append(float(np.mean(gray)))

                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                per_frame_laplacian.append(float(laplacian.var()))

        finally:
            cap.release()

        frames_captured = len(per_frame_brightness)
        result["frames_captured"] = frames_captured

        if frames_captured == 0:
            result["error"] = (
                f"No frames could be read from device index {idx}"
            )
            logger.warning("check_image_quality(%d): no frames captured", idx)
            return result

        mean_brightness = float(np.mean(per_frame_brightness))
        brightness_std = float(np.std(per_frame_brightness))
        laplacian_variance = float(np.mean(per_frame_laplacian))

        result["mean_brightness"] = mean_brightness
        result["brightness_std"] = brightness_std
        result["laplacian_variance"] = laplacian_variance
        result["is_dark"] = mean_brightness < _DARK_BRIGHTNESS_THRESHOLD
        result["is_saturated"] = mean_brightness > _SATURATED_BRIGHTNESS_THRESHOLD
        result["is_blurry"] = laplacian_variance < _BLURRY_LAPLACIAN_THRESHOLD

        logger.debug(
            "check_image_quality(%d): brightness=%.1f±%.1f, "
            "laplacian_var=%.1f, dark=%s, saturated=%s, blurry=%s",
            idx,
            mean_brightness,
            brightness_std,
            laplacian_variance,
            result["is_dark"],
            result["is_saturated"],
            result["is_blurry"],
        )

        return result
