"""
data_recorder.py
----------------
Records gaze data during experimental sessions by subscribing to the
GazeFollower sample stream.

This module works **alongside** GazeFollower's built-in ``save_data()``
and does NOT replace it.  GazeFollower's ``save_data(path)`` continues
to write its own raw CSV; this module adds two supplementary files:

* ``raw_data.csv``       — one row per sample, mirroring GazeInfo fields
* ``processed_data.csv`` — derived / computed fields per sample

Typical usage
-------------
::

    recorder = DataRecorder(session_path="/data/session_01", session_id="s01")
    recorder.start_recording()
    gf.add_subscriber(recorder.record_sample)

    # … run experiment …

    gf.remove_subscriber(recorder.record_sample)
    recorder.stop_recording()
    stats = recorder.get_statistics()
"""

from __future__ import annotations

import csv
import logging
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CSV column definitions
# ---------------------------------------------------------------------------

_RAW_COLUMNS: Tuple[str, ...] = (
    "session_id",
    "timestamp",
    "raw_x",
    "raw_y",
    "calibrated_x",
    "calibrated_y",
    "filtered_x",
    "filtered_y",
    "left_openness",
    "right_openness",
    "tracking_status",
    "status",
    "trigger",
    "acquisition_time",
)

_PROCESSED_COLUMNS: Tuple[str, ...] = (
    "session_id",
    "timestamp",
    "gaze_x",
    "gaze_y",
    "is_valid",
    "processing_time_ms",
    "sample_rate_hz",
)


# ---------------------------------------------------------------------------
# DataRecorder
# ---------------------------------------------------------------------------


class DataRecorder:
    """Records gaze samples from a GazeFollower subscriber callback.

    Writes two CSV files inside *session_path*:

    * ``raw_data.csv``       — full GazeInfo fields per sample
    * ``processed_data.csv`` — validity flag, timing, and sample-rate

    The recorder is designed to be used as a GazeFollower subscriber::

        gf.add_subscriber(recorder.record_sample)

    All file I/O is protected by a ``threading.Lock`` so the callback can
    be invoked from a background thread without data corruption.

    Parameters
    ----------
    session_path:
        Directory where the two CSV files will be written.  Created
        automatically if it does not already exist.
    session_id:
        String identifier embedded in every data row for traceability.
    """

    def __init__(self, session_path: str, session_id: str) -> None:
        self._session_path: str = session_path
        self._session_id: str = session_id

        # File handles and CSV writers — only valid between
        # start_recording() and stop_recording()
        self._raw_file: Optional[Any] = None
        self._processed_file: Optional[Any] = None
        self._raw_writer: Optional[csv.DictWriter] = None
        self._processed_writer: Optional[csv.DictWriter] = None

        # State
        self._recording: bool = False
        self._lock: threading.Lock = threading.Lock()

        # Statistics
        self._sample_count: int = 0
        self._invalid_count: int = 0
        self._processing_times: list[float] = []  # milliseconds

        # Timing helpers for sample-rate estimation
        self._last_sample_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start_recording(self) -> None:
        """Open CSV files and write column headers.

        Must be called once before ``gf.add_subscriber(recorder.record_sample)``.

        Raises
        ------
        RuntimeError
            If recording is already active.
        OSError
            If the session directory cannot be created or files cannot be
            opened for writing.
        """
        with self._lock:
            if self._recording:
                raise RuntimeError(
                    "DataRecorder.start_recording() called while already recording."
                )

            os.makedirs(self._session_path, exist_ok=True)

            raw_path = os.path.join(self._session_path, "raw_data.csv")
            processed_path = os.path.join(self._session_path, "processed_data.csv")

            try:
                # newline='' is required by the csv module on all platforms
                self._raw_file = open(raw_path, "w", newline="", encoding="utf-8")
                self._processed_file = open(
                    processed_path, "w", newline="", encoding="utf-8"
                )
            except OSError as exc:
                # Clean up whichever file was opened before the error
                self._close_files_unsafe()
                raise OSError(
                    f"DataRecorder could not open CSV files in '{self._session_path}': {exc}"
                ) from exc

            self._raw_writer = csv.DictWriter(
                self._raw_file, fieldnames=_RAW_COLUMNS
            )
            self._processed_writer = csv.DictWriter(
                self._processed_file, fieldnames=_PROCESSED_COLUMNS
            )

            self._raw_writer.writeheader()
            self._processed_writer.writeheader()

            # Flush headers to disk immediately
            self._raw_file.flush()
            self._processed_file.flush()

            # Reset statistics
            self._sample_count = 0
            self._invalid_count = 0
            self._processing_times = []
            self._last_sample_time = None

            self._recording = True
            logger.info(
                "DataRecorder started. raw=%s  processed=%s",
                raw_path,
                processed_path,
            )

    def record_sample(self, face_info: Any, gaze_info: Any) -> None:
        """Subscriber callback — called by GazeFollower on every sample.

        Registers ``face_info`` and ``gaze_info`` as received from the
        GazeFollower subscriber mechanism.  When ``gaze_info`` is ``None``
        or ``gaze_info.status`` is ``False`` the sample is counted as
        invalid; a row is still written so the timeline stays intact.

        Parameters
        ----------
        face_info:
            Face-detection payload delivered by GazeFollower (may be
            ``None``).  Not written to CSV but accepted to match the
            expected ``func(face_info, gaze_info)`` signature.
        gaze_info:
            GazeInfo object delivered by GazeFollower, or ``None`` when
            no gaze could be estimated.
        """
        acquisition_time = time.perf_counter()

        with self._lock:
            if not self._recording:
                return

            t_start = time.perf_counter()

            # ---- determine validity and extract fields ----------------
            is_valid: bool = gaze_info is not None and bool(gaze_info.status)

            # Timestamps
            timestamp = gaze_info.timestamp if is_valid else ""

            # Raw coordinates
            raw_x: str = ""
            raw_y: str = ""
            if is_valid and gaze_info.raw_gaze_coordinates is not None:
                raw_x, raw_y = (
                    str(gaze_info.raw_gaze_coordinates[0]),
                    str(gaze_info.raw_gaze_coordinates[1]),
                )

            # Calibrated coordinates
            calibrated_x: str = ""
            calibrated_y: str = ""
            if is_valid and gaze_info.calibrated_gaze_coordinates is not None:
                calibrated_x, calibrated_y = (
                    str(gaze_info.calibrated_gaze_coordinates[0]),
                    str(gaze_info.calibrated_gaze_coordinates[1]),
                )

            # Filtered coordinates (used as primary gaze position)
            filtered_x: str = ""
            filtered_y: str = ""
            if is_valid and gaze_info.filtered_gaze_coordinates is not None:
                filtered_x, filtered_y = (
                    str(gaze_info.filtered_gaze_coordinates[0]),
                    str(gaze_info.filtered_gaze_coordinates[1]),
                )

            # Eye openness
            left_openness: str = (
                str(gaze_info.left_openness) if is_valid else ""
            )
            right_openness: str = (
                str(gaze_info.right_openness) if is_valid else ""
            )

            # Tracking state & event
            tracking_status: str = (
                str(gaze_info.tracking_state) if is_valid else ""
            )
            status_val: str = str(is_valid)
            trigger: str = str(gaze_info.event) if is_valid else ""

            # ---- write raw row ----------------------------------------
            raw_row: Dict[str, Any] = {
                "session_id": self._session_id,
                "timestamp": timestamp,
                "raw_x": raw_x,
                "raw_y": raw_y,
                "calibrated_x": calibrated_x,
                "calibrated_y": calibrated_y,
                "filtered_x": filtered_x,
                "filtered_y": filtered_y,
                "left_openness": left_openness,
                "right_openness": right_openness,
                "tracking_status": tracking_status,
                "status": status_val,
                "trigger": trigger,
                "acquisition_time": acquisition_time,
            }
            self._raw_writer.writerow(raw_row)  # type: ignore[union-attr]

            # ---- compute sample rate ----------------------------------
            now = time.perf_counter()
            sample_rate_hz: str = ""
            if self._last_sample_time is not None:
                interval = now - self._last_sample_time
                if interval > 0:
                    sample_rate_hz = f"{1.0 / interval:.4f}"
            self._last_sample_time = now

            # ---- write processed row ----------------------------------
            gaze_x: str = filtered_x
            gaze_y: str = filtered_y

            t_end = time.perf_counter()
            processing_time_ms = (t_end - t_start) * 1_000.0

            processed_row: Dict[str, Any] = {
                "session_id": self._session_id,
                "timestamp": timestamp,
                "gaze_x": gaze_x,
                "gaze_y": gaze_y,
                "is_valid": is_valid,
                "processing_time_ms": f"{processing_time_ms:.4f}",
                "sample_rate_hz": sample_rate_hz,
            }
            self._processed_writer.writerow(processed_row)  # type: ignore[union-attr]

            # ---- update statistics ------------------------------------
            self._sample_count += 1
            if not is_valid:
                self._invalid_count += 1
            self._processing_times.append(processing_time_ms)

    def stop_recording(self) -> None:
        """Flush and close both CSV files.

        Safe to call even if ``start_recording()`` was never called (no-op).
        After this call the recorder can be restarted with a fresh
        ``start_recording()`` invocation.
        """
        with self._lock:
            if not self._recording:
                logger.debug("DataRecorder.stop_recording() called while not recording — no-op.")
                return

            self._recording = False
            self._close_files_unsafe()
            logger.info(
                "DataRecorder stopped. samples=%d  invalid=%d",
                self._sample_count,
                self._invalid_count,
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Return a snapshot of recording statistics.

        Returns
        -------
        dict
            ``sample_count``          — total samples received
            ``invalid_count``         — samples where ``gaze_info.status`` was False
            ``data_loss_pct``         — percentage of invalid samples (0–100)
            ``mean_processing_time_ms`` — mean per-sample processing time in ms
        """
        with self._lock:
            total = self._sample_count
            invalid = self._invalid_count
            data_loss_pct = (invalid / total * 100.0) if total > 0 else 0.0
            mean_proc = (
                sum(self._processing_times) / len(self._processing_times)
                if self._processing_times
                else 0.0
            )
            return {
                "sample_count": total,
                "invalid_count": invalid,
                "data_loss_pct": round(data_loss_pct, 2),
                "mean_processing_time_ms": round(mean_proc, 4),
            }

    def get_sample_count(self) -> int:
        """Return the total number of samples received so far.

        Thread-safe; may be called while recording is active.
        """
        with self._lock:
            return self._sample_count

    def get_invalid_count(self) -> int:
        """Return the number of invalid samples (``gaze_info.status`` False).

        Thread-safe; may be called while recording is active.
        """
        with self._lock:
            return self._invalid_count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _close_files_unsafe(self) -> None:
        """Flush and close open file handles.

        **Must be called with ``self._lock`` already held.**
        Tolerates partially-opened state (e.g. an error mid-``start_recording``).
        """
        for fh in (self._raw_file, self._processed_file):
            if fh is not None:
                try:
                    fh.flush()
                    fh.close()
                except OSError as exc:
                    logger.warning("DataRecorder: error closing file: %s", exc)

        self._raw_file = None
        self._processed_file = None
        self._raw_writer = None
        self._processed_writer = None
