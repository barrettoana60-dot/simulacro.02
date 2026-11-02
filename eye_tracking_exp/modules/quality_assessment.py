"""
quality_assessment.py
---------------------
Computes data quality metrics exclusively from real collected eye-tracking
data.  **No simulated or fabricated values are ever produced.**

Metrics
-------
A) Sample rate   — effective_sample_rate = valid_samples / elapsed_time_s
B) Data loss     — (invalid_samples / total_samples) * 100
C) Processing time — mean / std / max of per-sample processing times (ms)
D) Tracking stability — std of filtered gaze over a sliding window of N
                        consecutive valid samples
E) Accuracy      — mean Euclidean distance to ground-truth target positions.
                   Computed ONLY when caller supplies validation target events.
F) Precision     — RMS spread of gaze samples inside caller-defined fixation
                   windows.

Pixel-to-degree conversion is available when `screen_params` is supplied at
construction time (see `__init__` for required keys).  Otherwise all spatial
metrics are reported in pixels only.

Thread safety
-------------
`add_sample` is protected by a threading.Lock so the class can be used from
a GazeFollower subscriber callback running in a background thread.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _Sample:
    """Single accumulated gaze sample."""
    timestamp: float          # seconds (or any consistent unit)
    gaze_x: float             # pixels
    gaze_y: float             # pixels
    is_valid: bool
    processing_time_ms: float # milliseconds


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class QualityAssessment:
    """
    Accumulates real eye-tracking samples and computes quality metrics.

    Parameters
    ----------
    screen_params : dict, optional
        Physical and logical screen dimensions required for pixel-to-degree
        conversion.  If omitted (or partially provided), spatial metrics are
        reported in pixels only.

        Expected keys
        ~~~~~~~~~~~~~
        - ``physical_screen_width_cm``  : float — screen width in centimetres
        - ``physical_screen_height_cm`` : float — screen height in centimetres
        - ``viewing_distance_cm``       : float — eye-to-screen distance (cm)
        - ``screen_width_px``           : int   — horizontal resolution
        - ``screen_height_px``          : int   — vertical resolution

    Notes
    -----
    * Never call any method with fabricated or placeholder data — all inputs
      must come from real hardware or recorded files.
    * All ``compute_*`` methods return ``None`` / ``{}`` when insufficient real
      data are available; callers should check the return value before use.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, screen_params: Optional[Dict] = None) -> None:
        self._lock = threading.Lock()
        self._samples: List[_Sample] = []

        # Validate and store screen parameters
        self._screen_params: Optional[Dict] = None
        if screen_params is not None:
            required_keys = {
                "physical_screen_width_cm",
                "physical_screen_height_cm",
                "viewing_distance_cm",
                "screen_width_px",
                "screen_height_px",
            }
            missing = required_keys - screen_params.keys()
            if missing:
                raise ValueError(
                    f"screen_params is missing required key(s): {missing}. "
                    "Either supply all required keys or omit screen_params "
                    "entirely to report in pixels only."
                )
            # Validate that all values are positive numbers
            for key in required_keys:
                val = screen_params[key]
                if not isinstance(val, (int, float)) or val <= 0:
                    raise ValueError(
                        f"screen_params['{key}'] must be a positive number, "
                        f"got {val!r}."
                    )
            self._screen_params = dict(screen_params)

    # ------------------------------------------------------------------
    # Sample accumulation
    # ------------------------------------------------------------------

    def add_sample(
        self,
        timestamp: float,
        gaze_x: float,
        gaze_y: float,
        is_valid: bool,
        processing_time_ms: float,
    ) -> None:
        """
        Accumulate one real gaze sample.

        This method is thread-safe and may be called directly from a
        GazeFollower subscriber callback.

        Parameters
        ----------
        timestamp : float
            Sample timestamp in the same unit used for elapsed-time
            calculations (e.g. seconds from experiment start).
        gaze_x : float
            Horizontal gaze coordinate in pixels (filtered preferred;
            raw accepted when filtered is unavailable).
        gaze_y : float
            Vertical gaze coordinate in pixels.
        is_valid : bool
            ``True`` when the tracker reports a confident estimate
            (``gaze_info.status is True`` and tracking state is good).
        processing_time_ms : float
            Per-sample processing latency in milliseconds as reported by the
            pipeline (> 0).

        Raises
        ------
        TypeError
            If any argument has an unexpected type.
        ValueError
            If ``processing_time_ms`` is negative.
        """
        if not isinstance(is_valid, bool):
            raise TypeError(
                f"is_valid must be bool, got {type(is_valid).__name__!r}."
            )
        if processing_time_ms < 0:
            raise ValueError(
                f"processing_time_ms must be >= 0, got {processing_time_ms}."
            )

        sample = _Sample(
            timestamp=float(timestamp),
            gaze_x=float(gaze_x),
            gaze_y=float(gaze_y),
            is_valid=is_valid,
            processing_time_ms=float(processing_time_ms),
        )
        with self._lock:
            self._samples.append(sample)

    # ------------------------------------------------------------------
    # Metric A — Sample rate
    # ------------------------------------------------------------------

    def compute_sample_rate(self, elapsed_seconds: float) -> Optional[float]:
        """
        Effective sample rate: valid samples per second of elapsed time.

        ``effective_sample_rate = valid_samples / elapsed_seconds``

        Parameters
        ----------
        elapsed_seconds : float
            Wall-clock duration of the recording in seconds (must be > 0).

        Returns
        -------
        float
            Samples per second, or ``None`` if there are no samples or
            ``elapsed_seconds`` is not positive.
        """
        if elapsed_seconds <= 0:
            return None

        with self._lock:
            valid = sum(1 for s in self._samples if s.is_valid)

        if valid == 0:
            return 0.0

        return valid / elapsed_seconds

    # ------------------------------------------------------------------
    # Metric B — Data loss
    # ------------------------------------------------------------------

    def compute_data_loss(self) -> Optional[float]:
        """
        Percentage of samples that are invalid (data loss).

        ``data_loss_pct = (invalid_samples / total_samples) * 100``

        Returns
        -------
        float
            Data loss percentage in [0, 100], or ``None`` if no samples have
            been accumulated yet.
        """
        with self._lock:
            total = len(self._samples)
            if total == 0:
                return None
            invalid = sum(1 for s in self._samples if not s.is_valid)

        return (invalid / total) * 100.0

    # ------------------------------------------------------------------
    # Metric C — Processing time statistics
    # ------------------------------------------------------------------

    def compute_processing_stats(self) -> Dict:
        """
        Descriptive statistics of per-sample processing times in milliseconds.

        Computed over **all** accumulated samples (valid and invalid), because
        processing latency is independent of tracking validity.

        Returns
        -------
        dict
            ``{"mean_ms": float, "std_ms": float, "max_ms": float}``
            or ``{}`` if no samples are available.
        """
        with self._lock:
            times = [s.processing_time_ms for s in self._samples]

        if not times:
            return {}

        n = len(times)
        mean_ms = sum(times) / n
        max_ms = max(times)
        variance = sum((t - mean_ms) ** 2 for t in times) / n
        std_ms = math.sqrt(variance)

        return {
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "max_ms": max_ms,
        }

    # ------------------------------------------------------------------
    # Metric D — Tracking stability
    # ------------------------------------------------------------------

    def compute_stability(self, window_size: int = 30) -> Dict:
        """
        Spatial stability of gaze measured as the standard deviation of
        filtered gaze coordinates over a sliding window of consecutive
        valid samples.

        The *last* ``window_size`` valid samples in the buffer are used so
        that the metric reflects the most recent recording quality.

        Parameters
        ----------
        window_size : int
            Number of consecutive valid samples in the analysis window.
            Default is 30 (≈ 1 second at 30 Hz).

        Returns
        -------
        dict
            Keys present when real data are available:

            - ``"std_x_px"`` — horizontal std in pixels
            - ``"std_y_px"`` — vertical std in pixels
            - ``"std_x_deg"`` — horizontal std in degrees (requires
              ``screen_params``)
            - ``"std_y_deg"`` — vertical std in degrees (requires
              ``screen_params``)
            - ``"n_samples"`` — number of samples used
            - ``"window_size"`` — requested window size

            Returns ``{}`` if fewer than 2 valid samples are available.
        """
        if window_size < 2:
            raise ValueError("window_size must be >= 2.")

        with self._lock:
            valid_samples = [s for s in self._samples if s.is_valid]

        if len(valid_samples) < 2:
            return {}

        window = valid_samples[-window_size:]
        n = len(window)

        xs = [s.gaze_x for s in window]
        ys = [s.gaze_y for s in window]

        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        std_x_px = math.sqrt(sum((x - mean_x) ** 2 for x in xs) / n)
        std_y_px = math.sqrt(sum((y - mean_y) ** 2 for y in ys) / n)

        result: Dict = {
            "std_x_px": std_x_px,
            "std_y_px": std_y_px,
            "n_samples": n,
            "window_size": window_size,
        }

        std_x_deg = self.pixels_to_degrees(std_x_px, axis="x")
        std_y_deg = self.pixels_to_degrees(std_y_px, axis="y")
        if std_x_deg is not None:
            result["std_x_deg"] = std_x_deg
            result["std_y_deg"] = std_y_deg

        return result

    # ------------------------------------------------------------------
    # Metric E — Accuracy (ground-truth required)
    # ------------------------------------------------------------------

    def compute_accuracy(self, target_events: List[Dict]) -> Dict:
        """
        Mean Euclidean distance between known target positions and the nearest
        real gaze sample in time.

        .. important::
            This method **only** computes accuracy when real validation target
            events are provided.  It raises ``ValueError`` if
            ``target_events`` is empty or ``None`` — accuracy is never
            estimated without ground truth.

        Parameters
        ----------
        target_events : list of dict
            Each dict must contain:

            - ``"timestamp"`` : float — time of target presentation
            - ``"target_x"``  : float — target x position in pixels
            - ``"target_y"``  : float — target y position in pixels

        Returns
        -------
        dict
            - ``"mean_error_px"``  : float
            - ``"std_error_px"``   : float
            - ``"max_error_px"``   : float
            - ``"n_targets"``      : int   — number of matched targets
            - ``"mean_error_deg"`` : float (only if ``screen_params`` set)
            - ``"std_error_deg"``  : float (only if ``screen_params`` set)

            Returns ``{}`` if no valid gaze samples are available to match
            against targets.

        Raises
        ------
        ValueError
            If ``target_events`` is ``None`` or empty — accuracy cannot be
            computed without ground-truth targets.
        """
        if not target_events:
            raise ValueError(
                "compute_accuracy() requires at least one target event with "
                "ground-truth coordinates. Accuracy is never computed without "
                "real validation targets."
            )

        # Validate required keys in each event
        for i, ev in enumerate(target_events):
            for key in ("timestamp", "target_x", "target_y"):
                if key not in ev:
                    raise ValueError(
                        f"target_events[{i}] is missing required key "
                        f"'{key}'."
                    )

        with self._lock:
            valid_samples = [s for s in self._samples if s.is_valid]

        if not valid_samples:
            return {}

        errors_px: List[float] = []

        for ev in target_events:
            t_ref = float(ev["timestamp"])
            tx = float(ev["target_x"])
            ty = float(ev["target_y"])

            # Find the valid sample with the smallest timestamp difference
            nearest = min(
                valid_samples,
                key=lambda s: abs(s.timestamp - t_ref),
            )

            dist = math.sqrt(
                (nearest.gaze_x - tx) ** 2 + (nearest.gaze_y - ty) ** 2
            )
            errors_px.append(dist)

        n = len(errors_px)
        mean_err_px = sum(errors_px) / n
        max_err_px = max(errors_px)
        variance = sum((e - mean_err_px) ** 2 for e in errors_px) / n
        std_err_px = math.sqrt(variance)

        result: Dict = {
            "mean_error_px": mean_err_px,
            "std_error_px": std_err_px,
            "max_error_px": max_err_px,
            "n_targets": n,
        }

        mean_err_deg = self.pixels_to_degrees(mean_err_px, axis="x")
        std_err_deg = self.pixels_to_degrees(std_err_px, axis="x")
        if mean_err_deg is not None:
            result["mean_error_deg"] = mean_err_deg
            result["std_error_deg"] = std_err_deg

        return result

    # ------------------------------------------------------------------
    # Metric F — Precision
    # ------------------------------------------------------------------

    def compute_precision(self, fixation_windows: List[Dict]) -> Dict:
        """
        RMS (root mean square) precision of gaze samples within caller-defined
        fixation windows.

        Precision measures gaze *spread* during a presumed stable fixation.
        A lower value indicates a more precise tracker.

        Parameters
        ----------
        fixation_windows : list of dict
            Each dict must contain:

            - ``"start_t"`` : float — window start timestamp (inclusive)
            - ``"end_t"``   : float — window end timestamp (inclusive)

        Returns
        -------
        dict
            - ``"rms_precision_px"`` : float — overall RMS in pixels
            - ``"rms_precision_deg"``: float — RMS in degrees (if
              ``screen_params`` available)
            - ``"n_windows"``        : int   — number of windows used
            - ``"n_samples"``        : int   — total valid samples analysed

            Returns ``{}`` if no fixation windows are given or no valid
            samples fall within any window.

        Raises
        ------
        ValueError
            If ``fixation_windows`` contains entries missing ``start_t`` or
            ``end_t``, or if ``start_t >= end_t``.
        """
        if not fixation_windows:
            return {}

        for i, win in enumerate(fixation_windows):
            for key in ("start_t", "end_t"):
                if key not in win:
                    raise ValueError(
                        f"fixation_windows[{i}] is missing required key "
                        f"'{key}'."
                    )
            if win["start_t"] >= win["end_t"]:
                raise ValueError(
                    f"fixation_windows[{i}]: start_t must be < end_t "
                    f"(got {win['start_t']} >= {win['end_t']})."
                )

        with self._lock:
            all_samples = list(self._samples)

        squared_errors: List[float] = []
        windows_used = 0

        for win in fixation_windows:
            st = float(win["start_t"])
            et = float(win["end_t"])

            window_samples = [
                s for s in all_samples
                if s.is_valid and st <= s.timestamp <= et
            ]

            if len(window_samples) < 2:
                # Not enough data in this window — skip silently
                continue

            windows_used += 1
            xs = [s.gaze_x for s in window_samples]
            ys = [s.gaze_y for s in window_samples]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)

            for s in window_samples:
                dist_sq = (s.gaze_x - cx) ** 2 + (s.gaze_y - cy) ** 2
                squared_errors.append(dist_sq)

        if not squared_errors:
            return {}

        rms_px = math.sqrt(sum(squared_errors) / len(squared_errors))

        result: Dict = {
            "rms_precision_px": rms_px,
            "n_windows": windows_used,
            "n_samples": len(squared_errors),
        }

        rms_deg = self.pixels_to_degrees(rms_px, axis="x")
        if rms_deg is not None:
            result["rms_precision_deg"] = rms_deg

        return result

    # ------------------------------------------------------------------
    # Aggregated report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        elapsed_seconds: Optional[float] = None,
        target_events: Optional[List[Dict]] = None,
        fixation_windows: Optional[List[Dict]] = None,
        stability_window_size: int = 30,
    ) -> Dict:
        """
        Aggregate all available quality metrics into a single report dict.

        Metrics that require optional inputs (accuracy, sample rate) are
        included only when the necessary arguments are supplied.  No metric is
        fabricated or estimated from synthetic data.

        Parameters
        ----------
        elapsed_seconds : float, optional
            Required for sample-rate computation.
        target_events : list of dict, optional
            Required for accuracy computation (see :meth:`compute_accuracy`).
        fixation_windows : list of dict, optional
            Required for precision computation (see
            :meth:`compute_precision`).
        stability_window_size : int
            Passed to :meth:`compute_stability`. Default 30.

        Returns
        -------
        dict
            ``{
                "data_loss_pct":      float | None,
                "processing_stats":  dict,
                "stability":         dict,
                "sample_rate_hz":    float | None,   # only if elapsed_seconds given
                "accuracy":          dict,            # only if target_events given
                "precision":         dict,            # only if fixation_windows given
            }``
        """
        report: Dict = {}

        report["data_loss_pct"] = self.compute_data_loss()
        report["processing_stats"] = self.compute_processing_stats()
        report["stability"] = self.compute_stability(
            window_size=stability_window_size
        )

        if elapsed_seconds is not None:
            report["sample_rate_hz"] = self.compute_sample_rate(elapsed_seconds)

        if target_events:
            try:
                report["accuracy"] = self.compute_accuracy(target_events)
            except ValueError as exc:
                report["accuracy"] = {"error": str(exc)}

        if fixation_windows:
            report["precision"] = self.compute_precision(fixation_windows)

        return report

    # ------------------------------------------------------------------
    # Pixel-to-degree conversion utility
    # ------------------------------------------------------------------

    def pixels_to_degrees(
        self, px: float, axis: str = "x"
    ) -> Optional[float]:
        """
        Convert a pixel distance to visual degrees using physical screen
        parameters.

        The conversion uses the formula::

            degrees = 2 * arctan(physical_size / (2 * viewing_distance))

        where ``physical_size = px * (screen_physical_cm / screen_px)``.

        Parameters
        ----------
        px : float
            Distance in pixels to convert (must be >= 0).
        axis : str
            ``'x'`` (horizontal) or ``'y'`` (vertical).  Determines which
            physical dimension and pixel count are used.

        Returns
        -------
        float
            Visual degrees corresponding to ``px``, or ``None`` if
            ``screen_params`` were not provided at construction.

        Raises
        ------
        ValueError
            If ``axis`` is not ``'x'`` or ``'y'``.
        """
        if axis not in ("x", "y"):
            raise ValueError(
                f"axis must be 'x' or 'y', got {axis!r}."
            )

        if self._screen_params is None:
            return None

        p = self._screen_params
        viewing_distance_cm: float = p["viewing_distance_cm"]

        if axis == "x":
            physical_cm: float = p["physical_screen_width_cm"]
            screen_px: int = p["screen_width_px"]
        else:
            physical_cm = p["physical_screen_height_cm"]
            screen_px = p["screen_height_px"]

        # cm per pixel along this axis
        cm_per_px = physical_cm / screen_px
        physical_size_cm = px * cm_per_px

        # Visual angle in degrees
        angle_rad = 2.0 * math.atan(
            physical_size_cm / (2.0 * viewing_distance_cm)
        )
        return math.degrees(angle_rad)

    # ------------------------------------------------------------------
    # Helpers / repr
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """
        Discard all accumulated samples.

        Use this to reset the assessor between experimental blocks without
        recreating the object (preserving ``screen_params``).
        """
        with self._lock:
            self._samples.clear()

    @property
    def total_samples(self) -> int:
        """Total number of samples accumulated (valid + invalid)."""
        with self._lock:
            return len(self._samples)

    @property
    def valid_samples(self) -> int:
        """Number of valid samples accumulated."""
        with self._lock:
            return sum(1 for s in self._samples if s.is_valid)

    def __repr__(self) -> str:
        with self._lock:
            n = len(self._samples)
            v = sum(1 for s in self._samples if s.is_valid)
        params = "set" if self._screen_params else "not set"
        return (
            f"QualityAssessment("
            f"total_samples={n}, valid_samples={v}, "
            f"screen_params={params})"
        )
