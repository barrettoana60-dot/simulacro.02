"""
gaze_estimator_wrapper.py
=========================
Wraps GazeFollower's gaze output for use inside the experiment pipeline.

This module does **NOT** re-implement gaze estimation; all estimation is
performed by the underlying :class:`GazeFollower` instance.  The wrapper's
responsibilities are:

- Subscribe to the GazeFollower sample stream as a callback.
- Extract the relevant fields from each :class:`GazeInfo` sample.
- Forward each processed sample to a ``QualityAssessment`` object.
- Maintain a thread-safe rolling history of recent samples.
- Expose a clean, typed accessor API to the rest of the experiment code.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
GazeSample = Dict[str, Any]
"""
A single processed gaze sample with the following keys:

- ``x``              : float or None — horizontal gaze coordinate (pixels)
- ``y``              : float or None — vertical gaze coordinate (pixels)
- ``raw_x``          : float or None — raw (uncalibrated) horizontal coordinate
- ``raw_y``          : float or None — raw (uncalibrated) vertical coordinate
- ``is_valid``       : bool          — True when gaze_info.status is True
- ``timestamp``      : float or None — sample timestamp from GazeFollower
- ``tracking_state`` : Any           — gaze_info.tracking_state enum value
- ``processing_time_ms`` : float     — wall-clock time taken to process this sample (ms)
"""

_HISTORY_MAXLEN: int = 1000  # Maximum samples kept in the rolling buffer


class GazeEstimatorWrapper:
    """Thin wrapper around a :class:`GazeFollower` instance.

    Subscribes to the GazeFollower sample stream, processes each incoming
    :class:`GazeInfo` object, and exposes the most recent gaze position
    together with a rolling history of past samples.

    Parameters
    ----------
    gaze_follower_instance:
        An already-constructed ``GazeFollower`` object.  The wrapper calls
        ``gf.add_subscriber`` / ``gf.remove_subscriber`` on this object.
    quality_assessment_instance:
        An object that accepts processed gaze samples for quality tracking.
        Must expose a ``process_sample(sample: GazeSample) -> None`` method.
    """

    def __init__(
        self,
        gaze_follower_instance: Any,
        quality_assessment_instance: Any,
    ) -> None:
        self._gf = gaze_follower_instance
        self._qa = quality_assessment_instance

        # Internal state — guarded by _lock
        self._lock: threading.Lock = threading.Lock()
        self._latest_sample: Optional[GazeSample] = None
        self._history: deque[GazeSample] = deque(maxlen=_HISTORY_MAXLEN)

    # ------------------------------------------------------------------
    # Subscriber management
    # ------------------------------------------------------------------

    def register_with_follower(self) -> None:
        """Register :meth:`on_gaze_sample` as a GazeFollower subscriber.

        After this call the wrapper will receive every gaze sample produced
        by the underlying ``GazeFollower`` instance.

        Raises
        ------
        RuntimeError
            If the underlying ``GazeFollower`` raises during subscription.
        """
        self._gf.add_subscriber(self.on_gaze_sample)

    def unregister_from_follower(self) -> None:
        """Remove :meth:`on_gaze_sample` from the GazeFollower subscriber list.

        Safe to call even when no new samples are being produced.

        Raises
        ------
        RuntimeError
            If the underlying ``GazeFollower`` raises during removal.
        """
        self._gf.remove_subscriber(self.on_gaze_sample)

    # ------------------------------------------------------------------
    # Sample callback
    # ------------------------------------------------------------------

    def on_gaze_sample(self, face_info: Any, gaze_info: Any) -> None:
        """GazeFollower subscriber callback — called once per captured frame.

        Extracts position and metadata from *gaze_info*, builds a
        :data:`GazeSample` dict, appends it to the rolling history, and
        forwards it to the quality-assessment object.

        .. warning::
            This method may be called from a background thread by
            ``GazeFollower``.  All shared state is protected by an internal
            :class:`threading.Lock`.

        Parameters
        ----------
        face_info:
            Face-tracking information provided by GazeFollower (not used
            directly here; reserved for future extensions).
        gaze_info:
            A ``GazeInfo`` object produced by GazeFollower.  When
            ``gaze_info.status`` is ``False`` the gaze coordinates are
            unavailable and stored as ``None``.
        """
        t_start: float = time.perf_counter()

        # ----------------------------------------------------------------
        # Extract fields — gaze_info.status is the canonical validity flag
        # ----------------------------------------------------------------
        is_valid: bool = bool(gaze_info.status)

        gaze_x: Optional[float] = None
        gaze_y: Optional[float] = None
        raw_x: Optional[float] = None
        raw_y: Optional[float] = None

        if is_valid:
            # filtered_gaze_coordinates is a (x, y) tuple
            filtered: Tuple[float, float] = gaze_info.filtered_gaze_coordinates
            gaze_x, gaze_y = filtered[0], filtered[1]

            # raw_gaze_coordinates is also a (x, y) tuple
            raw: Tuple[float, float] = gaze_info.raw_gaze_coordinates
            raw_x, raw_y = raw[0], raw[1]

        sample: GazeSample = {
            "x": gaze_x,
            "y": gaze_y,
            "raw_x": raw_x,
            "raw_y": raw_y,
            "is_valid": is_valid,
            "timestamp": gaze_info.timestamp,
            "tracking_state": gaze_info.tracking_state,
            "processing_time_ms": 0.0,  # updated below after QA call
        }

        # ----------------------------------------------------------------
        # Quality assessment — run while we still have the raw objects
        # ----------------------------------------------------------------
        try:
            self._qa.process_sample(sample)
        except Exception:
            # Never let a QA failure crash the sample pipeline
            pass

        # Record processing time *including* QA overhead
        sample["processing_time_ms"] = (time.perf_counter() - t_start) * 1_000.0

        # ----------------------------------------------------------------
        # Update shared state under the lock
        # ----------------------------------------------------------------
        with self._lock:
            self._latest_sample = sample
            self._history.append(sample)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_current_gaze(self) -> Optional[GazeSample]:
        """Return the most recent gaze sample using *filtered* coordinates.

        Returns
        -------
        dict or None
            A :data:`GazeSample` dict with keys ``x``, ``y``, ``is_valid``,
            ``timestamp``, and ``tracking_state``, or ``None`` if no sample
            has been received yet.

            When ``is_valid`` is ``False``, ``x`` and ``y`` are ``None``.
        """
        with self._lock:
            if self._latest_sample is None:
                return None
            s = self._latest_sample
            return {
                "x": s["x"],
                "y": s["y"],
                "is_valid": s["is_valid"],
                "timestamp": s["timestamp"],
                "tracking_state": s["tracking_state"],
            }

    def get_raw_gaze(self) -> Optional[GazeSample]:
        """Return the most recent gaze sample using *raw* (unfiltered) coordinates.

        Returns
        -------
        dict or None
            A :data:`GazeSample` dict with keys ``x``, ``y``, ``is_valid``,
            ``timestamp``, and ``tracking_state`` (where ``x``/``y`` map to
            the raw coordinates), or ``None`` if no sample has been received
            yet.

            When ``is_valid`` is ``False``, ``x`` and ``y`` are ``None``.
        """
        with self._lock:
            if self._latest_sample is None:
                return None
            s = self._latest_sample
            return {
                "x": s["raw_x"],
                "y": s["raw_y"],
                "is_valid": s["is_valid"],
                "timestamp": s["timestamp"],
                "tracking_state": s["tracking_state"],
            }

    def is_tracking(self) -> bool:
        """Return ``True`` if the most recent sample had a valid gaze fix.

        Returns ``False`` both when no sample has been received yet and when
        the last sample carried ``gaze_info.status == False``.
        """
        with self._lock:
            if self._latest_sample is None:
                return False
            return bool(self._latest_sample["is_valid"])

    def get_sample_history(self, n: int = 100) -> List[GazeSample]:
        """Return a snapshot of the *n* most recent samples.

        Parameters
        ----------
        n:
            Number of recent samples to return.  Clamped to the internal
            buffer size (``1000``).  Defaults to ``100``.

        Returns
        -------
        list of dict
            A new list containing up to *n* :data:`GazeSample` dicts, ordered
            from oldest to newest.  Returns an empty list if no samples have
            been recorded yet.
        """
        n = max(0, min(n, _HISTORY_MAXLEN))
        with self._lock:
            history_snapshot = list(self._history)

        # Slice the last n entries (history is ordered oldest → newest)
        return history_snapshot[-n:] if n > 0 else []

    def clear_history(self) -> None:
        """Discard all samples stored in the rolling history buffer.

        The latest-sample pointer is also cleared so that :meth:`is_tracking`
        returns ``False`` until the next sample arrives.
        """
        with self._lock:
            self._history.clear()
            self._latest_sample = None
