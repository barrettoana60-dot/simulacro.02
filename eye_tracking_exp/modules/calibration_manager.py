"""
calibration_manager.py
-----------------------
Wraps GazeFollower's built-in calibration sequence without reimplementing it.

The GazeFollower library owns the calibration algorithm entirely.  This module
adds:

* **Lifecycle tracking** — records timestamps, duration, and attempt count.
* **Logging** — emits structured log messages at every stage.
* **Error propagation** — catches all exceptions from ``gf.preview()`` /
  ``gf.calibrate()``, logs them to both the Python logger and the
  ``session_manager``, then re-raises with a descriptive message.

Typical usage
-------------
::

    cm = CalibrationManager(gaze_follower_instance=gf, session_manager=sm)

    cm.run_preview(win=pygame_win)          # optional camera preview
    result = cm.run_calibration(win=pygame_win)

    if result.success:
        print(f"Calibrated in {result.duration_seconds:.1f}s")

    # Later, if re-calibration is needed:
    result2 = cm.recalibrate(win=pygame_win)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CalibrationResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class CalibrationResult:
    """Immutable record of a single calibration attempt.

    Attributes
    ----------
    success:
        ``True`` when ``gf.calibrate()`` completed without raising an exception.
    timestamp:
        ISO-8601 UTC string recorded **after** the calibration call returned.
    duration_seconds:
        Wall-clock time (seconds) spent inside ``gf.calibrate()``.
    attempt_number:
        1-based counter indicating which attempt produced this result.
    error_message:
        Human-readable description of the exception if ``success`` is
        ``False``; ``None`` otherwise.
    """

    success: bool
    timestamp: str
    duration_seconds: float
    attempt_number: int
    error_message: Optional[str] = field(default=None)


# ---------------------------------------------------------------------------
# CalibrationManager
# ---------------------------------------------------------------------------


class CalibrationManager:
    """Lifecycle wrapper around GazeFollower's calibration primitives.

    This class does **not** implement any calibration algorithm.  All
    calibration logic is delegated entirely to ``gf.preview()`` and
    ``gf.calibrate()``.

    Parameters
    ----------
    gaze_follower_instance:
        An already-constructed ``GazeFollower`` object.
    session_manager:
        Any object that exposes a ``log(message: str)`` method (or
        equivalent) used to write calibration events into the session log.
        If the object does not expose ``log``, a warning is emitted and
        logging falls back to the Python ``logging`` module only.
    """

    def __init__(self, gaze_follower_instance: Any, session_manager: Any) -> None:
        self._gf: Any = gaze_follower_instance
        self._sm: Any = session_manager

        # Internal state — protected by a lock so that is_calibrated() and
        # get_calibration_info() are safe to call from any thread.
        self._lock: threading.Lock = threading.Lock()
        self._calibrated: bool = False
        self._calibration_time: Optional[str] = None      # ISO-8601 UTC
        self._calibration_duration: float = 0.0           # seconds
        self._attempt_count: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_preview(self, win: Any = None) -> bool:
        """Display the camera preview window via ``gf.preview()``.

        This is an optional pre-calibration step that lets the participant
        confirm their position and lighting before the calibration sequence
        begins.  The method delegates entirely to GazeFollower.

        Parameters
        ----------
        win:
            Pygame (or compatible) display surface passed through to
            ``gf.preview(win=win)``.  Pass ``None`` to let GazeFollower
            create its own window.

        Returns
        -------
        bool
            ``True`` when ``gf.preview()`` returns without raising an
            exception.

        Raises
        ------
        RuntimeError
            If ``gf.preview()`` raises any exception.  The original
            exception is chained and the session manager is notified before
            re-raising.
        """
        self._session_log("CalibrationManager: starting camera preview.")
        logger.info("CalibrationManager: calling gf.preview(win=%r).", win)

        try:
            self._gf.preview(win=win)
        except Exception as exc:
            msg = f"CalibrationManager: gf.preview() failed — {exc}"
            self._session_log(msg)
            logger.error(msg, exc_info=True)
            raise RuntimeError(msg) from exc

        self._session_log("CalibrationManager: camera preview completed successfully.")
        logger.info("CalibrationManager: gf.preview() completed successfully.")
        return True

    def run_calibration(self, win: Any = None) -> CalibrationResult:
        """Run the GazeFollower calibration sequence via ``gf.calibrate()``.

        Records the wall-clock time before and after the call so that
        ``duration_seconds`` reflects the real time spent inside the
        calibration routine.  On success ``is_calibrated()`` will return
        ``True`` and ``get_calibration_info()`` will be populated.

        Parameters
        ----------
        win:
            Pygame (or compatible) display surface passed through to
            ``gf.calibrate(win=win)``.  Pass ``None`` to let GazeFollower
            manage its own window.

        Returns
        -------
        CalibrationResult
            Dataclass describing the outcome of the attempt.

        Raises
        ------
        RuntimeError
            If ``gf.calibrate()`` raises any exception.  The original
            exception is chained, the session manager is notified, and
            ``is_calibrated()`` remains ``False`` (or its previous value).
        """
        with self._lock:
            self._attempt_count += 1
            attempt = self._attempt_count

        self._session_log(
            f"CalibrationManager: starting calibration attempt #{attempt}."
        )
        logger.info(
            "CalibrationManager: calling gf.calibrate(win=%r) — attempt #%d.",
            win,
            attempt,
        )

        import time  # imported here to avoid polluting module-level namespace

        t_start = time.perf_counter()

        try:
            self._gf.calibrate(win=win)
        except Exception as exc:
            t_end = time.perf_counter()
            duration = t_end - t_start
            ts = _utc_iso_now()

            msg = (
                f"CalibrationManager: gf.calibrate() failed on attempt #{attempt} "
                f"after {duration:.2f}s — {exc}"
            )
            self._session_log(msg)
            logger.error(msg, exc_info=True)

            result = CalibrationResult(
                success=False,
                timestamp=ts,
                duration_seconds=round(duration, 4),
                attempt_number=attempt,
                error_message=str(exc),
            )
            raise RuntimeError(msg) from exc

        t_end = time.perf_counter()
        duration = t_end - t_start
        ts = _utc_iso_now()

        with self._lock:
            self._calibrated = True
            self._calibration_time = ts
            self._calibration_duration = duration

        success_msg = (
            f"CalibrationManager: calibration attempt #{attempt} completed "
            f"successfully in {duration:.2f}s."
        )
        self._session_log(success_msg)
        logger.info(success_msg)

        return CalibrationResult(
            success=True,
            timestamp=ts,
            duration_seconds=round(duration, 4),
            attempt_number=attempt,
            error_message=None,
        )

    def is_calibrated(self) -> bool:
        """Return ``True`` if at least one calibration attempt has succeeded.

        Thread-safe; may be called from any thread at any time.
        """
        with self._lock:
            return self._calibrated

    def get_calibration_info(self) -> Dict[str, Any]:
        """Return a snapshot of the current calibration state.

        Returns
        -------
        dict
            ``calibration_time``   — ISO-8601 UTC timestamp of the last
                                     successful calibration, or ``None``.\n
            ``duration_seconds``   — wall-clock duration of the last
                                     successful calibration (float), or
                                     ``0.0`` if none has completed.\n
            ``completed``          — ``True`` if at least one calibration
                                     has succeeded.\n
            ``attempt_count``      — total number of calibration attempts
                                     (including failed ones).

        Thread-safe; may be called from any thread at any time.
        """
        with self._lock:
            return {
                "calibration_time": self._calibration_time,
                "duration_seconds": self._calibration_duration,
                "completed": self._calibrated,
                "attempt_count": self._attempt_count,
            }

    def recalibrate(self, win: Any = None) -> CalibrationResult:
        """Run the calibration sequence again.

        Functionally identical to :meth:`run_calibration` but clearly
        communicates intent at the call site.  ``attempt_count`` is
        incremented regardless of the outcome.

        Parameters
        ----------
        win:
            Pygame (or compatible) display surface forwarded to
            ``gf.calibrate(win=win)``.

        Returns
        -------
        CalibrationResult
            Dataclass describing the outcome of this attempt.

        Raises
        ------
        RuntimeError
            Propagated from :meth:`run_calibration` if ``gf.calibrate()``
            raises an exception.
        """
        self._session_log("CalibrationManager: recalibration requested.")
        logger.info("CalibrationManager: recalibrate() — delegating to run_calibration().")
        return self.run_calibration(win=win)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _session_log(self, message: str) -> None:
        """Forward *message* to the session manager's log method.

        Falls back gracefully if the session manager does not expose a
        ``log`` callable (a ``WARNING`` is emitted via the Python logger
        instead of raising).

        Parameters
        ----------
        message:
            Plain-text log message to record.
        """
        log_fn = getattr(self._sm, "log", None)
        if callable(log_fn):
            try:
                log_fn(message)
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "CalibrationManager: session_manager.log() raised %r — "
                    "message was: %s",
                    exc,
                    message,
                )
        else:
            logger.warning(
                "CalibrationManager: session_manager has no callable 'log' "
                "attribute — message: %s",
                message,
            )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _utc_iso_now() -> str:
    """Return the current UTC time as an ISO-8601 string with timezone info.

    Example: ``'2026-11-02T17:08:47+00:00'``
    """
    return datetime.now(tz=timezone.utc).isoformat()
