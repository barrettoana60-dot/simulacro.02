"""
experiment.py
=============
Main orchestration script for the eye-tracking experiment.

Procedure
---------
1.  Verify camera availability (quick open-and-release, no persistent hold).
2.  Initialise a :class:`~modules.session_manager.SessionManager` session.
3.  Construct a :class:`GazeFollower` instance (``use_multiprocessing=False``
    on macOS to avoid ``spawn`` issues).
4.  Optional camera preview via ``gf.preview()``.
5.  Calibrate via ``gf.calibrate()``.
6.  Initialise the pygame interface window (GazeFollower closes its own
    window after preview/calibration; we open ours afterwards).
7.  Set up a :class:`~modules.data_recorder.DataRecorder` and wire it as a
    GazeFollower subscriber.
8.  Run the main event/render loop:
    - ``R`` / on-screen button  →  start recording (``gf.start_sampling()``)
    - ``S`` / on-screen button  →  stop recording  (``gf.stop_sampling()``)
    - ``V`` key                 →  run validation sequence
    - ``ESC`` / ``Q``           →  cleanup and quit
9.  On stop: save GazeFollower CSV + DataRecorder CSVs.
10. On quit: finalise session, release GazeFollower resources.

macOS notes
-----------
*  ``use_multiprocessing=False`` is mandatory on macOS (``spawn`` start method
   breaks OpenCV camera capture in child processes).
*  GazeFollower's ``preview()`` and ``calibrate()`` each create and destroy
   their own pygame window.  After either returns we re-create our interface
   window so that the display is always owned by the currently active pygame
   instance.

Usage (CLI)
-----------
::

    python -m eye_tracking_exp.experiment              # auto participant code
    python -m eye_tracking_exp.experiment -p P007      # explicit code
    python -m eye_tracking_exp.experiment -c 1         # secondary camera
    python -m eye_tracking_exp.experiment --skip-preview --skip-calibration

Or run directly::

    python eye_tracking_exp/experiment.py --help
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Resolve the package root so relative imports work when the script is run
# directly (``python eye_tracking_exp/experiment.py``).
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

# ---------------------------------------------------------------------------
# macOS multiprocessing guard — must be decided *before* GazeFollower import.
# ---------------------------------------------------------------------------
_IS_MACOS: bool = platform.system().lower() == "darwin"
_USE_MULTIPROCESSING: bool = not _IS_MACOS  # False on macOS, True elsewhere

# ---------------------------------------------------------------------------
# Relative imports from the modules package
# ---------------------------------------------------------------------------
from modules.session_manager import SessionManager, SessionState  # noqa: E402
from modules.data_recorder import DataRecorder  # noqa: E402

# ---------------------------------------------------------------------------
# Third-party imports (GazeFollower, pygame, OpenCV)
# ---------------------------------------------------------------------------
try:
    from gazefollower import GazeFollower  # type: ignore[import]
except ImportError as _gf_err:
    raise ImportError(
        "GazeFollower is not installed.  "
        "Install it with: pip install gazefollower"
    ) from _gf_err

try:
    import pygame  # type: ignore[import]
except ImportError as _pg_err:
    raise ImportError(
        "pygame is required for the experiment interface.  "
        "Install it with: pip install pygame"
    ) from _pg_err

try:
    import cv2  # type: ignore[import]
except ImportError as _cv_err:
    raise ImportError(
        "OpenCV is required for camera verification.  "
        "Install it with: pip install opencv-python-headless"
    ) from _cv_err

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UI constants
# ---------------------------------------------------------------------------

_FPS: int = 60
_BG_COLOR: Tuple[int, int, int] = (15, 15, 25)           # near-black navy
_HUD_COLOR: Tuple[int, int, int] = (180, 220, 255)        # cool blue
_CURSOR_COLOR: Tuple[int, int, int] = (50, 255, 120)       # bright green
_CURSOR_INVALID_COLOR: Tuple[int, int, int] = (255, 80, 80)  # red
_BTN_IDLE: Tuple[int, int, int] = (40, 60, 90)
_BTN_HOVER: Tuple[int, int, int] = (60, 100, 160)
_BTN_ACTIVE: Tuple[int, int, int] = (30, 160, 80)
_CURSOR_RADIUS: int = 18
_VALIDATION_POINT_RADIUS: int = 20
_VALIDATION_DWELL_S: float = 1.5   # seconds per validation point

_VALIDATION_POINTS_NORM: Tuple[Tuple[float, float], ...] = (
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
)


# ===========================================================================
# EyeTrackingExperiment
# ===========================================================================

class EyeTrackingExperiment:
    """Full lifecycle manager for a single eye-tracking experiment session.

    Parameters
    ----------
    participant_code:
        Explicit participant identifier (e.g. ``"P003"``).  When *None*,
        :class:`~modules.session_manager.SessionManager` auto-generates the
        next sequential code.
    camera_index:
        OpenCV camera device index.  ``0`` is usually the built-in webcam.
    screen_size:
        ``(width, height)`` in pixels of the pygame interface window.
    config:
        Optional mapping of experiment-configuration keys that will be stored
        in ``metadata.json`` alongside session data.
    skip_preview:
        When ``True``, skip the ``gf.preview()`` step (useful in headless or
        automated testing contexts).
    skip_calibration:
        When ``True``, skip the ``gf.calibrate()`` step.  **For testing only**
        — gaze data without calibration will not be screen-accurate.
    """

    def __init__(
        self,
        participant_code: Optional[str] = None,
        camera_index: int = 0,
        screen_size: Tuple[int, int] = (1920, 1080),
        config: Optional[Dict[str, Any]] = None,
        skip_preview: bool = False,
        skip_calibration: bool = False,
    ) -> None:
        self._participant_code = participant_code
        self._camera_index = camera_index
        self._screen_size = screen_size
        self._config = config or {}
        self._skip_preview = skip_preview
        self._skip_calibration = skip_calibration

        # Core objects — initialised lazily
        self._session: Optional[SessionManager] = None
        self._gf: Optional[GazeFollower] = None
        self._recorder: Optional[DataRecorder] = None

        # Pygame state
        self._screen: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._font_large: Optional[pygame.font.Font] = None
        self._font_small: Optional[pygame.font.Font] = None

        # Experiment runtime state
        self._running: bool = False
        self._recording: bool = False
        self._latest_gaze_info: Any = None   # GazeInfo or None

        # Validation state
        self._validation_active: bool = False

        logger.info(
            "EyeTrackingExperiment created — participant=%s  camera=%d  "
            "multiprocessing=%s  macOS=%s",
            participant_code or "(auto)",
            camera_index,
            _USE_MULTIPROCESSING,
            _IS_MACOS,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full experiment procedure.

        This is the single public entry point.  It runs the complete
        lifecycle — camera check, session init, GazeFollower setup,
        preview, calibration, main loop, and cleanup — handling all
        exceptions gracefully so that ``cleanup()`` is always called.
        """
        try:
            if not self._initialize_system():
                logger.error("System initialisation failed.  Aborting.")
                return

            self._run_main_loop()
            self._save_session()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received — shutting down.")

        except Exception:  # pylint: disable=broad-except
            logger.exception("Unhandled exception during experiment:")
            if self._session is not None:
                self._session.mark_error(reason="Unhandled exception — see session.log")

        finally:
            self.cleanup()

    # ------------------------------------------------------------------
    # Initialisation pipeline
    # ------------------------------------------------------------------

    def _initialize_system(self) -> bool:
        """Run the full initialisation pipeline.

        Steps
        -----
        1. Verify camera.
        2. Create session directory.
        3. Construct GazeFollower.
        4. Run camera preview (unless ``skip_preview`` is set).
        5. Run calibration (unless ``skip_calibration`` is set).
        6. Open the pygame interface window.
        7. Attach gaze subscriber and data recorder.

        Returns
        -------
        bool
            ``True`` if every step succeeded; ``False`` otherwise.
        """
        # --- Step 1: camera verification --------------------------------
        if not self._verify_camera():
            return False

        # --- Step 2: session initialisation -----------------------------
        try:
            self._session = SessionManager(
                participant_code=self._participant_code,
                config=self._config,
            )
            session_path = self._session.create_session()
            logger.info("Session path: %s", session_path)
        except Exception:  # pylint: disable=broad-except
            logger.exception("Failed to initialise session:")
            return False

        # --- Step 3: GazeFollower construction --------------------------
        try:
            self._gf = GazeFollower(
                use_multiprocessing=_USE_MULTIPROCESSING,
            )
            logger.info("GazeFollower constructed (multiprocessing=%s).", _USE_MULTIPROCESSING)
        except Exception:  # pylint: disable=broad-except
            logger.exception("Failed to construct GazeFollower:")
            self._session.mark_error("GazeFollower construction failed")
            return False

        # --- Step 4: camera preview (optional) --------------------------
        if not self._skip_preview:
            try:
                logger.info("Opening camera preview window …")
                self._gf.preview()
                logger.info("Preview closed.")
            except Exception:  # pylint: disable=broad-except
                logger.exception("Camera preview raised an exception:")
                # Non-fatal; continue to calibration.

        # --- Step 5: calibration (optional) -----------------------------
        if not self._skip_calibration:
            try:
                logger.info("Running calibration sequence …")
                self._gf.calibrate()
                logger.info("Calibration complete.")
                self._session.update_metadata(calibration_info="completed")
            except Exception:  # pylint: disable=broad-except
                logger.exception("Calibration raised an exception:")
                self._session.mark_error("Calibration failed")
                return False
        else:
            logger.warning("Calibration SKIPPED — gaze accuracy will be uncalibrated.")
            self._session.update_metadata(calibration_info="skipped")

        # --- Step 6: interface window ------------------------------------
        # GazeFollower closes its pygame window after preview/calibrate, so
        # we re-initialise pygame here to own the display ourselves.
        try:
            self._init_interface()
        except Exception:  # pylint: disable=broad-except
            logger.exception("Failed to initialise pygame interface:")
            self._session.mark_error("pygame interface init failed")
            return False

        # --- Step 7: gaze subscriber + data recorder --------------------
        try:
            self._setup_gaze_pipeline()
        except Exception:  # pylint: disable=broad-except
            logger.exception("Failed to set up gaze pipeline:")
            self._session.mark_error("Gaze pipeline setup failed")
            return False

        # Persist hardware info
        self._session.update_metadata(
            camera_id=self._camera_index,
            resolution=list(self._screen_size),
        )

        logger.info("System initialisation complete.")
        return True

    # ------------------------------------------------------------------

    def _verify_camera(self) -> bool:
        """Open the camera briefly to confirm it is available, then release it.

        Returns
        -------
        bool
            ``True`` if the camera is accessible; ``False`` otherwise.
        """
        logger.info("Verifying camera index %d …", self._camera_index)
        cap = cv2.VideoCapture(self._camera_index)
        try:
            if not cap.isOpened():
                logger.error(
                    "Camera index %d could not be opened.  "
                    "Check that the device is connected and not in use.",
                    self._camera_index,
                )
                return False
            ret, _ = cap.read()
            if not ret:
                logger.error("Camera opened but failed to read a frame.")
                return False
            logger.info("Camera %d verified successfully.", self._camera_index)
            return True
        finally:
            cap.release()

    # ------------------------------------------------------------------

    def _init_interface(self) -> None:
        """Initialise (or re-initialise) the pygame display and fonts.

        Called after GazeFollower's preview/calibrate windows have closed
        so that our window is the sole pygame display surface.
        """
        if pygame.get_init():
            pygame.quit()

        pygame.init()
        pygame.display.set_caption(
            f"Eye Tracking — {self._session.participant_code}"  # type: ignore[union-attr]
            if self._session else "Eye Tracking"
        )

        flags = pygame.NOFRAME if _IS_MACOS else 0
        self._screen = pygame.display.set_mode(self._screen_size, flags)
        self._clock = pygame.time.Clock()
        self._font_large = pygame.font.SysFont("monospace", 22, bold=True)
        self._font_small = pygame.font.SysFont("monospace", 16)
        logger.info("pygame interface window initialised (%dx%d).", *self._screen_size)

    # ------------------------------------------------------------------

    def _setup_gaze_pipeline(self) -> None:
        """Wire up the gaze subscriber and DataRecorder.

        The subscriber callback ``_on_gaze_sample`` is registered with
        GazeFollower and updates ``self._latest_gaze_info`` in real-time.
        The DataRecorder is added as a *second* subscriber so both consumers
        operate on every sample.
        """
        assert self._gf is not None
        assert self._session is not None

        session_path = str(self._session.get_session_path())
        session_id = self._session.participant_code

        self._recorder = DataRecorder(
            session_path=session_path,
            session_id=session_id,
        )

        # Register the live-preview subscriber (updates the HUD cursor)
        self._gf.add_subscriber(self._on_gaze_sample)
        # Register the DataRecorder subscriber
        self._gf.add_subscriber(self._recorder.record_sample)

        logger.info("Gaze subscriber and DataRecorder registered.")

    # ------------------------------------------------------------------
    # Subscriber callback
    # ------------------------------------------------------------------

    def _on_gaze_sample(self, face_info: Any, gaze_info: Any) -> None:
        """GazeFollower subscriber — store the latest gaze information.

        This callback is invoked by GazeFollower on every sample.  It only
        updates the cached reference; heavy processing must *not* happen here
        because this may be called from a background thread.

        Parameters
        ----------
        face_info:
            Face-detection payload from GazeFollower (may be ``None``).
        gaze_info:
            :class:`GazeInfo` object or ``None`` when gaze cannot be estimated.
        """
        self._latest_gaze_info = gaze_info

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run_main_loop(self) -> None:
        """Event-driven render loop.

        Handles keyboard/window events, draws the HUD and gaze cursor, and
        dispatches recording / validation commands.

        Key bindings
        ------------
        ``R``   — start recording (``gf.start_sampling()`` + DataRecorder)
        ``S``   — stop recording  (``gf.stop_sampling()``  + DataRecorder)
        ``V``   — run validation sequence
        ``ESC`` / ``Q`` — quit the experiment
        """
        assert self._screen is not None
        assert self._clock is not None
        assert self._gf is not None

        self._running = True
        logger.info("Entering main experiment loop.  Press R to record, S to stop, V to validate, Q/ESC to quit.")

        while self._running:
            # ---- event handling ----------------------------------------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False

                elif event.type == pygame.KEYDOWN:
                    key = event.key
                    if key in (pygame.K_ESCAPE, pygame.K_q):
                        self._running = False

                    elif key == pygame.K_r:
                        self._start_recording()

                    elif key == pygame.K_s:
                        self._stop_recording()

                    elif key == pygame.K_v:
                        if not self._recording:
                            logger.warning(
                                "Validation requested but recording is not active. "
                                "Start recording first (R)."
                            )
                        else:
                            self._run_validation_sequence()

            if not self._running:
                break

            # ---- render ------------------------------------------------
            self._screen.fill(_BG_COLOR)
            self._draw_hud()
            self._draw_gaze_cursor()
            pygame.display.flip()
            self._clock.tick(_FPS)

        logger.info("Main loop exited.")

    # ------------------------------------------------------------------
    # Recording control
    # ------------------------------------------------------------------

    def _start_recording(self) -> None:
        """Begin gaze data acquisition.

        Calls ``gf.start_sampling()`` and activates the DataRecorder.
        Idempotent — safe to call when already recording.
        """
        if self._recording:
            logger.warning("_start_recording() called while already recording — ignored.")
            return

        assert self._gf is not None
        assert self._recorder is not None

        try:
            self._recorder.start_recording()
            self._gf.start_sampling()
            self._recording = True
            logger.info("Recording STARTED.")
            if self._session:
                self._session.log("Recording started.", level="INFO")
        except Exception:  # pylint: disable=broad-except
            logger.exception("Failed to start recording:")

    # ------------------------------------------------------------------

    def _stop_recording(self) -> None:
        """Stop gaze data acquisition and save collected data.

        Calls ``gf.stop_sampling()`` and flushes the DataRecorder.
        Idempotent — safe to call when not recording.
        """
        if not self._recording:
            logger.warning("_stop_recording() called while not recording — ignored.")
            return

        assert self._gf is not None
        assert self._recorder is not None

        try:
            self._gf.stop_sampling()
            self._recorder.stop_recording()
            self._recording = False

            stats = self._recorder.get_statistics()
            logger.info(
                "Recording STOPPED — samples=%d  invalid=%d  data_loss=%.1f%%",
                stats["sample_count"],
                stats["invalid_count"],
                stats["data_loss_pct"],
            )
            if self._session:
                self._session.log(
                    f"Recording stopped. samples={stats['sample_count']} "
                    f"data_loss={stats['data_loss_pct']:.1f}%",
                    level="INFO",
                )

            # Persist GazeFollower's own CSV into the session directory.
            session_path = str(self._session.get_session_path())  # type: ignore[union-attr]
            gf_csv_path = str(Path(session_path) / "gf_raw_data.csv")
            self._gf.save_data(gf_csv_path)
            logger.info("GazeFollower CSV saved to: %s", gf_csv_path)

        except Exception:  # pylint: disable=broad-except
            logger.exception("Failed to stop recording:")

    # ------------------------------------------------------------------
    # Validation sequence
    # ------------------------------------------------------------------

    def _run_validation_sequence(self) -> None:
        """Display a 9-point validation grid and collect gaze accuracy data.

        For each validation point the participant fixates the target dot for
        ``_VALIDATION_DWELL_S`` seconds while gaze coordinates are collected.
        A trigger is sent via ``gf.send_trigger()`` at the onset of each point.

        The sequence runs synchronously within the main loop; the display
        is updated every frame throughout.
        """
        assert self._screen is not None
        assert self._clock is not None
        assert self._gf is not None

        logger.info("Validation sequence started (%d points).", len(_VALIDATION_POINTS_NORM))
        if self._session:
            self._session.log("Validation sequence started.", level="INFO")

        self._validation_active = True
        w, h = self._screen_size

        for idx, (nx, ny) in enumerate(_VALIDATION_POINTS_NORM, start=1):
            px = int(nx * w)
            py = int(ny * h)

            # Notify downstream via trigger
            try:
                self._gf.send_trigger(idx)
            except Exception:  # pylint: disable=broad-except
                logger.warning("send_trigger(%d) failed.", idx)

            dwell_start = time.perf_counter()

            while time.perf_counter() - dwell_start < _VALIDATION_DWELL_S:
                # Event handling — allow quit during validation
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._running = False
                    elif event.type == pygame.KEYDOWN and event.key in (
                        pygame.K_ESCAPE, pygame.K_q
                    ):
                        self._running = False

                if not self._running:
                    break

                # Render validation screen
                self._screen.fill(_BG_COLOR)

                # Draw all target points (dimly) except current
                for j, (nx2, ny2) in enumerate(_VALIDATION_POINTS_NORM, start=1):
                    if j == idx:
                        continue
                    pygame.draw.circle(
                        self._screen,
                        (60, 60, 80),
                        (int(nx2 * w), int(ny2 * h)),
                        _VALIDATION_POINT_RADIUS // 2,
                    )

                # Draw current target point
                progress = min(
                    (time.perf_counter() - dwell_start) / _VALIDATION_DWELL_S, 1.0
                )
                ring_radius = int(_VALIDATION_POINT_RADIUS * (1.0 - progress * 0.4))
                pygame.draw.circle(
                    self._screen, (255, 220, 50), (px, py), _VALIDATION_POINT_RADIUS
                )
                pygame.draw.circle(
                    self._screen, (255, 255, 255), (px, py), ring_radius, 2
                )

                # HUD overlay
                if self._font_small:
                    label = self._font_small.render(
                        f"Validation  {idx}/{len(_VALIDATION_POINTS_NORM)}  — look at the dot",
                        True,
                        _HUD_COLOR,
                    )
                    self._screen.blit(label, (10, 10))

                self._draw_gaze_cursor()
                pygame.display.flip()
                self._clock.tick(_FPS)

            if not self._running:
                break

        self._validation_active = False
        logger.info("Validation sequence complete.")
        if self._session:
            self._session.log("Validation sequence complete.", level="INFO")

    # ------------------------------------------------------------------
    # HUD & cursor rendering
    # ------------------------------------------------------------------

    def _draw_hud(self) -> None:
        """Render the heads-up display overlay onto ``self._screen``.

        Displays participant code, recording state, sample count, and the
        last known gaze coordinates.
        """
        if self._screen is None or self._font_small is None:
            return

        gaze_info = self._latest_gaze_info
        is_valid = gaze_info is not None and bool(getattr(gaze_info, "status", False))

        # Build HUD lines
        participant = self._session.participant_code if self._session else "—"
        rec_state = "● REC" if self._recording else "○ IDLE"
        sample_count = (
            self._recorder.get_sample_count() if self._recorder else 0
        )

        if is_valid and gaze_info.filtered_gaze_coordinates is not None:
            gx, gy = gaze_info.filtered_gaze_coordinates
            gaze_str = f"Gaze: ({gx:.0f}, {gy:.0f})"
        else:
            gaze_str = "Gaze: --"

        lines = [
            f"Participant : {participant}",
            f"State       : {rec_state}",
            f"Samples     : {sample_count}",
            gaze_str,
            "",
            "[R] Record   [S] Stop   [V] Validate   [Q] Quit",
        ]

        y_offset = 10
        for line in lines:
            color = (255, 80, 80) if "REC" in line and self._recording else _HUD_COLOR
            surf = self._font_small.render(line, True, color)
            self._screen.blit(surf, (10, y_offset))
            y_offset += surf.get_height() + 3

    # ------------------------------------------------------------------

    def _draw_gaze_cursor(self) -> None:
        """Draw a circle at the current filtered gaze position.

        Uses ``filtered_gaze_coordinates`` when available.  The cursor is
        rendered in green for valid gaze and red when tracking is invalid.
        """
        if self._screen is None:
            return

        gaze_info = self._latest_gaze_info
        is_valid = gaze_info is not None and bool(getattr(gaze_info, "status", False))

        if is_valid and gaze_info.filtered_gaze_coordinates is not None:
            gx, gy = gaze_info.filtered_gaze_coordinates
            pos = (int(gx), int(gy))
            color = _CURSOR_COLOR
        else:
            # Draw a small indicator at an off-screen sentinel when invalid
            cx, cy = self._screen_size
            pos = (cx - 30, cy - 30)
            color = _CURSOR_INVALID_COLOR

        pygame.draw.circle(self._screen, color, pos, _CURSOR_RADIUS, 3)
        # Inner dot
        pygame.draw.circle(self._screen, color, pos, 4)

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def _save_session(self) -> None:
        """Persist session-level summary data after the main loop ends.

        If recording is still active when the loop exits (e.g. user pressed
        ESC without stopping), ``_stop_recording()`` is called first to
        ensure no data is lost.
        """
        if self._recording:
            logger.info("Recording was still active at exit — stopping now.")
            self._stop_recording()

        report = self._generate_report()

        if self._session is not None:
            self._session.update_metadata(**report)
            logger.info("Session report written to metadata.json.")

    # ------------------------------------------------------------------

    def _generate_report(self) -> Dict[str, Any]:
        """Compile a summary report of the session.

        Returns
        -------
        dict
            Keys: ``sample_count``, ``invalid_count``, ``data_loss_pct``,
            ``mean_processing_time_ms``.  All values default to ``0`` when
            the DataRecorder was never started.
        """
        if self._recorder is not None:
            stats = self._recorder.get_statistics()
        else:
            stats = {
                "sample_count": 0,
                "invalid_count": 0,
                "data_loss_pct": 0.0,
                "mean_processing_time_ms": 0.0,
            }

        report: Dict[str, Any] = {
            "sample_count": stats["sample_count"],
            "invalid_count": stats["invalid_count"],
            "data_loss_pct": stats["data_loss_pct"],
            "mean_processing_time_ms": stats["mean_processing_time_ms"],
        }
        return report

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Release all resources and finalise the session.

        Safe to call multiple times — all operations are idempotent.

        Order of teardown:
        1. Ensure recording is stopped and data flushed.
        2. Remove GazeFollower subscribers.
        3. Call ``gf.release()``.
        4. Quit pygame.
        5. Finalise or mark-error the session.
        """
        logger.info("Cleanup initiated.")

        # 1. Stop recording if still active
        if self._recording:
            try:
                self._stop_recording()
            except Exception:  # pylint: disable=broad-except
                logger.exception("Error while stopping recording during cleanup:")

        # 2. Remove subscribers
        if self._gf is not None:
            for cb in (self._on_gaze_sample,):
                try:
                    self._gf.remove_subscriber(cb)
                except Exception:  # pylint: disable=broad-except
                    pass  # Subscriber may already have been removed

            if self._recorder is not None:
                try:
                    self._gf.remove_subscriber(self._recorder.record_sample)
                except Exception:  # pylint: disable=broad-except
                    pass

        # 3. Release GazeFollower
        if self._gf is not None:
            try:
                self._gf.release()
                logger.info("GazeFollower released.")
            except Exception:  # pylint: disable=broad-except
                logger.exception("Error releasing GazeFollower:")
            finally:
                self._gf = None

        # 4. Quit pygame
        try:
            if pygame.get_init():
                pygame.quit()
                logger.info("pygame quit.")
        except Exception:  # pylint: disable=broad-except
            logger.exception("Error quitting pygame:")

        # 5. Finalise session
        if self._session is not None:
            try:
                if self._session.state not in (SessionState.COMPLETED, SessionState.ERROR):
                    self._session.finalize_session()
                    logger.info(
                        "Session finalised: %s", self._session.get_session_path()
                    )
            except Exception:  # pylint: disable=broad-except
                logger.exception("Error finalising session:")

        logger.info("Cleanup complete.")


# ===========================================================================
# CLI entry point
# ===========================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="experiment",
        description="Eye-tracking experiment runner (GazeFollower backend).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python experiment.py\n"
            "  python experiment.py -p P007 -c 1\n"
            "  python experiment.py --skip-preview --skip-calibration\n"
        ),
    )

    parser.add_argument(
        "-p", "--participant",
        metavar="CODE",
        default=None,
        help=(
            "Participant code to use for this session (e.g. P007).  "
            "When omitted the next sequential code is auto-generated."
        ),
    )

    parser.add_argument(
        "-c", "--camera",
        metavar="INDEX",
        type=int,
        default=0,
        help="Camera device index passed to OpenCV (default: 0).",
    )

    parser.add_argument(
        "--skip-preview",
        action="store_true",
        default=False,
        help="Skip the camera preview step.",
    )

    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        default=False,
        help=(
            "Skip the calibration step.  FOR TESTING ONLY — "
            "gaze data will not be screen-accurate without calibration."
        ),
    )

    return parser


def main() -> None:
    """CLI entry point — parse arguments and run the experiment."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.skip_calibration:
        logger.warning(
            "--skip-calibration is set.  Gaze data will NOT be screen-accurate.  "
            "Use only for testing."
        )

    experiment = EyeTrackingExperiment(
        participant_code=args.participant,
        camera_index=args.camera,
        skip_preview=args.skip_preview,
        skip_calibration=args.skip_calibration,
    )
    experiment.run()


if __name__ == "__main__":
    main()
