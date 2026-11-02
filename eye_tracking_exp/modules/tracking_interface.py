"""
tracking_interface.py
---------------------
Experimental Pygame interface for the eye-tracking session.

This module owns the *experiment* window that stays open throughout a session
(stimulus presentation, recording, etc.).  It is completely separate from the
temporary fullscreen windows that GazeFollower opens for ``preview()`` and
``calibrate()`` — those come and go before this interface is initialised.

Typical lifecycle
-----------------
::

    interface = TrackingInterface(screen_size=(1920, 1080))
    surface   = interface.initialize()

    # GazeFollower preview / calibration happen here (their own windows)
    gf.preview()
    gf.calibrate()

    # Experiment loop
    gf.start_sampling()
    running = True
    while running:
        action = interface.handle_events()
        if action == 'quit':
            running = False
        elif action == 'record':
            gf.start_sampling()
        elif action == 'stop':
            gf.stop_sampling()

        gaze_info = gf.get_gaze_info()
        state = { ... }
        interface.draw_status_panel(surface, state)
        if gaze_info and gaze_info.status:
            gx, gy = gaze_info.filtered_gaze_coordinates
            interface.draw_gaze_cursor(surface, gx, gy)
        pygame.display.flip()

    interface.cleanup()
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import pygame

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_WHITE   = (255, 255, 255)
_BLACK   = (  0,   0,   0)
_GREEN   = (  0, 220,   0)
_RED     = (220,   0,   0)
_YELLOW  = (255, 210,   0)
_GREY    = (120, 120, 120)
_PANEL_BG = (0, 0, 0, 160)   # RGBA — semi-transparent black HUD background


# ---------------------------------------------------------------------------
# TrackingInterface
# ---------------------------------------------------------------------------


class TrackingInterface:
    """Pygame interface for the experimental eye-tracking session.

    Handles window creation, HUD rendering, gaze cursor drawing,
    validation sequences, and keyboard event routing.

    This class does **not** interact with GazeFollower's internal pygame
    windows (``preview()`` / ``calibrate()``).  Those are temporary and
    expected to be finished before :meth:`initialize` is called.

    Parameters
    ----------
    screen_size:
        ``(width, height)`` in pixels for the experiment window.
        Defaults to ``(1920, 1080)``.
    show_camera_feed:
        Reserved for future use.  When ``True``, a camera thumbnail
        could be drawn in the HUD.  Currently stored but not rendered.
    """

    # Fonts are module-level singletons once pygame is initialised
    _FONT_LARGE: Optional[pygame.font.Font] = None
    _FONT_MEDIUM: Optional[pygame.font.Font] = None
    _FONT_SMALL: Optional[pygame.font.Font] = None

    def __init__(
        self,
        screen_size: Tuple[int, int] = (1920, 1080),
        show_camera_feed: bool = False,
    ) -> None:
        self._screen_size: Tuple[int, int] = screen_size
        self._show_camera_feed: bool = show_camera_feed

        # Set by initialize()
        self._surface: Optional[pygame.Surface] = None

        # FPS tracking — rolling window of frame timestamps
        self._frame_times: Deque[float] = deque(maxlen=60)
        self._fps: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> pygame.Surface:
        """Initialise pygame and create the experiment window.

        Must be called once before any drawing methods.  GazeFollower's
        own ``preview()`` / ``calibrate()`` windows should be finished
        **before** this is called so their fullscreen context is already
        released.

        Returns
        -------
        pygame.Surface
            The primary display surface.

        Raises
        ------
        RuntimeError
            If pygame cannot initialise the display.
        """
        pygame.init()

        w, h = self._screen_size
        try:
            self._surface = pygame.display.set_mode(
                (w, h), pygame.NOFRAME
            )
        except pygame.error as exc:
            raise RuntimeError(
                f"TrackingInterface: could not create display ({w}x{h}): {exc}"
            ) from exc

        pygame.display.set_caption("Eye Tracking Experiment")
        pygame.mouse.set_visible(False)

        # Initialise shared fonts (once per process)
        if TrackingInterface._FONT_LARGE is None:
            TrackingInterface._FONT_LARGE  = pygame.font.SysFont("monospace", 28, bold=True)
            TrackingInterface._FONT_MEDIUM = pygame.font.SysFont("monospace", 20)
            TrackingInterface._FONT_SMALL  = pygame.font.SysFont("monospace", 16)

        logger.info("TrackingInterface initialised (%dx%d).", w, h)
        return self._surface

    def cleanup(self) -> None:
        """Release all pygame resources.

        Safe to call even if :meth:`initialize` was never called.
        """
        pygame.quit()
        logger.info("TrackingInterface cleaned up.")

    # ------------------------------------------------------------------
    # HUD / drawing
    # ------------------------------------------------------------------

    def draw_status_panel(self, surface: pygame.Surface, state: Dict[str, Any]) -> None:
        """Draw a semi-transparent HUD panel in the top-left corner.

        Parameters
        ----------
        surface:
            Pygame surface to draw on.
        state:
            Dictionary of current experiment state.  Recognised keys:

            ``connected`` (*bool*) — camera connected flag.
            ``calibrated`` (*bool*) — calibration done flag.
            ``tracking_active`` (*bool*) — sampling is running.
            ``gaze_x`` (*float | None*) — current gaze X coordinate.
            ``gaze_y`` (*float | None*) — current gaze Y coordinate.
            ``fps`` (*float | None*) — override FPS value (uses internal
                counter when absent).
            ``invalid_samples`` (*int*) — cumulative invalid sample count.

            Missing keys are treated as ``None`` / ``False``.
        """
        font_m = TrackingInterface._FONT_MEDIUM
        font_s = TrackingInterface._FONT_SMALL
        assert font_m is not None and font_s is not None

        connected       = bool(state.get("connected", False))
        calibrated      = bool(state.get("calibrated", False))
        tracking_active = bool(state.get("tracking_active", False))
        gaze_x          = state.get("gaze_x")
        gaze_y          = state.get("gaze_y")
        fps             = state.get("fps", self._fps)
        invalid_samples = int(state.get("invalid_samples", 0))

        # ----- build text lines ----------------------------------------
        lines: List[Tuple[str, Tuple[int, int, int]]] = [
            (
                f"Camera:      {'CONNECTED'    if connected       else 'DISCONNECTED'}",
                _GREEN if connected else _RED,
            ),
            (
                f"Calibration: {'DONE'         if calibrated      else 'PENDING'}",
                _GREEN if calibrated else _YELLOW,
            ),
            (
                f"Tracking:    {'ACTIVE'       if tracking_active else 'INACTIVE'}",
                _GREEN if tracking_active else _GREY,
            ),
        ]

        if gaze_x is not None and gaze_y is not None:
            lines.append((f"Gaze:        ({int(gaze_x):4d}, {int(gaze_y):4d}) px", _WHITE))
        else:
            lines.append(("Gaze:        N/A", _GREY))

        lines.append((f"FPS:         {fps:5.1f}", _WHITE))
        lines.append(
            (
                f"Invalid:     {invalid_samples:4d} samples",
                _YELLOW if invalid_samples > 0 else _WHITE,
            )
        )

        separator = ("─" * 26, _GREY)
        key_hints: List[Tuple[str, Tuple[int, int, int]]] = [
            ("Press R : Start Recording", _WHITE),
            ("Press S : Stop",            _WHITE),
            ("Press V : Validate",        _WHITE),
            ("Press Q : Quit",            _WHITE),
        ]

        all_lines = lines + [separator] + key_hints

        # ----- measure panel size --------------------------------------
        padding    = 12
        line_h     = font_m.get_linesize() + 2
        sep_h      = font_s.get_linesize() + 2
        panel_h    = padding * 2 + len(all_lines) * line_h
        max_width  = max(font_m.size(text)[0] for text, _ in all_lines) + padding * 2
        panel_w    = max(max_width, 260)
        panel_rect = pygame.Rect(10, 10, panel_w, panel_h)

        # ----- semi-transparent background -----------------------------
        panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel_surf.fill(_PANEL_BG)
        surface.blit(panel_surf, (panel_rect.x, panel_rect.y))

        # ----- render text lines ---------------------------------------
        y = panel_rect.y + padding
        for text, colour in all_lines:
            if text.startswith("─"):
                rendered = font_s.render(text, True, colour)
                h = sep_h
            else:
                rendered = font_m.render(text, True, colour)
                h = line_h
            surface.blit(rendered, (panel_rect.x + padding, y))
            y += h

    def draw_gaze_cursor(
        self,
        surface: pygame.Surface,
        gaze_x: float,
        gaze_y: float,
        color: Tuple[int, int, int] = (0, 255, 0),
        radius: int = 20,
    ) -> None:
        """Draw a hollow circle at the current gaze position.

        Parameters
        ----------
        surface:
            Pygame surface to draw on.
        gaze_x:
            Horizontal gaze coordinate in pixels.
        gaze_y:
            Vertical gaze coordinate in pixels.
        color:
            RGB colour tuple for the cursor ring.  Defaults to green.
        radius:
            Radius of the cursor circle in pixels.
        """
        cx = int(round(gaze_x))
        cy = int(round(gaze_y))

        # Outer ring
        pygame.draw.circle(surface, color, (cx, cy), radius, 2)
        # Small centre dot
        pygame.draw.circle(surface, color, (cx, cy), max(2, radius // 8))

    def draw_validation_target(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        target_num: int,
        total: int,
    ) -> None:
        """Draw a validation target at ``(x, y)``.

        Parameters
        ----------
        surface:
            Pygame surface to draw on.
        x:
            Horizontal centre of the target in pixels.
        y:
            Vertical centre of the target in pixels.
        target_num:
            1-based index of the current target (shown as label).
        total:
            Total number of targets in the sequence (shown as label).
        """
        font_s = TrackingInterface._FONT_SMALL
        assert font_s is not None

        outer_r = 24
        inner_r = 6

        # Outer red ring
        pygame.draw.circle(surface, _RED,   (x, y), outer_r, 3)
        # White fill
        pygame.draw.circle(surface, _WHITE, (x, y), inner_r)
        # Centre dot
        pygame.draw.circle(surface, _RED,   (x, y), inner_r // 2)

        # Label
        label = font_s.render(f"{target_num}/{total}", True, _WHITE)
        lw, lh = label.get_size()
        surface.blit(label, (x - lw // 2, y + outer_r + 4))

    def draw_calibration_instruction(self, surface: pygame.Surface, message: str) -> None:
        """Render a centred instruction message on *surface*.

        Parameters
        ----------
        surface:
            Pygame surface to draw on.
        message:
            Instruction string.  A ``\\n`` within the string inserts a
            line break.
        """
        font_l = TrackingInterface._FONT_LARGE
        assert font_l is not None

        sw, sh = surface.get_size()
        lines  = message.split("\n")
        line_h = font_l.get_linesize() + 4
        total_h = line_h * len(lines)

        start_y = sh // 2 - total_h // 2
        for i, line in enumerate(lines):
            rendered = font_l.render(line, True, _WHITE)
            rw = rendered.get_width()
            surface.blit(rendered, (sw // 2 - rw // 2, start_y + i * line_h))

    # ------------------------------------------------------------------
    # Validation sequence
    # ------------------------------------------------------------------

    def run_validation_sequence(
        self,
        gaze_follower: Any,
        n_targets: int = 9,
        dwell_time_ms: int = 2000,
    ) -> List[Dict[str, Any]]:
        """Display validation targets and collect gaze samples for each.

        Targets are placed on a 3×3 grid with a 5 % inset from each edge
        (distinct from calibration points) so they do not overlap with the
        calibration grid.

        The method blocks until all targets have been visited.  Press **Q**
        during validation to abort early; the returned list will contain
        only the targets that were shown.

        Parameters
        ----------
        gaze_follower:
            Live :class:`GazeFollower` instance.  Must already be sampling
            (``start_sampling()`` called).
        n_targets:
            Number of targets.  Must be a perfect square ≤ 9.
            Defaults to ``9``.
        dwell_time_ms:
            Duration in milliseconds to collect samples per target.

        Returns
        -------
        list of dict
            One entry per shown target::

                {
                    "target_x":           int,
                    "target_y":           int,
                    "timestamp_start":    float,   # time.perf_counter()
                    "timestamp_end":      float,
                    "samples_during_dwell": list,  # raw GazeInfo objects (or None)
                }
        """
        if self._surface is None:
            raise RuntimeError(
                "TrackingInterface.run_validation_sequence() called before initialize()."
            )

        import math

        cols = int(math.sqrt(n_targets))
        rows = math.ceil(n_targets / cols)

        sw, sh = self._screen_size
        margin_x = int(sw * 0.05)
        margin_y = int(sh * 0.05)

        # Build grid positions
        targets: List[Tuple[int, int]] = []
        for r in range(rows):
            for c in range(cols):
                if len(targets) >= n_targets:
                    break
                tx = margin_x + c * ((sw - 2 * margin_x) // max(cols - 1, 1))
                ty = margin_y + r * ((sh - 2 * margin_y) // max(rows - 1, 1))
                targets.append((tx, ty))

        results: List[Dict[str, Any]] = []
        clock = pygame.time.Clock()

        for idx, (tx, ty) in enumerate(targets):
            samples: List[Any] = []

            # Instruction before each target
            self._surface.fill(_BLACK)
            self.draw_calibration_instruction(
                self._surface,
                f"Validation — look at the target\nTarget {idx + 1} of {n_targets}",
            )
            pygame.display.flip()
            pygame.time.wait(600)

            t_start = time.perf_counter()
            t_end_target = t_start + dwell_time_ms / 1000.0
            aborted = False

            while time.perf_counter() < t_end_target:
                # Event handling
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        aborted = True
                        break
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                        aborted = True
                        break
                if aborted:
                    break

                # Draw frame
                self._surface.fill(_BLACK)
                self.draw_validation_target(
                    self._surface, tx, ty, idx + 1, n_targets
                )

                # Overlay remaining time bar
                elapsed   = time.perf_counter() - t_start
                progress  = min(elapsed / (dwell_time_ms / 1000.0), 1.0)
                bar_w     = int(sw * 0.4 * progress)
                bar_rect  = pygame.Rect(sw // 2 - int(sw * 0.2), sh - 40, bar_w, 12)
                pygame.draw.rect(self._surface, _GREEN, bar_rect)

                pygame.display.flip()
                clock.tick(60)

                # Collect gaze sample
                gaze_info = gaze_follower.get_gaze_info()
                samples.append(gaze_info)

            t_end = time.perf_counter()
            results.append(
                {
                    "target_x":             tx,
                    "target_y":             ty,
                    "timestamp_start":      t_start,
                    "timestamp_end":        t_end,
                    "samples_during_dwell": samples,
                }
            )

            if aborted:
                logger.warning(
                    "Validation aborted by user at target %d/%d.", idx + 1, n_targets
                )
                break

        logger.info(
            "Validation complete: %d/%d targets shown.", len(results), n_targets
        )
        return results

    # ------------------------------------------------------------------
    # FPS counter
    # ------------------------------------------------------------------

    def update_fps_counter(self) -> float:
        """Record the current frame timestamp and return rolling FPS.

        Call once per frame **after** ``pygame.display.flip()``.

        Returns
        -------
        float
            Smoothed frames-per-second estimate based on up to the last
            60 frame timestamps.
        """
        now = time.perf_counter()
        self._frame_times.append(now)

        if len(self._frame_times) >= 2:
            span = self._frame_times[-1] - self._frame_times[0]
            if span > 0:
                self._fps = (len(self._frame_times) - 1) / span
        return self._fps

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def handle_events(self) -> str:
        """Process the pygame event queue and return a control token.

        Keyboard bindings
        -----------------
        ===========  ======================================================
        Key          Action token
        ===========  ======================================================
        ``Q``        ``'quit'``
        ``R``        ``'record'``
        ``S``        ``'stop'``
        ``C``        ``'recalibrate'``
        ``V``        ``'validate'``
        Window close ``'quit'``
        (other)      ``'none'``
        ===========  ======================================================

        Returns
        -------
        str
            One of ``'quit'``, ``'record'``, ``'stop'``, ``'recalibrate'``,
            ``'validate'``, or ``'none'``.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"

            if event.type == pygame.KEYDOWN:
                key = event.key
                if key == pygame.K_q or key == pygame.K_ESCAPE:
                    return "quit"
                if key == pygame.K_r:
                    return "record"
                if key == pygame.K_s:
                    return "stop"
                if key == pygame.K_c:
                    return "recalibrate"
                if key == pygame.K_v:
                    return "validate"

        return "none"

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def show_message(
        self,
        surface: pygame.Surface,
        title: str,
        message: str,
        duration_ms: int = 2000,
    ) -> None:
        """Display a centred title + message overlay for *duration_ms* ms.

        This method **blocks** for the specified duration while still
        pumping the pygame event queue to prevent the OS from marking the
        window as unresponsive.

        Parameters
        ----------
        surface:
            Pygame surface to draw on.
        title:
            Large bold heading text.
        message:
            Smaller body text shown below the title.
        duration_ms:
            How long to display the overlay in milliseconds.
        """
        font_l = TrackingInterface._FONT_LARGE
        font_m = TrackingInterface._FONT_MEDIUM
        assert font_l is not None and font_m is not None

        sw, sh = surface.get_size()

        # Background overlay
        overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        surface.blit(overlay, (0, 0))

        # Title
        title_surf = font_l.render(title, True, _WHITE)
        tw = title_surf.get_width()
        surface.blit(title_surf, (sw // 2 - tw // 2, sh // 2 - 40))

        # Body (multi-line)
        lines = message.split("\n")
        lh = font_m.get_linesize() + 2
        y_start = sh // 2 + 10
        for i, line in enumerate(lines):
            line_surf = font_m.render(line, True, _GREY)
            lw = line_surf.get_width()
            surface.blit(line_surf, (sw // 2 - lw // 2, y_start + i * lh))

        pygame.display.flip()

        # Block while still pumping events
        deadline = pygame.time.get_ticks() + duration_ms
        while pygame.time.get_ticks() < deadline:
            pygame.event.pump()
            pygame.time.wait(50)
