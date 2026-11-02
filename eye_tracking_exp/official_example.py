# -*- coding: utf-8 -*-
"""
Official GazeFollower Example Reproduction
===========================================

This script reproduces the official GazeFollower quick-start example
from https://github.com/GanchengZhu/GazeFollower

Run this FIRST to verify GazeFollower is correctly installed and that
your webcam and display are accessible. It walks through the full
pipeline: preview → calibrate → sample → visualise → save.

Requires
--------
- Webcam (default device index 0)
- Display (pygame window)
- gazefollower package installed (see requirements.txt)

Usage
-----
    python official_example.py

Output
------
    data/official_example_output.csv  — raw + calibrated + filtered gaze,
                                        openness, tracking state, triggers.
"""

import os
import sys
import time
import platform

# ---------------------------------------------------------------------------
# Dependency checks — fail fast with actionable messages
# ---------------------------------------------------------------------------

try:
    import pygame
    from pygame.locals import K_RETURN, K_ESCAPE, KEYDOWN, QUIT
except ImportError:
    print("ERROR: pygame is not installed.")
    print("       Fix: pip install pygame>=2.0.0")
    sys.exit(1)

try:
    from gazefollower import GazeFollower
except ImportError:
    print("ERROR: gazefollower is not installed.")
    print("       Fix: pip install gazefollower>=1.0.0")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Platform-specific configuration
# ---------------------------------------------------------------------------

# On macOS, Python's multiprocessing uses 'fork' by default which is unsafe
# with certain GUI/camera frameworks.  Disable it to avoid crashes / hangs.
USE_MULTIPROCESSING: bool = platform.system() != "Darwin"

# How long (seconds) to record gaze after calibration.
RECORDING_DURATION: float = 10.0

# Output paths (relative to this script's directory so the experiment folder
# stays self-contained regardless of the working directory).
SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(SCRIPT_DIR, "data")
OUTPUT_CSV: str = os.path.join(DATA_DIR, "official_example_output.csv")

# Display resolution — adjust to match your monitor if needed.
DISPLAY_WIDTH: int = 1280
DISPLAY_HEIGHT: int = 720

# Gaze cursor appearance
CURSOR_COLOUR: tuple[int, int, int] = (0, 255, 0)   # bright green ring
CURSOR_RADIUS: int = 30
CURSOR_THICKNESS: int = 3   # 0 = filled circle


# ===========================================================================
# Main routine
# ===========================================================================

def main() -> None:
    """Run the full GazeFollower quick-start pipeline."""

    # ------------------------------------------------------------------
    # 1. Initialise GazeFollower
    #    All arguments use library defaults; only multiprocessing is
    #    explicitly set for cross-platform safety.
    # ------------------------------------------------------------------
    print("[1/6] Initialising GazeFollower …")
    gaze_follower = GazeFollower(use_multiprocessing=USE_MULTIPROCESSING)

    try:
        # ------------------------------------------------------------------
        # 2. Camera preview
        #    Opens a live feed so you can check framing and lighting before
        #    calibration.  Close the preview window to continue.
        # ------------------------------------------------------------------
        print("[2/6] Opening camera preview — close the window to continue …")
        gaze_follower.preview()

        # ------------------------------------------------------------------
        # 3. Calibration
        #    Displays calibration targets.  Follow the on-screen instructions.
        #    A good calibration is essential for accurate gaze coordinates.
        # ------------------------------------------------------------------
        print("[3/6] Running calibration sequence …")
        gaze_follower.calibrate()

        # ------------------------------------------------------------------
        # 4. Start gaze sampling
        #    After this call the library begins estimating gaze in the
        #    background.  Timestamps, coordinates, and openness values are
        #    accumulated internally until stop_sampling() is called.
        # ------------------------------------------------------------------
        print("[4/6] Starting gaze sampling …")
        gaze_follower.start_sampling()

        # ------------------------------------------------------------------
        # 5. Stimulus presentation loop
        #    A plain grey screen is shown for RECORDING_DURATION seconds.
        #    The estimated gaze position is drawn as a green ring so you can
        #    verify tracking is working in real time.
        #
        #    Controls:
        #      ENTER  — end the recording early
        #      ESC    — abort and exit without saving
        # ------------------------------------------------------------------
        print(f"[5/6] Recording for {RECORDING_DURATION:.0f} s "
              f"(ENTER to stop early, ESC to abort) …")

        pygame.init()
        win = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
        pygame.display.set_caption("GazeFollower — Official Example")
        clock = pygame.time.Clock()

        # Optional: send a trigger to mark the start of the recording epoch.
        # Trigger codes are stored in the CSV 'trigger' column.
        gaze_follower.send_trigger(1)

        aborted: bool = False
        start: float = time.perf_counter()

        while time.perf_counter() - start < RECORDING_DURATION:
            # --- Event handling -------------------------------------------
            for ev in pygame.event.get():
                if ev.type == QUIT:
                    aborted = True
                elif ev.type == KEYDOWN:
                    if ev.key == K_ESCAPE:
                        aborted = True
                    elif ev.key == K_RETURN:
                        # Early stop — break out of the while loop cleanly.
                        break
            else:
                # Inner for-loop completed without a RETURN break; keep going.
                if aborted:
                    break

                # --- Draw stimulus ----------------------------------------
                win.fill((128, 128, 128))   # neutral grey background

                # --- Draw gaze cursor ------------------------------------
                # get_gaze_info() returns the most recent GazeInfo sample,
                # or None if no face / gaze has been detected yet.
                gaze_info = gaze_follower.get_gaze_info()

                if gaze_info is not None and gaze_info.status:
                    # Use filtered coordinates for a smoother cursor.
                    gx = int(gaze_info.filtered_gaze_coordinates[0])
                    gy = int(gaze_info.filtered_gaze_coordinates[1])

                    # Clamp to window bounds so the circle is always visible.
                    gx = max(CURSOR_RADIUS, min(DISPLAY_WIDTH - CURSOR_RADIUS, gx))
                    gy = max(CURSOR_RADIUS, min(DISPLAY_HEIGHT - CURSOR_RADIUS, gy))

                    pygame.draw.circle(
                        win,
                        CURSOR_COLOUR,
                        (gx, gy),
                        CURSOR_RADIUS,
                        CURSOR_THICKNESS,
                    )

                pygame.display.flip()
                clock.tick(60)   # cap at 60 fps to avoid busy-spinning
                continue          # go back to while condition

            # We reach here only when the inner for-loop was broken (RETURN).
            break

        # Mark end of recording epoch.
        gaze_follower.send_trigger(2)

        pygame.quit()

        if aborted:
            print("Recording aborted by user — no data will be saved.")
            return

        # ------------------------------------------------------------------
        # 6. Stop sampling and save data
        # ------------------------------------------------------------------
        print("[6/6] Stopping sampling and saving data …")
        gaze_follower.stop_sampling()

        os.makedirs(DATA_DIR, exist_ok=True)
        gaze_follower.save_data(OUTPUT_CSV)

        print(f"\nDone.  Data saved to:\n  {OUTPUT_CSV}")
        print("\nCSV columns: timestamp, raw_gaze_position_x/y, "
              "calibrated_gaze_position_x/y, filtered_gaze_position_x/y, "
              "left/right_eye_openness, tracking_status, status, event, trigger")

    except Exception as exc:
        # Broad catch so release() is always called even on unexpected errors.
        print(f"\nUnexpected error: {exc}")
        raise

    finally:
        # ------------------------------------------------------------------
        # Cleanup — always release camera and model resources.
        # ------------------------------------------------------------------
        gaze_follower.release()
        # Guard against pygame not having been initialised if an early error
        # occurred before step 5.
        if pygame.get_init():
            pygame.quit()


if __name__ == "__main__":
    main()
