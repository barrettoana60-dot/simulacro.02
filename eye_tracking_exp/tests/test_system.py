"""
test_system.py
==============
Automated test suite for the experimental eye-tracking system.

IMPORTANT NOTICE
----------------
All test data in this module are SYNTHETIC / ARTIFICIAL MOCK DATA.
They are used strictly to verify software mechanics, file integrity,
state transitions, and statistical edge cases.
They DO NOT represent real human eye movements, nor should they be used
to make claims about real-world tracking accuracy, precision, or performance.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from enum import Enum
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure modules package is in sys.path
_TEST_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _TEST_DIR.parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

from modules.camera_acquisition import CameraAcquisition
from modules.calibration_manager import CalibrationManager
from modules.data_recorder import DataRecorder
from modules.gaze_estimator_wrapper import GazeEstimatorWrapper
from modules.quality_assessment import QualityAssessment
from modules.session_manager import SessionManager, SessionState


# ===========================================================================
# Synthetic Mocks (Strictly Artificial Test Fixtures)
# ===========================================================================

class MockTrackingState(Enum):
    TRACKING = 0
    LOST = 1


class MockEvent(Enum):
    NONE = 0
    BLINK = 1


class MockGazeInfo:
    """Artificial gaze info data class for automated unit testing."""

    def __init__(
        self,
        status: bool = True,
        x: float = 640.0,
        y: float = 360.0,
        timestamp: float | None = None,
        trigger: int = 0,
    ) -> None:
        self.status = status
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.filtered_gaze_coordinates = (x, y) if status else (-65536.0, -65536.0)
        self.raw_gaze_coordinates = (x, y) if status else (-65536.0, -65536.0)
        self.calibrated_gaze_coordinates = (x, y) if status else (-65536.0, -65536.0)
        self.left_openness = 0.85 if status else 0.0
        self.right_openness = 0.82 if status else 0.0
        self.tracking_state = MockTrackingState.TRACKING if status else MockTrackingState.LOST
        self.event = MockEvent.NONE
        self.trigger = trigger


class MockFaceInfo:
    """Artificial face alignment data class for automated unit testing."""

    def __init__(self, detected: bool = True) -> None:
        self.status = detected
        self.can_gaze_estimation = detected


# ===========================================================================
# Test Cases
# ===========================================================================

def test_camera_acquisition_verify():
    """Verify that CameraAcquisition.verify_camera_available returns the required dictionary keys."""
    cam = CameraAcquisition(camera_index=0)
    result = cam.verify_camera_available()
    assert isinstance(result, dict)
    for key in ("available", "index", "resolution", "fps", "error"):
        assert key in result, f"Key '{key}' missing from verify_camera_available result"


def test_session_manager_creates_dirs(tmp_path: Path):
    """Verify that SessionManager creates session directory and expected files."""
    sm = SessionManager(base_path=tmp_path, participant_code="P999")
    session_dir = sm.create_session()

    assert session_dir.exists()
    assert session_dir.is_dir()
    assert (session_dir / "raw_data.csv").exists()
    assert (session_dir / "processed_data.csv").exists()
    assert (session_dir / "metadata.json").exists()
    assert (session_dir / "session.log").exists()
    assert sm.state == SessionState.RUNNING


def test_session_manager_metadata_json(tmp_path: Path):
    """Verify that metadata.json is correctly formatted with essential fields."""
    sm = SessionManager(base_path=tmp_path, participant_code="P101", config={"test_key": "val"})
    sm.create_session()
    sm.update_metadata(camera_id=0, resolution=[1920, 1080], calibration_info="completed")

    meta_file = sm.get_metadata_path()
    assert meta_file.exists()

    with open(meta_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["participant_code"] == "P101"
    assert "start_time" in data
    assert "os" in data
    assert "python_version" in data
    assert "gazefollower_version" in data
    assert data["camera_id"] == 0
    assert data["resolution"] == [1920, 1080]
    assert data["calibration_info"] == "completed"
    assert data["experiment_config"] == {"test_key": "val"}


def test_data_recorder_csv_integrity(tmp_path: Path):
    """Verify CSV header and record integrity with artificial gaze samples."""
    recorder = DataRecorder(session_path=str(tmp_path), session_id="P001")
    recorder.start_recording()

    fake_face = MockFaceInfo(detected=True)
    fake_gaze = MockGazeInfo(status=True, x=500.0, y=300.0, timestamp=1000.123)
    recorder.record_sample(fake_face, fake_gaze)
    recorder.stop_recording()

    raw_csv = tmp_path / "raw_data.csv"
    assert raw_csv.exists()
    with open(raw_csv, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) >= 2  # Header + 1 row
    assert "session_id" in lines[0]
    assert "raw_x" in lines[0]
    assert "filtered_x" in lines[0]
    assert "P001" in lines[1]
    assert "500.0" in lines[1]


def test_data_recorder_invalid_sample(tmp_path: Path):
    """Verify that samples with status=False are correctly identified as invalid."""
    recorder = DataRecorder(session_path=str(tmp_path), session_id="P001")
    recorder.start_recording()

    fake_face = MockFaceInfo(detected=False)
    fake_gaze = MockGazeInfo(status=False)
    recorder.record_sample(fake_face, fake_gaze)
    recorder.stop_recording()

    assert recorder.get_sample_count() == 1
    assert recorder.get_invalid_count() == 1


def test_data_recorder_statistics(tmp_path: Path):
    """Verify calculation of recording statistics (e.g. data loss percentage)."""
    recorder = DataRecorder(session_path=str(tmp_path), session_id="P001")
    recorder.start_recording()

    fake_face = MockFaceInfo(detected=True)
    # Add 10 valid samples
    for _ in range(10):
        recorder.record_sample(fake_face, MockGazeInfo(status=True))
    # Add 5 invalid samples
    for _ in range(5):
        recorder.record_sample(fake_face, MockGazeInfo(status=False))

    recorder.stop_recording()
    stats = recorder.get_statistics()

    assert stats["sample_count"] == 15
    assert stats["invalid_count"] == 5
    # 5 / 15 * 100 = 33.33%
    assert math.isclose(stats["data_loss_pct"], 33.33, abs_tol=0.1)


def test_quality_assessment_sample_rate():
    """Verify calculation of effective sample rate from synthetic valid samples."""
    qa = QualityAssessment()
    # 30 valid samples over 3.0 seconds = 10.0 Hz
    t0 = 1000.0
    for i in range(30):
        qa.add_sample(timestamp=t0 + i * 0.1, gaze_x=500.0, gaze_y=300.0, is_valid=True, processing_time_ms=5.0)

    rate = qa.compute_sample_rate(elapsed_seconds=3.0)
    assert rate is not None
    assert math.isclose(rate, 10.0, abs_tol=0.01)


def test_quality_assessment_data_loss():
    """Verify calculation of data loss proportion."""
    qa = QualityAssessment()
    # 8 valid + 2 invalid = 20.0% loss
    for i in range(8):
        qa.add_sample(timestamp=100.0 + i, gaze_x=500.0, gaze_y=300.0, is_valid=True, processing_time_ms=5.0)
    for i in range(2):
        qa.add_sample(timestamp=108.0 + i, gaze_x=0.0, gaze_y=0.0, is_valid=False, processing_time_ms=5.0)

    loss = qa.compute_data_loss()
    assert loss is not None
    assert math.isclose(loss, 20.0, abs_tol=0.01)


def test_quality_assessment_no_accuracy_without_targets():
    """Verify that compute_accuracy refuses to generate accuracy without real target events."""
    qa = QualityAssessment()
    qa.add_sample(timestamp=1.0, gaze_x=500.0, gaze_y=300.0, is_valid=True, processing_time_ms=5.0)

    # Calling compute_accuracy with empty targets must raise ValueError
    with pytest.raises(ValueError, match="requires at least one target event"):
        qa.compute_accuracy([])


def test_quality_assessment_stability():
    """Verify calculation of tracking stability (coordinate dispersion) on synthetic fixations."""
    qa = QualityAssessment()
    # 50 samples closely clustered around (640, 360)
    for i in range(50):
        noise = (i % 3 - 1) * 0.5  # -0.5, 0, 0.5 px
        qa.add_sample(timestamp=i * 0.033, gaze_x=640.0 + noise, gaze_y=360.0 + noise, is_valid=True, processing_time_ms=4.0)

    stability = qa.compute_stability(window_size=30)
    assert stability != {}
    assert "std_x_px" in stability
    assert "std_y_px" in stability
    assert stability["std_x_px"] < 1.0
    assert stability["std_y_px"] < 1.0


def test_gaze_estimator_wrapper_invalid_sample():
    """Verify that GazeEstimatorWrapper handles invalid gaze samples properly."""
    fake_gf = MagicMock()
    fake_qa = MagicMock()
    wrapper = GazeEstimatorWrapper(gaze_follower_instance=fake_gf, quality_assessment_instance=fake_qa)

    invalid_gaze = MockGazeInfo(status=False)
    wrapper.on_gaze_sample(MockFaceInfo(False), invalid_gaze)

    current = wrapper.get_current_gaze()
    assert current is not None
    assert current["is_valid"] is False
    assert current["x"] is None
    assert current["y"] is None
    assert wrapper.is_tracking() is False


def test_calibration_manager_tracks_state():
    """Verify that CalibrationManager tracks calibrated state without premature assumptions."""
    fake_gf = MagicMock()
    fake_sm = MagicMock()
    cm = CalibrationManager(gaze_follower_instance=fake_gf, session_manager=fake_sm)

    assert cm.is_calibrated() is False
    info = cm.get_calibration_info()
    assert info["completed"] is False
    assert info["attempt_count"] == 0


def test_session_log_writes(tmp_path: Path):
    """Verify that SessionManager.log writes timestamped messages to session.log."""
    sm = SessionManager(base_path=tmp_path, participant_code="P042")
    sm.create_session()
    sm.log("Test log entry message for verification", level="INFO")
    sm.finalize_session()

    log_path = sm.get_log_path()
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert "Test log entry message for verification" in content


def test_cleanup_after_failure(tmp_path: Path):
    """Verify that SessionManager handles errors and finalizes cleanly after failure."""
    sm = SessionManager(base_path=tmp_path, participant_code="P099")
    sm.create_session()
    sm.mark_error(reason="Simulated hardware connection failure")
    assert sm.state == SessionState.ERROR

    # finalize_session should not crash even when session is in ERROR state
    sm.finalize_session()
    assert sm.state in (SessionState.COMPLETED, SessionState.ERROR)
