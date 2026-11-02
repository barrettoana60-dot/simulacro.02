"""
modules/__init__.py
===================
Interface pública para o pacote ``eye_tracking_exp.modules``.

Exporta todas as classes principais da camada experimental:
- SessionManager, SessionState: gerenciamento de sessões e metadados
- CameraAcquisition: verificação e diagnóstico de câmera pré-inicialização
- CalibrationManager: wrapper do fluxo de calibração do GazeFollower
- GazeEstimatorWrapper: recepção e despacho de coordenadas de olhar em tempo real
- TrackingInterface: interface visual Pygame para acompanhamento experimental
- DataRecorder: gravação paralela de dados brutos e processados
- QualityAssessment: métricas de qualidade (sample rate, perda, estabilidade, acurácia, precisão)
"""

from __future__ import annotations

from .session_manager import SessionManager, SessionState
from .camera_acquisition import CameraAcquisition
from .calibration_manager import CalibrationManager, CalibrationResult
from .gaze_estimator_wrapper import GazeEstimatorWrapper
from .tracking_interface import TrackingInterface
from .data_recorder import DataRecorder
from .quality_assessment import QualityAssessment

__all__ = [
    "SessionManager",
    "SessionState",
    "CameraAcquisition",
    "CalibrationManager",
    "CalibrationResult",
    "GazeEstimatorWrapper",
    "TrackingInterface",
    "DataRecorder",
    "QualityAssessment",
]
