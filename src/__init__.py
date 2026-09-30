"""
LibraryGuard - AI-based prohibited activity detection for CCTV-style video.

Package layout::

    src.detection     YOLOv8 person/object/pose detection
    src.tracking      ByteTrack-style anonymous multi-person tracking
    src.behavior      Temporal behavior detectors + event aggregation engine
    src.preprocessing Video I/O and dataset tooling
    src.models        Trainable temporal classifier (PyTorch GRU)
    src.evaluation    Metrics + plots
    src.reporting     Event logs, CSV/JSON export, video annotation
    src.utils         Config loading, constants, geometry
    src.pipeline      End-to-end orchestration
"""

__version__ = "1.0.0"

__all__ = ["__version__"]