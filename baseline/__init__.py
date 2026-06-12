"""
Baseline memorization / contamination detection methods for comparison with PEARL.

RQ1 in the paper compares PEARL against:
  - CoDeC  (gray-box): in-context confidence comparison
  - CDD    (black-box): peakedness of edit-distance distribution
  - ACR    (white-box): adversarial compression ratio
"""
from .codec import CoDeC, ModelHandler, contamination_detection_pipeline
from .cdd import CDDDetector
from .acr import ACRDetector, ACRConfig

__all__ = [
    "CoDeC",
    "ModelHandler",
    "contamination_detection_pipeline",
    "CDDDetector",
    "ACRDetector",
    "ACRConfig",
]
