"""Visual Forensics / Deepfake Detection package.

Phases 1-5: Face Extraction, Dual-Branch Detector (RGB + Frequency),
Grad-CAM Explainability, and Degradation Hardening.
"""

from src.types import (
    AlignedFace,
    BoundingBox,
    DetectionResult,
    FacialLandmarks,
    ImageInput,
    InputColorSpace,
    StageExecutionLog,
    VideoInput,
)
from src.face_extraction import (
    FaceDetector,
    align_face_similarity_transform,
    crop_face_bounding_box,
    normalize_face_tensor,
)
from src.rgb_branch import RGBBranch, RGBClassifier
from src.frequency_branch import (
    FrequencyBranch,
    compute_fft_magnitude,
    apply_srm_filters,
    prepare_frequency_features,
)
from src.fusion import ScoreFusion
from src.gradcam import (
    GradCAM,
    composite_heatmap_multi_face,
    generate_gradcam_for_faces,
    map_heatmap_to_frame,
)
from src.degradation import (
    DEGRADATION_PROFILES,
    DegradationResult,
    apply_all_profiles,
    apply_degradation_profile,
    compute_psnr,
    downscale_upscale,
    gaussian_blur,
    jpeg_compress,
)
from src.utils import (
    create_overlay_image,
    create_synthetic_sample_image,
    generate_mock_heatmap,
    load_image,
    save_aligned_face_crops,
    validate_image_array,
)
from src.pipeline import ForensicsPipeline

__all__ = [
    # Types
    "DetectionResult",
    "BoundingBox",
    "AlignedFace",
    "FacialLandmarks",
    "StageExecutionLog",
    "InputColorSpace",
    "ImageInput",
    "VideoInput",
    # Face Extraction
    "FaceDetector",
    "align_face_similarity_transform",
    "crop_face_bounding_box",
    "normalize_face_tensor",
    # Dual-Branch
    "RGBBranch",
    "RGBClassifier",
    "FrequencyBranch",
    "compute_fft_magnitude",
    "apply_srm_filters",
    "prepare_frequency_features",
    "ScoreFusion",
    # Explainability
    "GradCAM",
    "generate_gradcam_for_faces",
    "composite_heatmap_multi_face",
    "map_heatmap_to_frame",
    # Degradation Hardening
    "DegradationResult",
    "DEGRADATION_PROFILES",
    "jpeg_compress",
    "gaussian_blur",
    "downscale_upscale",
    "compute_psnr",
    "apply_degradation_profile",
    "apply_all_profiles",
    # Pipeline & Utils
    "ForensicsPipeline",
    "load_image",
    "validate_image_array",
    "generate_mock_heatmap",
    "create_overlay_image",
    "create_synthetic_sample_image",
    "save_aligned_face_crops",
]

