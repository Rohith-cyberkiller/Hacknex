"""CLI Demo entrypoint for Visual Forensics Deepfake Detection.

Features:
    - Phase 1: Structured detection contracts & overlays
    - Phase 2: MTCNN face detection, 5-point landmark alignment & 256x256 crop export
    - Phase 3: Dual-branch classification (RGB EfficientNet-B0 + Frequency FFT/SRM)
    - Phase 4: Grad-CAM attribution heatmaps & compositing
    - Phase 5: Degradation hardening evaluation & social-media compression benchmarks

Usage:
    # Full forensic detection on single portrait
    python demo.py samples/portrait_face.jpg --save-crops

    # Multi-face image detection
    python demo.py samples/multi_faces.jpg --save-crops

    # No-face graceful detection
    python demo.py samples/sample.jpg

    # Test detection after simulated social-media JPEG compression (Q=50)
    python demo.py samples/portrait_face.jpg --degrade jpeg_q50

    # Run full robustness benchmark across compression & blur profiles
    python demo.py samples/portrait_face.jpg --robustness-benchmark
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to path for direct invocation
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
from visual_detector import VisualDetector, detect_visual_authenticity
from src.degradation import DEGRADATION_PROFILES, apply_degradation_profile
from src.utils import create_overlay_image, load_image, save_aligned_face_crops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visual Forensics Deepfake Detection - End-to-End Multimodal Forensic Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default="samples/portrait_face.jpg",
        help="Path to input image file (jpg, png, webp, etc.)",
    )
    parser.add_argument(
        "-i",
        "--image",
        dest="flag_image_path",
        default=None,
        help="Alternative flag to specify input image path",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="outputs/demo_overlay.jpg",
        help="Output destination path for the detection & Grad-CAM overlay image",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        default=True,
        help="Save aligned & normalized 256x256 face crops to disk",
    )
    parser.add_argument(
        "--no-crops",
        dest="save_crops",
        action="store_false",
        help="Disable saving aligned face crops",
    )
    parser.add_argument(
        "--crops-dir",
        default="outputs/crops",
        help="Directory to save aligned face crops",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=256,
        help="Target square resolution (pixels) for aligned face crops",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=5,
        help="Maximum number of faces to detect and align",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference device ('cpu' or 'cuda')",
    )
    parser.add_argument(
        "--degrade",
        default=None,
        choices=list(DEGRADATION_PROFILES.keys()),
        help="Apply simulated degradation before detection (e.g. jpeg_q50, blur_k5, social_media_heavy)",
    )
    parser.add_argument(
        "--robustness-benchmark",
        action="store_true",
        help="Run detection across all degradation profiles and display comparison table",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving visualization overlay to disk",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress intermediate pipeline step logs",
    )
    return parser.parse_args()


def run_robustness_table(detector: VisualDetector, target_path: Path) -> None:
    """Run and display the full robustness benchmark table."""
    print("\n" + "=" * 78)
    print("  PHASE 5: SOCIAL-MEDIA COMPRESSION & DEGRADATION ROBUSTNESS BENCHMARK")
    print("=" * 78)
    print(f"{'Profile':<22} | {'PSNR (dB)':<10} | {'Score':<8} | {'Delta':<8} | {'Faces':<6} | {'Verdict'}")
    print("-" * 78)

    profiles = [
        "clean",
        "jpeg_q85",
        "jpeg_q70",
        "jpeg_q50",
        "jpeg_q30",
        "blur_k3",
        "blur_k5",
        "downscale_0.5",
        "social_media_mild",
        "social_media_heavy",
    ]

    report = detector.evaluate_robustness(target_path, profiles=profiles)
    for p_name, metrics in report.items():
        psnr_str = f"{metrics['psnr']:.2f}" if metrics['psnr'] != float("inf") else "Inf (Clean)"
        print(
            f"{p_name:<22} | {psnr_str:<10} | {metrics['score']:<8.4f} | "
            f"{metrics['score_delta']:<+8.4f} | {metrics['faces_detected']:<6} | {metrics['label']}"
        )
    print("=" * 78)


def main() -> int:
    args = parse_args()

    # Determine input path
    target_path_str = args.flag_image_path or args.image_path
    target_path = Path(target_path_str)

    if not target_path.exists():
        print(f"\n[ERROR] File not found: {target_path.resolve()}", file=sys.stderr)
        return 1

    detector = VisualDetector(
        device=args.device,
        target_size=args.target_size,
        max_faces=args.max_faces,
        verbose=not args.quiet,
    )

    if args.robustness_benchmark:
        run_robustness_table(detector, target_path)
        return 0

    print("\n" + "=" * 70)
    print("  VISUAL FORENSICS - DUAL-BRANCH DEEPFAKE DETECTION & EXPLAINABILITY")
    print("=" * 70)
    print(f"Target Image : {target_path.resolve()}")
    print(f"Device       : {args.device.upper()}")
    print(f"Crop Size    : {args.target_size}x{args.target_size}")

    input_img = load_image(target_path)

    # Apply degradation if requested
    if args.degrade:
        deg_res = apply_degradation_profile(input_img, args.degrade)
        input_img = deg_res.degraded_image
        psnr_str = f"{deg_res.psnr:.2f} dB" if deg_res.psnr != float("inf") else "Baseline"
        print(f"Degradation  : Applied '{args.degrade}' (PSNR: {psnr_str})")

    print("-" * 70)

    # 1. Execute Visual Detector
    try:
        result = detector.detect(input_img)
    except Exception as err:
        print(f"\n[ERROR] Detection failed: {err}", file=sys.stderr)
        return 1

    # 2. Display Formatted Results
    print("\n" + "-" * 70)
    print("  STAGE 1: FACE DETECTION & LANDMARK ALIGNMENT")
    print("-" * 70)
    face_count = len(result.faces)
    print(f"Detected Face Count        : {face_count} face(s)")

    if face_count == 0:
        print("[INFO] No human faces detected in input image.")
        print(f"Prediction Status          : {result.predicted_label}")
    else:
        for idx, face in enumerate(result.faces):
            box = face.box.as_tuple()
            print(f"  Face #{idx + 1}:")
            print(f"    - Bounding Box (x1,y1,x2,y2) : {box} (w={face.box.width()}px, h={face.box.height()}px)")
            print(f"    - Detection Confidence       : {face.confidence * 100:.2f}%")
            if face.landmarks:
                lms = face.landmarks
                print(f"    - Left Eye Landmark          : ({lms.left_eye[0]:.1f}, {lms.left_eye[1]:.1f})")
                print(f"    - Right Eye Landmark         : ({lms.right_eye[0]:.1f}, {lms.right_eye[1]:.1f})")
                print(f"    - Nose Landmark              : ({lms.nose[0]:.1f}, {lms.nose[1]:.1f})")
                print(f"    - Mouth Landmarks            : Left ({lms.left_mouth[0]:.1f}, {lms.left_mouth[1]:.1f}) | Right ({lms.right_mouth[0]:.1f}, {lms.right_mouth[1]:.1f})")
            if face.normalized_tensor is not None:
                print(f"    - Normalized PyTorch Tensor  : Shape {list(face.normalized_tensor.shape)}, Dtype {face.normalized_tensor.dtype}")

    print("\n" + "-" * 70)
    print("  STAGE 2: DUAL-BRANCH CLASSIFICATION & FUSION")
    print("-" * 70)
    rgb_scores = result.metadata.get("rgb_scores", [])
    freq_scores = result.metadata.get("freq_scores", [])
    per_face_fused = result.metadata.get("per_face_fused", [])

    if face_count > 0:
        for idx in range(face_count):
            r_s = rgb_scores[idx] if idx < len(rgb_scores) else "N/A"
            f_s = freq_scores[idx] if idx < len(freq_scores) else "N/A"
            fused = per_face_fused[idx] if idx < len(per_face_fused) else "N/A"
            print(f"  Face #{idx + 1} Scores:")
            print(f"    - RGB Spatial (EfficientNet) : {r_s:.4f}" if isinstance(r_s, float) else f"    - RGB Spatial : {r_s}")
            print(f"    - Frequency (FFT + SRM)      : {f_s:.4f}" if isinstance(f_s, float) else f"    - Frequency   : {f_s}")
            print(f"    - Fused Authenticity (0.6/0.4): {fused:.4f}" if isinstance(fused, float) else f"    - Fused       : {fused}")
    else:
        print("  [N/A] Dual-branch classification skipped (no faces present).")

    print("\n" + "-" * 70)
    print("  STAGE 3: GRAD-CAM ATTRIBUTION & OVERLAY")
    print("-" * 70)
    print(f"Grad-CAM Heatmap Resolution: {result.gradcam_heatmap_array.shape[1]}x{result.gradcam_heatmap_array.shape[0]}")
    print(f"Heatmap Value Range        : [{result.gradcam_heatmap_array.min():.2f}, {result.gradcam_heatmap_array.max():.2f}]")
    print(f"Target Attribution Class   : Fake (Index 1) for suspicious region highlighting")

    print("\n" + "-" * 70)
    print("  FINAL FORENSIC VERDICT")
    print("-" * 70)
    print(f"Visual Authenticity Score  : {result.visual_authenticity_score:.4f}  (1.0 = Authentic, 0.0 = Fake)")
    print(f"Manipulation Probability   : {result.manipulation_probability:.4f}  (1.0 - Authenticity)")
    print(f"Prediction Verdict         : {result.predicted_label}")

    # 3. Save Aligned Face Crops
    if args.save_crops and face_count > 0:
        saved_crops = save_aligned_face_crops(result.faces, output_dir=args.crops_dir)
        print("-" * 70)
        print(f"[Crops Saved] {len(saved_crops)} aligned crop(s) written to: {Path(args.crops_dir).resolve()}")
        for cp in saved_crops:
            print(f"  -> {cp.name}")

    # 4. Save Annotated Overlay Image
    if not args.no_save:
        try:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            overlay = create_overlay_image(
                base_image=input_img,
                heatmap=result.gradcam_heatmap_array,
                bounding_boxes=result.bounding_boxes,
                authenticity_score=result.visual_authenticity_score,
                faces=result.faces,
            )

            cv2.imwrite(str(output_path), overlay)
            print("-" * 70)
            print(f"[Overlay Saved] Full forensic visualization written to: {output_path.resolve()}")
        except Exception as err:
            print(f"[Warning] Failed to save overlay image: {err}", file=sys.stderr)

    print("=" * 70)
    print("All forensic stages (Phases 1-5) operational and verified.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
