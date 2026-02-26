"""
E-Waste Classifier — Inference / Testing Script
=================================================
Classifies images into 7 waste categories:
  e-waste, cardboard, glass, metal, paper, plastic, trash

Usage:
  # Predict on a single image
  python predict.py path/to/image.jpg

  # Predict on a folder of images
  python predict.py path/to/folder/

  # Predict using webcam (live)
  python predict.py --webcam

  # Predict on a sample from the test set
  python predict.py --test-samples 5
"""

import sys
import os
import random
from pathlib import Path
from ultralytics import YOLO

# ──────────────── CONFIG ────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(r"D:\Work\EV_LAB\runs\classify\runs\classify\ewaste_binary\weights\best.pt")
TEST_DATASET = BASE_DIR / "data" / "binary_dataset" / "test"
IMG_SIZE = 224
CONF_THRESHOLD = 0.25  # minimum confidence to display
# ────────────────────────────────────────

# Class descriptions for user-friendly output
CLASS_INFO = {
    "e-waste":     "⚡ Electronic waste (phones, laptops, PCBs, TVs, batteries, etc.)",
    "non-ewaste":  "♻️  Non-electronic waste (cardboard, glass, metal, paper, plastic, trash)",
}


def predict_image(model, image_path):
    """Run prediction on a single image and print results."""
    print(f"\n{'─' * 50}")
    print(f"  📷 Image: {image_path}")
    print(f"{'─' * 50}")

    results = model.predict(
        source=str(image_path),
        imgsz=IMG_SIZE,
        device=0,
        verbose=False,
    )

    r = results[0]
    probs = r.probs

    # Top-1 prediction
    top1_idx = probs.top1
    top1_conf = probs.top1conf.item()
    top1_name = r.names[top1_idx]

    print(f"\n  🏆 Prediction:  {top1_name.upper()}")
    print(f"     Confidence:  {top1_conf * 100:.1f}%")
    if top1_name in CLASS_INFO:
        print(f"     Description: {CLASS_INFO[top1_name]}")

    # Top-5 predictions
    top5_indices = probs.top5
    top5_confs = probs.top5conf.tolist()

    print(f"\n  📊 Top-5 Predictions:")
    print(f"     {'Rank':<6} {'Class':<15} {'Confidence':<12}")
    print(f"     {'─' * 35}")
    for rank, (idx, conf) in enumerate(zip(top5_indices, top5_confs), 1):
        name = r.names[idx]
        bar = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
        print(f"     {rank:<6} {name:<15} {conf * 100:>6.1f}%  {bar}")

    # E-waste detection alert
    if top1_name == "e-waste" and top1_conf > 0.5:
        print(f"\n  ⚠️  E-WASTE DETECTED! This item should be disposed")
        print(f"     of at a certified e-waste recycling facility.")

    return top1_name, top1_conf


def predict_folder(model, folder_path):
    """Run prediction on all images in a folder."""
    folder = Path(folder_path)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    images = [f for f in folder.iterdir()
              if f.is_file() and f.suffix.lower() in image_extensions]

    if not images:
        print(f"  ❌ No images found in {folder}")
        return

    print(f"\n  Found {len(images)} images in {folder}")

    results_summary = {}
    for img in images:
        name, conf = predict_image(model, img)
        results_summary[name] = results_summary.get(name, 0) + 1

    # Summary
    print(f"\n{'=' * 50}")
    print(f"  📋 SUMMARY — {len(images)} images classified")
    print(f"{'=' * 50}")
    for cls_name, count in sorted(results_summary.items(), key=lambda x: -x[1]):
        pct = count / len(images) * 100
        print(f"     {cls_name:<15} {count:>4} images  ({pct:.1f}%)")


def test_samples(model, n=5):
    """Pick random samples from the test set and predict."""
    if not TEST_DATASET.exists():
        print(f"  ❌ Test dataset not found at {TEST_DATASET}")
        print(f"     Run test.py first to prepare the dataset.")
        return

    # Gather all test images with their true labels
    all_samples = []
    for class_dir in TEST_DATASET.iterdir():
        if class_dir.is_dir():
            for img in class_dir.iterdir():
                if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    all_samples.append((img, class_dir.name))

    if not all_samples:
        print("  ❌ No test images found.")
        return

    samples = random.sample(all_samples, min(n, len(all_samples)))

    correct = 0
    total = len(samples)

    print(f"\n{'=' * 50}")
    print(f"  🧪 TESTING ON {total} RANDOM SAMPLES")
    print(f"{'=' * 50}")

    for img_path, true_label in samples:
        pred_name, pred_conf = predict_image(model, img_path)
        is_correct = pred_name == true_label
        if is_correct:
            correct += 1
        status = "✅" if is_correct else "❌"
        print(f"\n     True label: {true_label}  |  Predicted: {pred_name}  {status}")

    accuracy = correct / total * 100
    print(f"\n{'=' * 50}")
    print(f"  🎯 Test Accuracy: {correct}/{total} = {accuracy:.1f}%")
    print(f"{'=' * 50}")


def webcam_predict(model):
    """Run live predictions from webcam."""
    import cv2

    print("\n  📹 Starting webcam prediction...")
    print("     Press 'q' to quit.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ❌ Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run prediction
        results = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            device=0,
            verbose=False,
        )

        r = results[0]
        probs = r.probs
        top1_idx = probs.top1
        top1_conf = probs.top1conf.item()
        top1_name = r.names[top1_idx]

        # Draw prediction on frame
        label = f"{top1_name}: {top1_conf * 100:.1f}%"
        color = (0, 0, 255) if top1_name == "e-waste" else (0, 255, 0)

        cv2.putText(frame, label, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        if top1_name == "e-waste" and top1_conf > 0.5:
            cv2.putText(frame, "E-WASTE DETECTED!", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("E-Waste Classifier", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ──────────────── MAIN ────────────────
if __name__ == "__main__":
    # Check model exists
    if not MODEL_PATH.exists():
        print(f"  ❌ Model not found at {MODEL_PATH}")
        print(f"     Run test.py first to train the model.")
        sys.exit(1)

    print("\n🔋 E-Waste Classifier — Inference")
    print(f"   Model: {MODEL_PATH.name}")
    print(f"   Size:  {MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB")

    # Load model
    model = YOLO(str(MODEL_PATH))
    print("   ✅ Model loaded successfully!\n")

    # Parse arguments
    if len(sys.argv) < 2:
        # No arguments — run on test samples by default
        print("   No input specified. Running on 5 random test samples...\n")
        print("   Usage:")
        print("     python predict.py image.jpg        — single image")
        print("     python predict.py folder/           — all images in folder")
        print("     python predict.py --webcam          — live webcam")
        print("     python predict.py --test-samples 10 — random test samples")
        test_samples(model, n=5)

    elif sys.argv[1] == "--webcam":
        webcam_predict(model)

    elif sys.argv[1] == "--test-samples":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        test_samples(model, n=n)

    else:
        target = Path(sys.argv[1])
        if target.is_dir():
            predict_folder(model, target)
        elif target.is_file():
            predict_image(model, target)
        else:
            print(f"  ❌ Path not found: {target}")
            sys.exit(1)

    print()
