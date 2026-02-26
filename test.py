"""
E-Waste Binary Classifier Training Script
==========================================
Trains a YOLOv8s-cls image classifier with 2 classes:
  - e-waste      (~8,014 images)
  - non-ewaste   (~7,852 images)

Optimized for RTX 3050 (4 GB VRAM).
"""

from pathlib import Path

# ──────────────────────── CONFIG ────────────────────────
BASE_DIR = Path(__file__).resolve().parent
BINARY_DATASET_DIR = BASE_DIR / "data" / "binary_dataset"

EPOCHS = 10
IMG_SIZE = 256
BATCH_SIZE = 32
WORKERS = 8
MODEL_NAME = "yolov8s-cls.pt"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# ─────────────────────────────────────────────────────────


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def show_distribution():
    print("\n  📊 Class distribution:")
    print(f"     {'Class':<15} {'Train':>7} {'Val':>7} {'Test':>7}")
    print(f"     {'─' * 40}")
    for cls_name in ["e-waste", "non-ewaste"]:
        train_n = count_images(BINARY_DATASET_DIR / "train" / cls_name)
        val_n = count_images(BINARY_DATASET_DIR / "val" / cls_name)
        test_n = count_images(BINARY_DATASET_DIR / "test" / cls_name)
        print(f"     {cls_name:<15} {train_n:>7} {val_n:>7} {test_n:>7}")


def train():
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("  TRAINING BINARY E-WASTE CLASSIFIER")
    print(f"  Model : {MODEL_NAME}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  ImgSz : {IMG_SIZE}")
    print(f"  Batch : {BATCH_SIZE}")
    print(f"  AMP   : True (FP16 for RTX 3050)")
    print("=" * 60 + "\n")

    model = YOLO(MODEL_NAME)

    results = model.train(
        data=str(BINARY_DATASET_DIR),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=0,
        amp=True,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        weight_decay=0.01,
        patience=5,
        project="runs/classify",
        name="ewaste_binary",
        exist_ok=True,
        verbose=True,
    )
    return results


def validate():
    from ultralytics import YOLO

    best_path = Path("runs/classify/ewaste_binary/weights/best.pt")
    if not best_path.exists():
        alt = Path(r"D:\Work\EV_LAB\runs\classify\runs\classify\ewaste_binary\weights\best.pt")
        if alt.exists():
            best_path = alt
        else:
            print("  [WARN] Best model not found — skipping validation.")
            return

    print("\n" + "=" * 60)
    print("  VALIDATING BEST MODEL")
    print("=" * 60 + "\n")

    model = YOLO(str(best_path))
    val_metrics = model.val(
        data=str(BINARY_DATASET_DIR),
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=0,
    )

    print("\n  📊 Validation Results:")
    print(f"     Top-1 Accuracy: {val_metrics.top1:.4f}")
    print(f"     Top-5 Accuracy: {val_metrics.top5:.4f}")


# ──────────────────────── MAIN ────────────────────────
if __name__ == "__main__":
    print("\n🔋 E-Waste Binary Classification Pipeline")
    print("─" * 45)

    if not BINARY_DATASET_DIR.exists():
        print(f"\n  ❌ Dataset not found at {BINARY_DATASET_DIR}")
        exit(1)

    # Clear old YOLO cache so it re-scans
    for cache_file in BINARY_DATASET_DIR.glob("*.cache"):
        cache_file.unlink()
        print(f"  Cleared cache: {cache_file.name}")

    show_distribution()
    train()
    validate()

    print("\n🎉 Done! Model saved under runs/classify/ewaste_binary/")
    print("   Use best.pt for inference.\n")