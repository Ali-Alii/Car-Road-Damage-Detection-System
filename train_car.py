from pathlib import Path
import shutil, argparse, yaml, json, pandas as pd
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "car_damage"
RUNS_DIR = PROJECT_ROOT / "runs"
CHECKPOINTS_DIR = PROJECT_ROOT / "backend" / "checkpoints"

def ensure_dataset():
    required = [DATASET_DIR / "data.yaml", DATASET_DIR / "train", DATASET_DIR / "valid", DATASET_DIR / "test"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Car dataset missing:\n- " + "\n- ".join(missing))

def build_local_yaml():
    ensure_dataset()
    src = DATASET_DIR / "data.yaml"
    with open(src, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["path"] = str(DATASET_DIR.resolve())
    data["train"] = "train/images"
    data["val"] = "valid/images"
    data["test"] = "test/images"
    out = DATASET_DIR / "data_local.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return out

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=832)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name", default="car_damage_train_v2")
    return ap.parse_args()

def save_metrics(run_dir):
    metrics = {}
    csv_path = run_dir / "results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        metrics["final_epoch_metrics"] = df.iloc[-1].to_dict()
    (CHECKPOINTS_DIR / "car_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

def main():
    args = parse_args()
    yaml_path = build_local_yaml()
    CHECKPOINTS_DIR.mkdir(exist_ok=True)
    model = YOLO(args.model)
    model.train(
        data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project=str(RUNS_DIR), name=args.name, exist_ok=True,
        patience=20, cache=True, cos_lr=True, close_mosaic=10,
        degrees=5, translate=0.08, scale=0.2, fliplr=0.5,
        hsv_h=0.01, hsv_s=0.4, hsv_v=0.3, mixup=0.05
    )
    best_pt = RUNS_DIR / args.name / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, CHECKPOINTS_DIR / "car_best.pt")
    model = YOLO(str(CHECKPOINTS_DIR / "car_best.pt"))
    m = model.val(data=str(yaml_path), split="test", imgsz=args.imgsz, device=args.device)
    summary = {"precision": float(m.box.mp), "recall": float(m.box.mr), "map50": float(m.box.map50), "map50_95": float(m.box.map)}
    (CHECKPOINTS_DIR / "car_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_metrics(RUNS_DIR / args.name)
    print("Car validation:", summary)

if __name__ == "__main__":
    main()
