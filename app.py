from pathlib import Path
import uuid, time, json, cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = PROJECT_ROOT / "backend" / "checkpoints"
CAR_MODEL_PATH = CHECKPOINTS / "car_best.pt"
ROAD_MODEL_PATH = CHECKPOINTS / "road_best.pt"
API_OUTPUTS = PROJECT_ROOT / "outputs" / "api"
API_OUTPUTS.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
CORS(app)
car_model = None
road_model = None
recent_runs = []

def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")
    return YOLO(str(path))

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

def serialize_boxes(result):
    boxes = result.boxes
    names = result.names
    detections = []
    if boxes is None:
        return detections
    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        detections.append({
            "class_id": cls_id,
            "class_name": names.get(cls_id, str(cls_id)),
            "confidence": round(conf, 4),
            "bbox_xyxy": [round(v, 2) for v in box.xyxy[0].tolist()],
        })
    return detections

def summarize(detections):
    counts, confs = {}, []
    for d in detections:
        counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
        confs.append(d["confidence"])
    return {
        "count": len(detections),
        "class_counts": counts,
        "avg_confidence": round(sum(confs) / len(confs), 4) if confs else 0.0,
        "confidence_values": confs,
    }

def run_prediction(model, input_path: Path, output_name: str):
    t0 = time.time()
    results = model(str(input_path), conf=0.25)
    infer_ms = round((time.time() - t0) * 1000, 2)
    result = results[0]
    annotated = result.plot()
    output_path = API_OUTPUTS / output_name
    cv2.imwrite(str(output_path), annotated)
    detections = serialize_boxes(result)
    summary = summarize(detections)
    summary["inference_ms"] = infer_ms
    return detections, summary, output_path

def track(run_type, summary):
    recent_runs.append({"type": run_type, **summary})
    if len(recent_runs) > 20:
        del recent_runs[0]

@app.get("/")
def home():
    return "Unified Damage Detection backend is running."

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "car_model": CAR_MODEL_PATH.exists(),
        "road_model": ROAD_MODEL_PATH.exists(),
        "car_eval": read_json(CHECKPOINTS / "car_eval.json"),
        "road_eval": read_json(CHECKPOINTS / "road_eval.json")
    })

@app.get("/dashboard")
def dashboard():
    return jsonify({
        "history": recent_runs[-10:],
        "car_eval": read_json(CHECKPOINTS / "car_eval.json"),
        "road_eval": read_json(CHECKPOINTS / "road_eval.json")
    })

@app.post("/predict-car")
def predict_car():
    global car_model
    if car_model is None:
        car_model = load_model(CAR_MODEL_PATH)
    if "image" not in request.files:
        return jsonify({"error": "Upload an image with key 'image'"}), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "No image selected"}), 400
    ext = Path(f.filename).suffix.lower() or ".jpg"
    uid = uuid.uuid4().hex
    input_path = API_OUTPUTS / f"{uid}{ext}"
    f.save(str(input_path))
    detections, summary, out = run_prediction(car_model, input_path, f"{uid}_car.jpg")
    track("car", summary)
    return jsonify({"type": "car", "detections": detections, "summary": summary, "annotated_image_url": f"http://127.0.0.1:5000/outputs/{out.name}"})

@app.post("/predict-road")
def predict_road():
    global road_model
    if road_model is None:
        road_model = load_model(ROAD_MODEL_PATH)
    if "image" not in request.files:
        return jsonify({"error": "Upload an image with key 'image'"}), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "No image selected"}), 400
    ext = Path(f.filename).suffix.lower() or ".jpg"
    uid = uuid.uuid4().hex
    input_path = API_OUTPUTS / f"{uid}{ext}"
    f.save(str(input_path))
    detections, summary, out = run_prediction(road_model, input_path, f"{uid}_road.jpg")
    track("road", summary)
    return jsonify({"type": "road", "detections": detections, "summary": summary, "annotated_image_url": f"http://127.0.0.1:5000/outputs/{out.name}"})

@app.get("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(API_OUTPUTS, filename)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
