import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

MODEL_NAME = "IDEA-Research/grounding-dino-tiny"

# on garde le modele charge entre les appels pour eviter de le recharger
model = None
processor = None
device = None


def get_device():
    global device
    if device is None:
        if torch.cuda.is_available(): # pour les bon pc comme celui de l'isir ce sera encore plus rapide >:D
            device = "cuda"
        else:
            device = "cpu"
    return device


def load_model():
    global model, processor
    if model is None:
        dev = get_device()
        print(f"[groundingdino] chargement {MODEL_NAME}")
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_NAME).to(dev)
        model.eval()
        print(f"[groundingdino] pret")
    return model, processor


def detect_contours(image, labels_en, box_threshold=0.3, text_threshold=0.25):
    # image
    # labels_en: liste de labels anglais (groundingDINO marche mieux en anglais)
    # retourne: [{"label": str, "contour": [x_min, y_min, x_max, y_max], "score": float}]
    vus = set()
    labels_clean = []
    for label in labels_en:
        l = (label or "").strip().lower()
        if l and l not in vus:
            vus.add(l)
            labels_clean.append(l)
    if not labels_clean:
        return []

    mdl, proc = load_model()
    dev = get_device()
    # GroundingDINO attend les labels separes par des points
    text = ". ".join(labels_clean) + "."
    inputs = proc(images=image, text=text, return_tensors="pt").to(dev)
    with torch.no_grad():
        outputs = mdl(**inputs)
    results = proc.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )
    if not results:
        return []

    detections = []
    for box, label, score in zip(results[0]["boxes"], results[0]["labels"], results[0]["scores"]):
        detections.append({
            "label": str(label).strip().lower(),
            "contour": [float(v) for v in box.tolist()],
            "score": float(score),
        })
    return detections
