from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import torch

# =============================
# MODELOS
# =============================
from app.clasification import load_classifier, predict_classifier
from app.segmentation import load_segmentation_model, predict_segmentation


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
app = Flask(__name__)


# =============================
# CARGA DE MODELOS
# =============================
print("[CARGANDO MODELO DE CLASIFICACIÓN…]")
clf_model = load_classifier(device=DEVICE)

print("[CARGANDO MODELO DE SEGMENTACIÓN…]")
seg_model = load_segmentation_model(device=DEVICE)

print("[OK] Modelos cargados correctamente.")
# =============================


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if "image" not in data:
            return jsonify({"error": "Falta el campo 'image'"}), 400

        img_bytes = base64.b64decode(data["image"])
        img = Image.open(BytesIO(img_bytes)).convert("RGB")

        # Clasificación
        clase, prob = predict_classifier(clf_model, img, device=DEVICE)

        # Segmentación (dict)
        seg = predict_segmentation(seg_model, img, device=DEVICE)

        return jsonify({
            "clasificacion": {
                "clase": clase,
                "confianza": float(prob)
            },
            "segmentacion": seg
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
