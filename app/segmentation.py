import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import numpy as np
import cv2

NUM_CLASSES = 4
IMG_SIZE = 256

CLASE_MAPEO = {
    0: "Background",
    1: "Harvest",
    2: "Seedling",
    3: "Vegetative"
}

RUTA_PESOS = "app/models/ensambleA_methodA_BBOX_output.pth"

# ==============================
# Modelos base EXACTOS
# ==============================
def load_unet_efficientnet():
    return smp.Unet(
        encoder_name="timm-efficientnet-b0",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None
    )


def load_unetplusplus():
    return smp.UnetPlusPlus(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None
    )


def load_deeplabv3():
    return smp.DeepLabV3(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None
    )


# ==============================
# ENSAMBLE EXACTO AL ENTRENADO
# ==============================
class EnsembleSeg(nn.Module):
    """Versión EXACTA al código que te enviaron"""
    def __init__(self, m1, m2, m3):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

    def forward(self, x):
        with torch.no_grad():
            p1 = self.m1(x)
            p2 = self.m2(x)
            p3 = self.m3(x)

            fusion = p1 + 0.5 * p2 + 0.25 * p3
            mask_logits = fusion.squeeze(0).cpu().numpy()

            pred = np.argmax(mask_logits, axis=0).astype(np.uint8)

            etiquetas, conteos = np.unique(pred, return_counts=True)
            conteo_dict = dict(zip(etiquetas, conteos))
            planta = {k: v for k, v in conteo_dict.items() if k != 0}

            etiqueta_dom = 0 if not planta else max(planta, key=planta.get)
            clase_dom = CLASE_MAPEO.get(etiqueta_dom)

            bbox = [0, 0, 0, 0]

            if etiqueta_dom != 0:
                mask_bin = (pred == etiqueta_dom).astype(np.uint8) * 255
                contornos, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contornos:
                    c = max(contornos, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(c)
                    bbox = [int(x), int(y), int(x + w), int(y + h)]

            return {
                "clase_dom": clase_dom,
                "bbox": bbox,
                "etiqueta_id": int(etiqueta_dom)
            }


# ==============================
# CARGA DEL MODELO
# ==============================
def load_segmentation_model(device):
    m1 = load_unet_efficientnet()
    m2 = load_unetplusplus()
    m3 = load_deeplabv3()

    model = EnsembleSeg(m1, m2, m3).to(device)

    state = torch.load(RUTA_PESOS, map_location=device)
    model.load_state_dict(state, strict=True)

    model.eval()
    return model


# ==============================
# PREDICT — versión adaptada a tu backend
# ==============================
from torchvision import transforms

def predict_segmentation(model, img, device="cpu"):
    model.eval()

    # Preprocesamiento NORMAL, como requiere tu ensemble
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    inp = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(inp)   # out = dict con clase_dom, bbox, etiqueta_id

    # Asegurar tipo JSON serializable
    return {
        "clase_dominante": out["clase_dom"],
        "bbox": out["bbox"],
        "etiqueta_id": int(out["etiqueta_id"])
    }
