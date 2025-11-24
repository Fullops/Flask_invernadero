import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

CLASSES = ["Harvest Stage", "Seedling Stage", "Vegetative Stage"]
MODEL_PATH = "app/models/ensambleA_SoftVoting_3_modelos.pth"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# ARQUITECTURAS BASE
# ============================================================
def load_architecture(arch, num_classes=3):
    if arch == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(512, num_classes)

    elif arch == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(768, num_classes)

    elif arch == "mobilenetv3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(1280, num_classes)

    return model


# ============================================================
# ENSAMBLE
# ============================================================
class EnsembleClf(nn.Module):
    def __init__(self, m1, m2, m3):
        super().__init__()
        self.resnet = m1
        self.convnext = m2
        self.mobilenet = m3

    def forward(self, x):
        with torch.no_grad():
            p1 = torch.softmax(self.resnet(x), dim=1)
            p2 = torch.softmax(self.convnext(x), dim=1)
            p3 = torch.softmax(self.mobilenet(x), dim=1)
            return torch.mean(torch.stack([p1, p2, p3]), dim=0)


# ============================================================
# CARGA DEL MODELO COMPLETO
# ============================================================
def load_classifier(device="cpu"):
    m1 = load_architecture("resnet34")
    m2 = load_architecture("convnext_tiny")
    m3 = load_architecture("mobilenetv3_large")

    model = EnsembleClf(m1, m2, m3).to(device)

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    return model


# ============================================================
# PREDICCIÓN
# ============================================================
def predict_classifier(model, img: Image.Image, device="cpu"):
    tensor = transform(img).unsqueeze(0).to(device)
    preds = model(tensor)

    prob, idx = torch.max(preds, 1)
    return CLASSES[idx.item()], float(prob.item())
