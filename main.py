import io
import base64
import torch
import timm
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from torchvision import transforms
import torch.nn.functional as F

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Tooth Classification API", version="1.0.0")

# ── Classes & Recommendations ──────────────────────────────────────────────────
CLASSES = {
    0: "Data caries",
    1: "Gingivitis",
    2: "Mouth Ulcer",
    3: "Normal",
    4: "Tooth Discoloration",
    5: "cancer",
    6: "hypodontia",
}

RECOMMENDATIONS = {
    "Data caries": {
        "severity": "Moderate",
        "recommendations": [
            "Visit a dentist for professional cleaning and treatment.",
            "Brush teeth twice daily with fluoride toothpaste.",
            "Reduce sugar and acidic food intake.",
            "Use dental floss daily to remove plaque between teeth.",
            "Consider fluoride treatments or dental sealants.",
        ],
    },
    "Gingivitis": {
        "severity": "Moderate",
        "recommendations": [
            "Schedule a dental cleaning appointment soon.",
            "Brush gently along the gumline twice a day.",
            "Use an antiseptic mouthwash daily.",
            "Floss carefully to remove plaque near gums.",
            "Avoid smoking as it worsens gum disease.",
        ],
    },
    "Mouth Ulcer": {
        "severity": "Low",
        "recommendations": [
            "Avoid spicy, acidic, or rough foods until healed.",
            "Use over-the-counter antiseptic gel for relief.",
            "Rinse with warm salt water several times a day.",
            "If ulcer persists more than 2 weeks, consult a doctor.",
            "Stay hydrated and maintain good oral hygiene.",
        ],
    },
    "Normal": {
        "severity": "None",
        "recommendations": [
            "Great! Your teeth appear healthy.",
            "Continue brushing twice daily with fluoride toothpaste.",
            "Floss daily and use mouthwash regularly.",
            "Schedule routine dental checkups every 6 months.",
            "Maintain a balanced diet low in sugar.",
        ],
    },
    "Tooth Discoloration": {
        "severity": "Low",
        "recommendations": [
            "Consult a dentist to identify the cause of discoloration.",
            "Reduce consumption of coffee, tea, and tobacco.",
            "Use whitening toothpaste as recommended by your dentist.",
            "Professional whitening treatments may be needed.",
            "Maintain good oral hygiene habits.",
        ],
    },
    "cancer": {
        "severity": "High Risk",
        "recommendations": [
            "Seek immediate consultation with an oncologist.",
            "Do not delay — early diagnosis greatly improves outcomes.",
            "A biopsy may be required to confirm the diagnosis.",
            "Avoid tobacco and alcohol completely.",
            "Follow up with your dentist and specialist regularly.",
        ],
    },
    "hypodontia": {
        "severity": "Moderate",
        "recommendations": [
            "Consult an orthodontist or prosthodontist for evaluation.",
            "Dental implants or bridges may be recommended.",
            "Early treatment in children prevents future complications.",
            "Maintain excellent oral hygiene around existing teeth.",
            "Regular X-rays help monitor tooth development.",
        ],
    },
}

NUM_CLASSES = len(CLASSES)

# ── Model ──────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=NUM_CLASSES)
state_dict = torch.load("tooth_model.pth", map_location=device, weights_only=False)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ── Transform ──────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Grad-CAM ───────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        # Hook on the last conv block of EfficientNet-B3
        target_layer = model.blocks[-1][-1].conv_pwl
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor, class_idx):
        self.model.zero_grad()
        output = self.model(tensor)
        output[0, class_idx].backward()

        # Global average pooling over gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.squeeze().cpu().numpy()


gradcam = GradCAM(model)


def apply_heatmap(original_image: Image.Image, cam: np.ndarray) -> str:
    """Overlay Grad-CAM heatmap on original image and return base64 string."""
    import cv2

    # Resize CAM to original image size
    orig_w, orig_h = original_image.size
    cam_resized = np.uint8(255 * cam)
    cam_resized = cv2.resize(cam_resized, (orig_w, orig_h))

    # Apply colormap
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend with original
    orig_array = np.array(original_image.convert("RGB"))
    blended = cv2.addWeighted(orig_array, 0.5, heatmap, 0.5, 0)

    # Encode to base64
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode("utf-8")


# ── Helpers ────────────────────────────────────────────────────────────────────
def run_predict(image_bytes: bytes, with_gradcam: bool = False):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    if with_gradcam:
        tensor.requires_grad_()

    with torch.set_grad_enabled(with_gradcam):
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top_idx = int(probs.argmax())
    top_label = CLASSES[top_idx]
    top_conf = float(probs[top_idx])
    all_probs = {CLASSES[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)}
    clinical = RECOMMENDATIONS[top_label]

    result = {
        "prediction": top_label,
        "confidence": round(top_conf, 4),
        "severity": clinical["severity"],
        "recommendations": clinical["recommendations"],
        "probabilities": all_probs,
    }

    if with_gradcam:
        cam = gradcam.generate(tensor, top_idx)
        result["gradcam_image"] = f"data:image/jpeg;base64,{apply_heatmap(image, cam)}"

    return result


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Tooth Classification API is running 🦷"}


@app.post("/predict")
async def classify(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    return run_predict(await file.read(), with_gradcam=False)


@app.post("/predict-gradcam")
async def classify_gradcam(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    return run_predict(await file.read(), with_gradcam=True)
