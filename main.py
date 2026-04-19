import io
import base64
import torch
import timm
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from torchvision import transforms
import torch.nn.functional as F

app = FastAPI(title="Tooth Classification API", version="1.0.0")

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

device = torch.device("cpu")
model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=NUM_CLASSES)
state_dict = torch.load("tooth_model.pth", map_location=device, weights_only=False)
model.load_state_dict(state_dict)
model.eval()

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def jet_colormap(cam: np.ndarray) -> np.ndarray:
    """Pure numpy JET colormap — no OpenCV needed."""
    cam = np.clip(cam, 0, 1)
    r = np.clip(1.5 - np.abs(cam * 4 - 3), 0, 1)
    g = np.clip(1.5 - np.abs(cam * 4 - 2), 0, 1)
    b = np.clip(1.5 - np.abs(cam * 4 - 1), 0, 1)
    heatmap = np.stack([r, g, b], axis=-1)
    return (heatmap * 255).astype(np.uint8)


def get_cam(tensor: torch.Tensor, class_idx: int) -> np.ndarray:
    activations = None

    def hook(module, input, output):
        nonlocal activations
        activations = output.detach()

    handle = model.conv_head.register_forward_hook(hook)
    with torch.no_grad():
        model(tensor)
    handle.remove()

    classifier_weights = model.classifier.weight[class_idx].detach()
    cam = (classifier_weights[:, None, None] * activations[0]).sum(dim=0)
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.numpy()


def apply_heatmap(original_image: Image.Image, cam: np.ndarray) -> str:
    orig_w, orig_h = original_image.size

    # Resize CAM to original image size using PIL
    cam_pil = Image.fromarray((cam * 255).astype(np.uint8))
    cam_pil = cam_pil.resize((orig_w, orig_h), Image.BILINEAR)
    cam_resized = np.array(cam_pil) / 255.0

    # Apply JET colormap
    heatmap = jet_colormap(cam_resized)

    # Blend with original
    orig_array = np.array(original_image.convert("RGB"))
    blended = (orig_array * 0.5 + heatmap * 0.5).astype(np.uint8)

    # Encode to base64
    result_img = Image.fromarray(blended)
    buffer = io.BytesIO()
    result_img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def run_predict(image_bytes: bytes, with_gradcam: bool = False):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
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
        cam = get_cam(tensor, top_idx)
        result["gradcam_image"] = f"data:image/jpeg;base64,{apply_heatmap(image, cam)}"

    return result


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
