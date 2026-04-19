import io
import torch
import timm
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from torchvision import transforms

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Tooth Classification API", version="1.0.0")

# ── Classes ────────────────────────────────────────────────────────────────────
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

# ── Helpers ────────────────────────────────────────────────────────────────────
def predict(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top_idx   = int(probs.argmax())
    top_label = CLASSES[top_idx]
    top_conf  = float(probs[top_idx])

    all_probs = {CLASSES[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)}
    clinical  = RECOMMENDATIONS[top_label]

    return {
        "prediction": top_label,
        "confidence": round(top_conf, 4),
        "severity": clinical["severity"],
        "recommendations": clinical["recommendations"],
        "probabilities": all_probs,
    }

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Tooth Classification API is running 🦷"}

@app.post("/predict")
async def classify(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    image_bytes = await file.read()
    result = predict(image_bytes)
    return result
