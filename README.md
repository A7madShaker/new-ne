# 🦷 Tooth Classification API

FastAPI + EfficientNet-B3 model for dental condition classification.

## Classes
| Label | Condition |
|-------|-----------|
| 0 | Data caries |
| 1 | Gingivitis |
| 2 | Mouth Ulcer |
| 3 | Normal |
| 4 | Tooth Discoloration |
| 5 | cancer |
| 6 | hypodontia |

## Endpoints

### `GET /`
Health check.

### `POST /predict`
Upload an image and get a prediction.

**Request:** `multipart/form-data` with field `file` (image)

**Response:**
```json
{
  "prediction": "Gingivitis",
  "confidence": 0.9312,
  "probabilities": {
    "Data caries": 0.012,
    "Gingivitis": 0.9312,
    "Mouth Ulcer": 0.003,
    "Normal": 0.021,
    "Tooth Discoloration": 0.018,
    "cancer": 0.008,
    "hypodontia": 0.007
  }
}
```

## Run locally
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
