FROM python:3.11-slim

WORKDIR /app

# Install gdown to download from Google Drive
RUN pip install --no-cache-dir gdown

# Download model at build time
RUN gdown "1upoYABMr3LvLXLq3ZZw0La1K0SlbPQoX" -O tooth_model.pth

# Install CPU-only PyTorch (~800MB instead of ~2.5GB with CUDA)
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    timm \
    pillow \
    python-multipart \
    "numpy<2"

# Copy only what's needed
COPY main.py .

EXPOSE $PORT

CMD uvicorn main:app --host 0.0.0.0 --port $PORT
