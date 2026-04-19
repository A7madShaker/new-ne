FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir gdown

RUN gdown "1upoYABMr3LvLXLq3ZZw0La1K0SlbPQoX" -O tooth_model.pth

RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    timm \
    pillow \
    python-multipart \
    "numpy<2"

COPY main.py .

EXPOSE $PORT

CMD uvicorn main:app --host 0.0.0.0 --port $PORT
