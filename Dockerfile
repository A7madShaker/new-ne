# Python slim base - much smaller than default
FROM python:3.11-slim

WORKDIR /app

# Install CPU-only torch first (avoids downloading CUDA = saves ~3GB)
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install the rest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY main.py .
COPY tooth_model.pth .

EXPOSE $PORT

CMD uvicorn main:app --host 0.0.0.0 --port $PORT
