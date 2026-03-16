FROM python:3.11-slim

WORKDIR /app

# Install deps first (cached layer - only rebuilds if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Cloud Run requires port 8080
ENV PORT=8080
EXPOSE 8080

# Single worker is REQUIRED for WebSocket sticky sessions
# Multiple workers = sessions randomly routed to different workers = dropped connections
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--ws", "websockets"]
