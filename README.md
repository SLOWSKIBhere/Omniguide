# OmniGuide — Real-Time Multimodal AI Screen Co-Pilot

> An AI agent that sees your screen, hears your question, and tells you exactly what to do — in real time.

Built for the **Gemini Live Agent Challenge** hackathon.

## 🔗 Live Demo
**Backend Health Check:** [OmniGuide Cloud Run Instance](https://omniguide-backend-973581476293.us-central1.run.app/health)

## 🧠 Architecture

```text
Browser (Screen Capture + Voice)
    ↓ POST /ask (JPEG frame + transcript)
Cloud Run — FastAPI Backend
    ↓
Gemini 2.0 Flash — Observer Agent
    → "APP: VS Code / TASK: debugging / FOCUS: error on line 24"
    ↓
Gemini 2.0 Flash — Guide Agent
    → "That TypeError means your variable is None. Add a null check..."
    ↓
Firestore — agent_telemetry (logs every interaction)
    ↓
Browser — displays response
🛠 Stack
AI: Google GenAI SDK + Gemini 2.0 Flash

Backend: Python FastAPI on Google Cloud Run

Database: Google Firestore

Frontend: Single HTML file (no frameworks, native Web APIs)

💻 Run Locally
Install dependencies:

Bash
pip install -r requirements.txt
Start the FastAPI backend:

Bash
uvicorn main:app --host 0.0.0.0 --port 8080
Serve the frontend in a new terminal window:

Bash
python -m http.server 8000
Then navigate to: http://localhost:8000/omniguide.html

☁️ Deploy to Cloud Run
Enable required Google Cloud services and deploy with session affinity:

Bash
gcloud services enable aiplatform.googleapis.com run.googleapis.com firestore.googleapis.com

gcloud run deploy omniguide-backend \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --session-affinity
📁 Project Structure
main.py — FastAPI server with /health and /ask endpoints.

agent.py — Two-stage Observer + Guide Gemini pipeline.

telemetry.py — Firestore interaction logging.

omniguide.html — Frontend (screen capture, voice recognition, and UI).

Dockerfile — Container configuration for Cloud Run.


### Step 3: Commit and Push
Once saved, commit the file and push it to your GitHub repository:
```bash
git add README.md
git commit -m "docs: add project README for hackathon submission"
git push origin main


