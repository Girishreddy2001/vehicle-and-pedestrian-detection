# Vehicle and Pedestrian Detection System

AI-powered vehicle and pedestrian detection using YOLOv8, deployed with Docker.

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-nano-orange.svg)

## Features

- 🚗 Real-time vehicle and pedestrian detection using YOLOv8
- 🌐 Web-based interface with Streamlit
- 🐳 Dockerized microservices architecture
- 💻 Multi-platform support (Windows, Mac Intel/ARM, Linux)

## Architecture

```
Frontend (Streamlit) → Backend API (FastAPI) → AI Service (YOLOv8)
   Port 8501              Port 8500               Port 8000
```

## Quick Start

**Prerequisites**: Docker and Docker Compose

### Using Pre-built Images (Recommended)

```bash
# Clone and start
git clone https://github.com/YOUR_USERNAME/vehicle-and-pedestrian-detection.git
cd vehicle-and-pedestrian-detection
docker-compose -f docker-compose-hub.yml up -d

# Access at http://localhost:8501
```

### Building from Source

```bash
# Clone and build
git clone https://github.com/YOUR_USERNAME/vehicle-and-pedestrian-detection.git
cd vehicle-and-pedestrian-detection
docker-compose up -d --build

# Access at http://localhost:8501
```

## Usage

1. Open `http://localhost:8501` in your browser
2. Upload a JPEG/PNG image (max 10MB)
3. View detection results with bounding boxes
4. Results saved in `ai_backend/outputs/`

**Detects**: Vehicles (car, bus, truck, motorbike, bicycle) and pedestrians

## API Endpoints

**AI Backend** (Port 8000)
- `POST /detect` - Upload image for detection

**UI Backend** (Port 8500)
- `POST /upload` - Upload via UI backend
- `GET /health` - Health check

**UI Frontend** (Port 8501)
- Web interface

## Development

### Run Locally (without Docker)
```bash
# Terminal 1: AI Backend
cd ai_backend && python app.py

# Terminal 2: UI Backend
cd ui_backend && python app.py

# Terminal 3: Frontend
cd ui_frontend && streamlit run streamlit_app.py
```

## Troubleshooting

**View logs:**
```bash
docker-compose -f docker-compose-hub.yml logs -f
```

**Stop services:**
```bash
docker-compose -f docker-compose-hub.yml down
```

**Common Issues:**
- **ARM64 error on Mac**: Images support multi-platform. Pull latest or build locally.
- **Port in use**: Change ports in `docker-compose-hub.yml`
- **Model not found**: Download `yolov8n.pt` from [Ultralytics](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)

## Docker Images

Pre-built images on Docker Hub (AMD64 + ARM64):
- `girish2809/vehicle-detection-ai-backend:latest`
- `girish2809/vehicle-detection-ui-backend:latest`
- `girish2809/vehicle-detection-ui-frontend:latest`

## Technologies Used

- **YOLOv8** - Object detection
- **FastAPI** - Backend APIs
- **Streamlit** - Web UI
- **Docker** - Containerization
- **Python 3.10** - Programming language


## Author

**Girish** | [Docker Hub: girish2809](https://hub.docker.com/u/girish2809)

---

*Built with YOLOv8, FastAPI, and Streamlit*
