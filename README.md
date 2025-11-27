# Planet Sentinel: Geospatial MLOps Pipeline

**Planet Sentinel** is a prototype automated MLOps pipeline for deforestation detection using Sentinel-2 satellite imagery. It demonstrates an end-to-end workflow from data ingestion to model deployment and monitoring.

## Features
*   **Pillar 1 (Data)**: Automated ingestion from Google Earth Engine, labeling support, and DVC versioning.
*   **Pillar 2 (Training)**: Lightweight Siamese U-Net model, MLflow tracking, and automated training on Kaggle.
*   **Pillar 3 (Deployment)**: FastAPI service, Docker containerization, and automated deployment via GitHub Actions.
*   **Pillar 4 (Monitoring)**: Data drift detection triggering automated retraining.

---

## 1. Environment Setup

### Prerequisites
*   **OS**: Windows (with WSL2 recommended for Docker) or Linux.
*   **Python**: 3.9 or higher.
*   **Docker**: Installed and running (Docker Desktop for Windows).
*   **Accounts**:
    *   Google Earth Engine (GEE) - [Sign up](https://earthengine.google.com/)
    *   Kaggle (for training) - [Sign up](https://www.kaggle.com/)
    *   GitHub (for CI/CD)

### Installation
Install the required Python packages:

```bash
pip install -r requirements.txt
pip install dvc mlflow kaggle rasterio
```

### Credentials Configuration
1.  **Google Earth Engine**:
    ```bash
    earthengine authenticate
    ```
    Follow the browser prompts to generate the token.

2.  **Kaggle**:
    Place your `kaggle.json` API token in `C:\Users\<USERNAME>\.kaggle\` (Windows) or `~/.kaggle/` (Linux/Mac).

---

## 2. Infrastructure: GitHub Self-Hosted Runner (Windows)

To enable the automated deployment workflow, you must set up a self-hosted runner on your local machine.

**1. Create a Directory:**
Open PowerShell and run:
```powershell
mkdir C:\actions-runner; cd C:\actions-runner
```

**2. Download the Runner:**
*Note: Check your GitHub Repo -> Settings -> Actions -> Runners -> New self-hosted runner for the exact latest link.*
```powershell
Invoke-WebRequest -Uri https://github.com/actions/runner/releases/download/v2.321.0/actions-runner-win-x64-2.321.0.zip -OutFile actions-runner-win-x64-2.321.0.zip
Add-Type -AssemblyName System.IO.Compression.FileSystem ; [System.IO.Compression.ZipFile]::ExtractToDirectory("$PWD/actions-runner-win-x64-2.321.0.zip", "$PWD")
```

**3. Configure:**
Replace `YOUR_USERNAME`, `YOUR_REPO`, and `YOUR_TOKEN` with values from your GitHub runner settings page.
```powershell
./config.cmd --url https://github.com/YOUR_USERNAME/YOUR_REPO --token YOUR_TOKEN
```
*   Press `Enter` for all prompts to accept defaults.

**4. Start the Runner:**
```powershell
./run.cmd
```
*Keep this PowerShell window open to process jobs.*

---

## 3. The MLOps Pipeline

### Stage 1: Data Ingestion (GEE)
Downloads Sentinel-2 imagery for a specific Area of Interest (AOI) and time range.

**Command:**
```bash
python ingest/gee_ingest.py ^
  --project YOUR_GCP_PROJECT_ID ^
  --aoi aoi/india_wayanad.geojson ^
  --before 2024-01-01 2024-01-31 ^
  --after 2024-11-01 2024-11-30 ^
  --name india_wayanad ^
  --drive-folder EO_Exports
```
*   **Action**: After running this, go to your Google Drive folder `EO_Exports` and download the generated `.tif` files to `data/raw/` in your local project.

### Stage 2: Data Preprocessing
Chips large GeoTIFFs into smaller tiles (e.g., 256x256) suitable for model training.

**Command:**
```bash
python preprocess/chip_dataset.py ^
  --before data/raw/india_wayanad_before.tif ^
  --after data/raw/india_wayanad_after.tif ^
  --mask data/raw/india_wayanad_mask.tif ^
  --out-dir data/chips
```

### Stage 3: Monitoring Baseline
Creates a statistical baseline of the dataset to detect drift later.

**Command:**
```bash
python monitor/create_baseline.py --chips-dir data/chips --out baseline.json
```

### Stage 4: Model Training
Trains the Siamese U-Net model.

**Local Training:**
```bash
python train/train.py --data-dir data/chips --epochs 5
```
*   **Output**: Saves the model to `runs/model_inference.pth`.

**Automated Training (GitHub Actions):**
1.  Push your code to GitHub.
2.  Go to **Actions** tab -> **Train Model**.
3.  Click **Run workflow**.
*   *Note: Requires `KAGGLE_USERNAME` and `KAGGLE_KEY` secrets in GitHub.*

### Stage 5: Deployment
Deploys the model as a FastAPI service using Docker.

**Local Manual Deployment:**
1.  **Build Image**:
    ```bash
    docker build -t planet-sentinel-serve -f serve/Dockerfile .
    ```
2.  **Run Container**:
    ```bash
    docker run -d --name planet-sentinel -p 8000:8000 planet-sentinel-serve
    ```
3.  **Test API**:
    ```bash
    curl -X POST "http://localhost:8000/predict" -F "file=@data/chips/tile_00000_before.tif"
    ```

**Automated Deployment (GitHub Actions):**
*   Ensure your Self-Hosted Runner is running (Section 2).
*   The workflow `.github/workflows/deploy.yaml` runs automatically on schedule (every 15 mins) or can be triggered manually.

### Stage 6: Monitoring (Drift Detection)
Simulates new data arriving and checks for distribution drift against the baseline.

**Command:**
```bash
python monitor/monitor.py ^
  --baseline baseline.json ^
  --new-data data/new_chips ^
  --repo YOUR_GITHUB_USERNAME/YOUR_REPO_NAME ^
  --token YOUR_GITHUB_PAT_TOKEN
```
*   *Note: If drift is detected, this script can trigger a GitHub Action to retrain the model.*

---

## 4. CI/CD Workflows Explained

### `deploy.yaml`
*   **Trigger**: Schedule (every 15 mins) or Manual.
*   **Runner**: `self-hosted` (Your local machine).
*   **Steps**:
    1.  Checks out code.
    2.  Installs dependencies.
    3.  (Simulated) Checks MLflow for a new "production-ready" model.
    4.  Builds the Docker image locally.
    5.  Stops/Removes the old container.
    6.  Starts the new container.

### `train.yaml`
*   **Trigger**: Manual (`workflow_dispatch`).
*   **Runner**: `ubuntu-latest` (GitHub Cloud).
*   **Steps**:
    1.  Sets up Python environment.
    2.  Installs Kaggle/MLflow/DVC.
    3.  Triggers a training kernel on Kaggle (using API).

---

## Documentation
*   [Requirements (FAQ)](docs/PLANET_SENTINEL_FAQ.md)
*   [Labeling Guide](docs/labeling_guide.md)
*   [Implementation Plan](docs/implementation_plan.md)
*   [Walkthrough](docs/walkthrough.md)