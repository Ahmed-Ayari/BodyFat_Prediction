# Body Fat Analyser
 
> Upload a front and side photo — get an estimated body fat percentage in seconds.
 
A full-stack ML application that estimates body fat percentage from body silhouettes using a two-model pipeline: **EfficientNet-B0** extracts body measurements from silhouette images, and a **tabular MLP** converts those measurements into a body fat estimate. Built with PyTorch, FastAPI, and Next.js 15.
 
---
 
## Demo
 
> 🚧 Live demo coming soon — model training in progress.
 
---
 
## How it works
 
```
Front photo + Side photo
        │
        ▼
┌──────────────────────┐
│  EfficientNet-B0     │  ← fine-tuned on BodyM dataset
│  Silhouette encoder  │    predicts 14 body measurements
└──────────┬───────────┘
           │  waist, hip, chest, shoulder...
           ▼
┌──────────────────────┐
│  Tabular MLP         │  ← trained on Body Fat Prediction dataset
│  Regression head     │    inputs: measurements + height + weight + gender
└──────────┬───────────┘
           │
           ▼
    Body fat %  →  Category (Athletic / Fitness / Acceptable / Obese)
```
 
The silhouette approach strips clothing and lighting noise from the input, feeding only body shape into the model. This improves accuracy and protects user privacy since no identifiable photo is stored.
 
---
 
## Tech stack
 
| Layer | Technology |
|---|---|
| ML backbone | EfficientNet-B0 via `timm`, fine-tuned in PyTorch |
| Regression head | Custom MLP (PyTorch) |
| API | FastAPI + Uvicorn |
| Frontend | Next.js 15 (App Router) + TypeScript + Tailwind CSS |
| Deployment | Vercel (frontend) · Railway (backend) |
 
---
 
## Project structure
 
```
body-fat-analyser/
│
├── ml/
│   ├── data/                  # raw CSVs and BodyM silhouettes
│   ├── notebooks/             # EDA and training experiments
│   ├── src/
│   │   ├── dataset.py         # PyTorch Dataset classes (tabular + image)
│   │   ├── model.py           # EfficientNet-B0 wrapper + MLP definition
│   │   ├── train.py           # training loop with freeze/unfreeze strategy
│   │   └── evaluate.py        # MAE / RMSE evaluation
│   └── checkpoints/           # saved .pt model weights
│
├── backend/
│   ├── main.py                # FastAPI app + /predict endpoint
│   ├── inference.py           # model loading + preprocessing pipeline
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx           # upload UI (front + side photo)
│   │   └── result/page.tsx    # prediction display + body fat category
│   ├── components/
│   └── public/
│
├── docker-compose.yml
└── README.md
```
 
---
 
## ML pipeline
 
### Phase 1 — Tabular regression (MLP)
 
Trained on the [Body Fat Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset) (252 male subjects, 14 anthropometric features, hydrostatic weighing ground truth).
 
- Input: 14 body measurements (waist, hip, chest, etc.) + height + weight + gender
- Output: body fat percentage (regression)
- Loss: MSELoss
- Target MAE: < 4%
### Phase 2 — Silhouette encoder (EfficientNet-B0)
 
Fine-tuned on the [BodyM Dataset](https://registry.opendata.aws/bodym/) (2,505 real subjects, 8,978 frontal + lateral silhouettes, 14 labelled body measurements).
 
- Input: front silhouette + side silhouette (224×224, grayscale → RGB)
- Output: 14 body measurements (multi-output regression)
- Fine-tuning strategy: freeze backbone for 5 epochs, unfreeze all layers at lr=1e-5
- Pretrained weights: ImageNet via `timm`
### Combined inference
 
```
image pair → silhouette encoder → 14 measurements → tabular MLP → body fat %
```
 
---
 
## API
 
### `POST /predict`
 
Accepts a front and side photo, returns body fat percentage and category.
 
**Request**
```
Content-Type: multipart/form-data
 
front_image: <file>
side_image:  <file>
height_cm:   float
weight_kg:   float
gender:      "male" | "female"
```
 
**Response**
```json
{
  "body_fat_percentage": 18.4,
  "category": "Fitness",
  "measurements": {
    "waist_cm": 82.3,
    "hip_cm": 96.1,
    "chest_cm": 98.7
  }
}
```
 
---
 
## Body fat categories
 
Based on ACE (American Council on Exercise) classification:
 
| Category | Men | Women |
|---|---|---|
| Essential fat | 2–5% | 10–13% |
| Athletic | 6–13% | 14–20% |
| Fitness | 14–17% | 21–24% |
| Acceptable | 18–24% | 25–31% |
| Obesity | 25%+ | 32%+ |
 
---
 
## Getting started
 
### Prerequisites
 
- Python 3.10+
- Node.js 18+
- CUDA-capable GPU (optional but recommended for training)
### 1. ML — train the models
 
```bash
python -m venv venv && source venv/bin/activate

cd ml

pip install -r requirements.txt
 
# Download datasets
kaggle datasets download -d fedesoriano/body-fat-prediction-dataset -p data/
 
# Train tabular MLP first
python src/train.py --mode tabular
 
# Then train the silhouette encoder
python src/train.py --mode image
```
 
### 2. Backend — run the API
 
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
 
API docs available at `http://localhost:8000/docs`
 
### 3. Frontend — run the web app
 
```bash
cd frontend
npm install
npm run dev
```
 
Open `http://localhost:3000`
 
### Docker (full stack)
 
```bash
docker-compose up --build
```
 
---
 
## Datasets
 
| Dataset | Used for | Source |
|---|---|---|
| Body Fat Prediction Dataset | Tabular MLP training | [Kaggle](https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset) |
| BodyM Dataset | Silhouette encoder training | [AWS Open Data](https://registry.opendata.aws/bodym/) |
 
---
 
## Results
 
> Results will be updated as training completes.
 
| Model | Metric | Value |
|---|---|---|
| Tabular MLP | MAE | _ |
| Silhouette encoder | MAE (measurements) | _ |
| Full pipeline | Body fat MAE | _ |
 
---
 
## Roadmap
 
- [x] Project architecture and pipeline design
- [x] Tabular MLP — train and evaluate
- [x] EfficientNet-B0 — fine-tune on BodyM
- [ ] FastAPI backend with `/predict` endpoint
- [ ] Next.js 15 frontend (upload UI + results page)
- [ ] Deploy to Vercel + Railway
- [ ] Mobile app (React Native / Expo) — planned
---
 
## References
 
- Ruiz et al. : *Human Body Measurement Estimation with Adversarial Augmentation* (BodyM dataset paper)
- Surrogate body fat estimation via silhouette: [Koop et al., Nature Communications Medicine 2022](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9329470/)
- EfficientNet: Tan & Le, *EfficientNet: Rethinking Model Scaling for CNNs*, ICML 2019
- [`timm` library](https://github.com/huggingface/pytorch-image-models), PyTorch Image Models
---
 
## Author
 
**Ahmed**, M2 DSIR @ ISAMM
 
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/ahmed-ayari-767102266/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/Ahmed-Ayari)