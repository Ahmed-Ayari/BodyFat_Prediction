# Body Fat Analyser

> Upload a front and side photo — get an estimated body fat percentage in seconds.

A full-stack ML application that estimates body fat percentage from body silhouettes using a two-model pipeline: **EfficientNet-B0** extracts body measurements from silhouette images, and a **tabular MLP** converts those measurements into a body fat estimate. Built with PyTorch, FastAPI, and Next.js 15.

---

## Demo

> 🚧 Live demo coming soon — frontend in progress.

---

## How it works

```
Front photo + Side photo + gender + age + weight + height
        │
        ▼
┌──────────────────────┐
│  rembg segmentation  │  ← converts raw photo to silhouette mask
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  EfficientNet-B0     │  ← fine-tuned on BodyM dataset, gender-aware
│  Silhouette encoder  │    predicts 14 body measurements
└──────────┬───────────┘
           │  waist, hip, chest, shoulder...
           ▼
   combine with user-submitted
   age, weight, height
           │
     gender == male?
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐   measurements
│ Tabular │   returned only,
│  MLP    │   body fat not
└────┬────┘   supported yet
     │         for this v1
     ▼
 Body fat %  →  Category (Athletic / Fitness / Acceptable / Obese)
```

The silhouette approach strips clothing and lighting noise from the input, feeding only body shape into the model. This improves accuracy and protects user privacy since no identifiable photo is stored, uploaded photos are converted to silhouette masks and deleted immediately after inference.

**Known v1 limitation:** the tabular MLP is trained exclusively on a male-only anthropometric dataset (see below), so body fat percentage is only computed for male users in this version. Female users still receive the full set of predicted body measurements from the vision model. Extending this to a mixed-gender tabular model is planned future work.

---

## Tech stack

| Layer | Technology |
|---|---|
| ML backbone | EfficientNet-B0 via `torchvision`, fine-tuned in PyTorch |
| Regression head | Custom MLP (PyTorch) |
| Photo → silhouette | `rembg` (background/foreground segmentation) |
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
│   │   ├── dataset.py         # PyTorch Dataset classes (tabular + image, gender-aware)
│   │   ├── model.py           # EfficientNet-B0 wrapper (gender embedding) + MLP definition
│   │   ├── train.py           # training loops (tabular + vision, freeze/unfreeze experiments)
│   │   └── evaluate.py        # MAE / RMSE / R² evaluation, per-measurement breakdown
│   └── checkpoints/           # saved .pt model weights + scaler
│
├── backend/
│   ├── main.py                # FastAPI app + /predict endpoint
│   ├── schemas.py             # Pydantic request/response models (Gender enum, PredictResponse)
│   ├── preprocessing.py       # rembg photo → silhouette mask conversion
│   ├── inference.py           # model loading, prediction, and vision→MLP feature mapping
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx           # upload UI (front + side photo, age, weight, height, gender)
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

Trained on the [Body Fat Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset) (252 male subjects, sourced from Penrose, Nelson & Fisher's 1985 anthropometric study, hydrostatic-weighing-derived ground truth).

- Input: 11 body measurements — Age, Weight, Height, Chest, Abdomen, Hip, Thigh, Ankle, Biceps, Forearm, Wrist — **no gender feature**, since the source dataset contains male subjects only
- Output: body fat percentage (regression)
- Architecture: MLP with BatchNorm + Dropout, hidden sizes [64, 32]
- Loss: MSELoss
- Units: dataset uses Height in inches and Weight in pounds (unusual, given all other measurements are in cm); the API converts user-submitted cm/kg internally before this step

**Neck and Knee dropped:** the vision model has no corresponding photo-derived output for Neck or Knee circumference (not part of BodyM's 14 measurements). Rather than approximate them with auxiliary estimators, the MLP was retrained without these two features. This was validated empirically, MAE **improved** from 3.9634 (13 features) to 3.7219 (11 features, R² 0.6133 vs. a near-zero baseline), confirming the drop didn't cost predictive power, likely because removing features reduced overfitting risk on a dataset of only 252 rows.

**Male-only constraint:** since this dataset has no female subjects, the fat-distribution/density relationship it learned doesn't generalize across sexes. Rather than silently misestimating for female users, the API gates this step by gender (see API section below). A mixed-gender dataset for retraining is noted as future work.

### Phase 2 — Silhouette encoder (EfficientNet-B0)

Fine-tuned on the [BodyM Dataset](https://registry.opendata.aws/bodym/) (front + lateral silhouette masks, 14 labelled body measurements, mixed-gender subjects).

- Input: front silhouette + side silhouette, stacked as a 2-channel 224×224 tensor, plus a gender flag (0/1)
- Gender handling: gender is projected through a small linear embedding (1 → 8 dims) and concatenated with the pooled visual features before the classifier head, so the model learns gender-specific measurement patterns directly
- Output: 14 body measurements (multi-output regression)
- Fine-tuning strategy: **backbone frozen entirely**, only the classifier head + gender embedding trained. Partial unfreezing of the last two backbone blocks was tested and did not improve validation loss (train/val gap widened instead), so the fully-frozen approach was kept for v1.
- Data augmentation: `RandomHorizontalFlip(p=0.5)` applied to training images only, so the model isn't strictly locked to left-side photos and is more robust to either side being submitted
- Pretrained weights: ImageNet, via `torchvision`

**Per-measurement validation MAE** ranges from ~0.87 cm (wrist) to ~6.9 cm (waist). Measurements with less distinguishable silhouette signal (leg-length, calf, arm-length) show weaker R², a known limitation for this approach.

### Combined inference

```
raw photo pair → rembg segmentation → silhouette masks → EfficientNet-B0 → 14 measurements
                                                                                    │
                                                          + user's age/weight/height ▼
                                                                          map to MLP features
                                                                          (waist → Abdomen proxy)
                                                                                    │
                                                              (if gender == male)   ▼
                                                                              Tabular MLP → body fat %
```

**Abdomen approximation:** the vision model predicts `waist`, not `Abdomen` (the MLP's actual trained feature). These are anatomically distinct measurement points, waist is typically the narrowest point of the torso, abdomen is usually measured at navel level, so this is a documented approximation, not an exact substitution. It's a known source of error in the final body fat estimate, alongside general calibration drift (below).

**Height and Weight taken directly from the user**, not predicted by the vision model, since the vision model's absolute-scale outputs are implicitly calibrated to BodyM's controlled capture conditions (fixed camera distance/setup), which real user phone photos won't match. Age is naturally not obtainable from a photo either. The vision model still predicts height internally for its own measurement set, but that value is not used as MLP input.

---

## API

### `POST /predict`

Accepts a front and side photo, gender, age, weight, and height. Returns predicted body measurements and, for male users, an estimated body fat percentage.

**Request**
```
Content-Type: multipart/form-data

front_image: <file>
side_image:  <file>   (side/profile view; either left or right should work)
age:         float
weight:      float   (kg)
height:      float   (cm)
gender:      "male" | "female"
```

**Response**
```json
{
  "measurements": {
    "waist": 96.02,
    "hip": 106.52,
    "chest": 107.50,
    "...": "..."
  },
  "body_fat_percentage": 22.57,
  "body_fat_supported": true,
  "message": null
}
```

For non-male users in this v1:
```json
{
  "measurements": { "...": "..." },
  "body_fat_percentage": null,
  "body_fat_supported": false,
  "message": "Body fat percentage estimation currently supports male users only. Body measurements are still provided."
}
```

**Implementation notes:**
- Uploaded photos are validated by extension, converted to silhouette masks via `rembg` in-memory, passed directly to the vision model (no intermediate mask file), then the original uploads are deleted in a `finally` block regardless of success or failure.
- Gender is validated as a strict enum (`male`/`female` only) at the request-parsing layer, rejecting anything ambiguous before it reaches the models, this was added deliberately after an early test showed a mismatched/wrong gender value can silently produce a badly skewed prediction.
- The male-only gate is enforced *before* the MLP is called, not after, non-male requests never run the male-only model at all.

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

# Train tabular MLP
python src/train.py --mode tabular

# Train the silhouette encoder (frozen backbone)
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
| Body Fat Prediction Dataset (Penrose et al., male-only) | Tabular MLP training | [Kaggle](https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset) |
| BodyM Dataset | Silhouette encoder training | [AWS Open Data](https://registry.opendata.aws/bodym/) |

---

## Results

| Model | Metric | Value |
|---|---|---|
| Tabular MLP (13 features, with Neck/Knee) | MAE | 3.9634 |
| Tabular MLP (11 features, final) | MAE / RMSE / R² | 3.7219 / 4.6740 / 0.6133 |
| Tabular MLP baseline (mean predictor) | MAE / R² | 6.4216 / -0.0132 |
| Silhouette encoder | Best val loss (frozen backbone, flip-augmented) | 0.5289 |
| Silhouette encoder | MAE range across 14 measurements | ~0.87 cm (wrist) – ~6.9 cm (waist) |
| Full pipeline | Body fat MAE | _ (no ground-truth-labeled end-to-end test set yet; validated informally on real photos, plausible results) |

---

## Known v1 limitations

- **Male-only body fat prediction**: the tabular MLP's source dataset has no female subjects; extending to mixed-gender data is planned future work.
- **Abdomen approximated from waist**: the vision model doesn't predict Abdomen directly; waist circumference is used as a proxy, a documented source of error.
- **Neck and Knee dropped**: no photo-derived equivalent exists for these; removing them was validated to not hurt (and slightly improve) MLP accuracy, so no substitute was introduced.
- **Segmentation quality**: `rembg` output on real-world user photos (varied lighting/backgrounds) will be noisier than BodyM's controlled silhouette captures.
- **Side orientation**: trained with horizontal-flip augmentation to reduce left/right sensitivity, but not formally benchmarked separately for each side.
- **Measurement calibration drift**: absolute-scale measurements from the vision model are implicitly calibrated to BodyM's controlled capture conditions and may drift on uncalibrated real-world photos. Height, weight, and age are taken directly from the user rather than predicted, to avoid compounding this into the body fat estimate specifically.

---

## Roadmap

- [x] Project architecture and pipeline design
- [x] Tabular MLP — train and evaluate
- [x] EfficientNet-B0 — gender-aware, fine-tuned on BodyM, flip-augmented
- [x] FastAPI backend with `/predict` endpoint
- [ ] Next.js 15 frontend (upload UI + results page)
- [ ] Deploy to Vercel + Railway
- [ ] Mixed-gender tabular dataset for full gender support
- [ ] Mobile app (React Native / Expo) — planned

---

## References

- Ruiz et al.: *Human Body Measurement Estimation with Adversarial Augmentation* (BodyM dataset paper)
- Penrose, Nelson & Fisher (1985): *Generalized body composition prediction equation for men using simple measurement techniques*
- Surrogate body fat estimation via silhouette: [Koop et al., Nature Communications Medicine 2022](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9329470/)
- EfficientNet: Tan & Le, *EfficientNet: Rethinking Model Scaling for CNNs*, ICML 2019
- `torchvision.models`, PyTorch pretrained model zoo
- `rembg`, background/foreground segmentation library

---

## Author

**Ahmed**, M1 DSIR @ ISAMM

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/ahmed-ayari-767102266/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/Ahmed-Ayari)