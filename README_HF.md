---
title: BodyFat Prediction
emoji: 🏃
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.20.0
python_version: '3.12'
app_file: app.py
pinned: false
---

## BodyFat Prediction

Upload a front photo, a side photo, and a few basic details to estimate body fat percentage and body measurements.

This Space is a Gradio frontend for the Body Fat Analyser project. It connects to a FastAPI backend that processes silhouette masks from the uploaded images, predicts body measurements with a vision model, and estimates body fat for supported users.

Notes:

- Body fat percentage is currently supported for male users only.
- Female users still receive the predicted body measurements.
- Uploaded photos are converted to silhouettes for inference and then removed after processing.




