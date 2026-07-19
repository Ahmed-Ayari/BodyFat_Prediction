from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from pathlib import Path
import schemas
import os

import inference
import preprocessing

upload_dir = Path(__file__).parent / "uploads"
upload_dir.mkdir(exist_ok=True)
allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

app = FastAPI()


def get_body_fat_feedback(body_fat_percentage: float) -> tuple[str, str]:
    # Thresholds are tailored for adult males because female body-fat prediction is not enabled.
    if body_fat_percentage < 6:
        return (
            "Essential fat",
            "Very low body-fat level. Keep training smart and make sure nutrition and recovery are sufficient.",
        )
    if body_fat_percentage < 14:
        return (
            "Athletic",
            "Great athletic range. Maintain your routine with balanced strength training, cardio, and protein intake.",
        )
    if body_fat_percentage < 18:
        return (
            "Fitness",
            "Healthy and fit range. Stay consistent with workouts and monitor diet quality to keep progressing.",
        )
    if body_fat_percentage < 25:
        return (
            "Average",
            "Average range. To improve body composition, increase activity and follow a moderate calorie deficit.",
        )
    if body_fat_percentage < 30:
        return (
            "Overweight",
            "Above the healthy range. Focus on regular gym sessions, daily movement, and a structured nutrition plan.",
        )
    return (
        "Obese",
        "High body-fat level. It is strongly recommended to start a supervised fitness plan and consult a healthcare professional.",
    )

@app.get("/")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict_body_fat(
    front_image: UploadFile = File(...),
    side_image: UploadFile = File(...),
    age: float = Form(...),
    weight: float = Form(...),
    height: float = Form(...),
    gender: schemas.Gender = schemas.Gender.MALE,
):
    if Path(front_image.filename).suffix.lower() not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Invalid file type for front image.")

    if Path(side_image.filename).suffix.lower() not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Invalid file type for side image.")

    try:
        front_image_path = upload_dir / front_image.filename
        side_image_path = upload_dir / side_image.filename

        with open(front_image_path, "wb") as f:
            f.write(await front_image.read())
        
        with open(side_image_path, "wb") as f:
            f.write(await side_image.read())

        front_mask = preprocessing.photo_to_mask(front_image_path)
        side_mask = preprocessing.photo_to_mask(side_image_path)

        gender_value = 1 if gender == schemas.Gender.MALE else 0

        # Call the inference function
        prediction = inference.predict_vision_model(
            inference.vision_model, 
            inference.target_dict, 
            front_mask, 
            side_mask, 
            inference.device, 
            inference.measurement_cols, 
            gender_value
        )

        combined_input = inference.combine_predictions(prediction, age, weight, height)
        if gender == schemas.Gender.MALE:
            fat_prediction = inference.predict(
                inference.model,
                inference.scaler,
                inference.feature_columns,
                combined_input,
                inference.device,
            )
        else:
            fat_prediction = None  # Body fat prediction is not supported for females
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if front_image_path.exists():
            os.remove(front_image_path)
        if side_image_path.exists():
            os.remove(side_image_path)

    category = None
    message = None

    if fat_prediction is None:
        message = "Body fat prediction is not supported for females at the moment."
    else:
        category, recommendation = get_body_fat_feedback(fat_prediction)
        message = (
            f"Category: {category}. "
            f"\n"
            f"Recommendation: {recommendation}"
        )

    return schemas.PredictResponse(
        body_fat_percentage=fat_prediction,
        measurements=prediction,
        body_fat_supported=(gender == schemas.Gender.MALE),
        category=category,
        message=message,
    )