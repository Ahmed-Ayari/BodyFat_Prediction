import gradio as gr
import requests

gender_list = {"Male": "male", "Female": "female"}

def predict_body_fat(front_image, side_image, age, weight, height, gender):
    if front_image is None or side_image is None:
        raise gr.Error("Please upload both front and side images.")
    
    with open(front_image, "rb") as f_front, open(side_image, "rb") as f_side:
        files = {
            "front_image": ("front_image.jpg", f_front, "image/jpeg"),
            "side_image": ("side_image.jpg", f_side, "image/jpeg"),
        }
        data = {
            "age": age,
            "weight": weight,
            "height": height,
            "gender": gender_list[gender]
        }

        try:
            response = requests.post("http://localhost:8000/predict", files=files, data=data)
        except requests.exceptions.RequestException as e:
            raise gr.Error(f"Error connecting to the backend: {e}")

        if response.status_code == 200:
            result = response.json()
            return (
                result.get("body_fat_percentage"),
                result.get("measurements"),
                result.get("body_fat_supported"),
                result.get("category"),
                result.get("message"),
            )
        else :
            raise gr.Error(f"Error from backend: {response.status_code} - {response.text}")
        
with gr.Blocks() as demo:
    gr.Markdown("## Body Fat Prediction")
    
    with gr.Row():
        front_image_input = gr.Image(label="Front Image", height=300, type="filepath")
        side_image_input = gr.Image(label="Side Image", height=300, type="filepath")
        age_input = gr.Number(label="Age", value=25)
        weight_input = gr.Number(label="Weight (kg)", value=70)
        height_input = gr.Number(label="Height (cm)", value=175)
        gender_input = gr.Dropdown(label="Gender", choices=list(gender_list.keys()), value="Male")

    submit_button = gr.Button("Predict Body Fat")

    with gr.Column():
        body_fat_output = gr.Textbox(label="Body Fat Percentage", interactive=False)
        measurements_output = gr.Textbox(label="Measurements", interactive=False)
        body_fat_supported_output = gr.Textbox(label="Body Fat Supported", interactive=False)
        category_output = gr.Textbox(label="Category", interactive=False)
        message_output = gr.Textbox(label="Message", interactive=False)
    submit_button.click(
        fn=predict_body_fat,
        inputs=[front_image_input, side_image_input, age_input, weight_input, height_input, gender_input],
        outputs=[body_fat_output, measurements_output, body_fat_supported_output, category_output, message_output]
    )

demo.launch()
