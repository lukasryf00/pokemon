import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np

model_path = "pokemon_classifier_model.keras"
model = tf.keras.models.load_model(model_path)


labels = ['Bulbasaur', 'Jigglypuff', 'Rhyhorn']
    
def predict_image(image):
    # Preprocess image
    image = Image.fromarray(image.astype('uint8'))  # Convert numpy array to PIL image
    image = image.resize((224, 224))  # Resize the image to 224x224 pixels
    image = np.array(image) / 255.0   # Convert to float and normalize

    # Ensure the image has 3 color channels
    if image.ndim == 2:  # If grayscale, convert to RGB
        image = np.stack((image,)*3, axis=-1)

    prediction = model.predict(image[None, ...])  # Adding batch dimension
    confidences = {labels[i]: float(prediction[0][i]) for i in range(len(labels))}
    return confidences


input_image = gr.Image()
output_text = gr.Textbox(label="Predicted Value")


iface = gr.Interface(
    fn=predict_image,
    inputs=input_image, 
    outputs=gr.Label(),
    title="Pokémon Classifier",
    examples=["pokemon/Bulbasur.png", "pokemon/Jigglypuff.png", "pokemon/Rhyhorn.png"],
    description="Upload an image of Bulbasur, Jigglypuff, or Rhyhorn and the classifier will predict which one it is."
)



iface.launch()
