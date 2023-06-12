from fastapi import FastAPI, File, UploadFile
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

app = FastAPI()

# path to tflite model
model_path = "model/model_update_1.tflite"

# list of class name
class_names = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
    'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
    'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
    'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
    'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
    'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
    'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
    'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
    'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
    'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
    'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup',
    'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters',
    'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
    'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
    'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto',
    'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits',
    'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
    'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare',
    'waffles'
]

# initialize tflite interpreter
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

def preprocess_image(image):

    # preprocess image before making a prediction
    # resize image to 224x224 pixels and apply preprocessing specific to the efficientnet model

    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = image_array.astype('float32')  # Convert to float32
    image_array = tf.keras.applications.efficientnet.preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_image(image):

    # run image classification using tflite model
    # return the predicted class name and prediction

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']

    input_data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction_idx = np.argmax(output_data)
    predicted_class_name = class_names[prediction_idx]
    prediction_prob = output_data[0][prediction_idx]

    return predicted_class_name, prediction_prob

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # endpoint for predicting the class of an uploaded image

    contents = await file.read()

    # check if the file is in JPEG format
    if not file.filename.lower().endswith((".jpg", ".jpeg")):
        return {
            "error": "Only support .jpg/.jpeg file"
            }
    
    image = Image.open(io.BytesIO(contents))

    # perform prediction
    predicted_class, prediction_prob = predict_image(image)

    return {
        "predicted_class": predicted_class,
        "prediction_prob": float(prediction_prob)
        }

@app.get("/")
def read_root():
    # root endpoint for checking if the API is running
    return {"Detect food API started"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
