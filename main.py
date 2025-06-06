from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np
import io
import os
from PIL import Image, ImageOps
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


model = keras.models.load_model('best_model.h5')


def transform_image(pillow_image):
    target_size = (331, 331)
    resize_image = ImageOps.fit(pillow_image, target_size, Image.LANCZOS)

    grayscale_image = ImageOps.grayscale(resize_image)

    img_array = np.array(grayscale_image)

    img_array_eq = cv2.equalizeHist(img_array)
    img_array_eq_rgb = cv2.cvtColor(img_array_eq, cv2.COLOR_GRAY2RGB)

    normalized_image = img_array_eq_rgb / 255.0

    normalized_image = cv2.resize(normalized_image, (331, 331))

    normalized_image = np.expand_dims(normalized_image, axis=0)

    return normalized_image


def predict(x):
    predictions = model(x)
    predictions = tf.nn.softmax(predictions)
    pred0 = predictions[0]
    label0 = np.argmax(pred0)
    return label0


app = Flask(__name__)


@app.route('/check', methods=['POST'])
def index():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})

        try:
            image_bytes = file.read()
            pillow_img = Image.open(io.BytesIO(image_bytes)).convert('L')
            tensor = transform_image(pillow_img)
            prediction = predict(tensor)

            label_names = {0: 'No DR (No Diabetic Retinopathy)',
                           1: 'Mild DR',
                           2: 'Moderate DR',
                           3: 'Severe DR',
                           4: 'Proliferative DR'}

            label_name = label_names.get(prediction, 'Unknown')

            # Konversi nilai 'prediction' menjadi int
            prediction = int(prediction)

            recommendation = {}

            if label_name == 'No DR (No Diabetic Retinopathy)':
                message = ["Check your blood sugar levels regularly. Controlled blood sugar levels are key to managing diabetes."]
                general_recommendation = ["How to control blood sugar levels\n",
                                          "1. Eat the right foods: According to a study in the journal Education and Health Promotion, skipping meals for too long can cause blood sugar levels to drop and then spike rapidly.\n",
                                          "2. Control your portion sizes: Diabetics with normal body weight should also watch their portion sizes to avoid obesity.\n",
                                          "3. Stay active and exercise regularly: Exercise helps your muscles take in more glucose and convert it into energy, thereby lowering blood sugar. Avoid a sedentary lifestyle (being lazy) and minimal physical activity or energy expenditure, such as watching TV, playing games on devices, or sitting too long in front of a computer.\n",
                                          "4. Manage stress well: Excessive stress can also cause blood sugar levels to rise due to the release of cortisol, also known as the stress hormone. To prevent stress from causing blood sugar levels to spike, it is important to understand how to control stress and try various things that can improve your mood, relax your body, and calm your mind."
                                          "5. Get enough rest: Getting enough sleep can balance hormones, avoid stress, and give you enough energy for activities and exercise the next day. This will help keep your blood sugar levels under control.\n",
                                          "6. Check your blood sugar regularly: By continuously monitoring changes in your blood sugar levels, it will be easier to determine whether you need to adjust your diet or medication."
                                          ]
                
                recommendation = {
                    "message": message,
                    "general_recommendation": general_recommendation
                }
            elif label_name == 'Mild DR':
                message = ["Check your blood sugar levels regularly. Controlled blood sugar levels are key to managing diabetes."]
                general_recommendation = ["How to control blood sugar levels\n",
                                          "1. Eat the right foods: According to a study in the journal Education and Health Promotion, skipping meals for too long can cause blood sugar levels to drop and then spike rapidly.\n",
                                          "2. Control your portion sizes: Diabetics with normal body weight should also watch their portion sizes to avoid obesity.\n",
                                          "3. Stay active and exercise regularly: Exercise helps your muscles take in more glucose and convert it into energy, thereby lowering blood sugar levels. Avoid a sedentary lifestyle (being lazy) and minimal physical activity or energy expenditure, such as watching TV, playing games on electronic devices, or sitting for long periods in front of a computer.\n",
                                          "4. Manage stress well: Excessive stress can also cause blood sugar levels to rise due to the release of cortisol, also known as the stress hormone. To prevent stress from causing blood sugar levels to spike, it is important to understand how to control stress and try various things that can improve your mood, relax your body, and calm your mind."
                                          "5. Get enough rest: Getting enough sleep can balance hormones, avoid stress, and give you enough energy to be active and exercise the next day. This will help keep your blood sugar levels under control.\n",
                                          "6. Check your blood sugar regularly: By continuously monitoring changes in your blood sugar levels, it will be easier to determine whether you need to adjust your diet or medication."
                                          ]
                recommendation = {
                    "message": message,
                    "general_recommendation": general_recommendation
                }
            elif label_name == 'Moderate DR':
                message = ["Consult a doctor. The doctor will conduct further examinations to determine the appropriate treatment for moderate DR."]
                general_recommendation = ["Things you may do after consulting with your doctor\n",
                                          "1. Eye injection: Your doctor will inject steroids into your eye to stop inflammation and prevent new blood vessels from forming. Anti-VEGF injections may also be recommended, which can reduce swelling in the macula and improve vision.\n",
                                          "2. Laser surgery: A laser procedure called photocoagulation reduces swelling in the retina and removes abnormal blood vessels.\n",
                                          "3. Vitrectomy: If you have advanced diabetic retinopathy, you may need a vitrectomy. This eye surgery addresses problems with the retina and vitreous, the jelly-like substance in the center of the eye. The surgery can remove blood or fluid, scar tissue, and part of the vitreous gel so that light can focus properly on the retina."
                                          ]
                recommendation = {"message": message,
                                  "general_recommendation": general_recommendation}
            elif label_name == 'Severe DR':
                message = ["Consult a doctor. The doctor will conduct further examinations to determine the appropriate treatment for severe DR."]
                general_recommendation = ["Things you may do after consulting with your doctor\n",
                                          "1. Eye injection: The doctor will inject steroids into the eye to stop inflammation and prevent new blood vessels from forming. Anti-VEGF injections may also be recommended, which can reduce swelling in the macula and improve vision.\n",
                                          "2. Laser surgery: A laser procedure called photocoagulation reduces swelling in the retina and removes abnormal blood vessels.\n",
                                          "3. Vitrectomy: If you have advanced diabetic retinopathy, you may need a vitrectomy. This eye surgery addresses problems with the retina and vitreous, the jelly-like substance in the center of the eye. The surgery can remove blood or fluid, scar tissue, and part of the vitreous gel so that light can focus properly on the retina."
                                          ]
                recommendation = {"message": message,
                                  "general_recommendation": general_recommendation}
            elif label_name == 'Proliferative DR':
                message = ["Consult a doctor. The doctor will conduct further examinations to determine the appropriate treatment for Proliferative DR (Proliferative)."]
                general_recommendation = ["Things you may do after consulting with your doctor\n",
                                          "1. Eye injection: Your doctor will inject steroids into your eye to stop inflammation and prevent new blood vessels from forming. Anti-VEGF injections may also be recommended, which can reduce swelling in the macula and improve vision.\n",
                                          "2. Laser surgery: A laser procedure called photocoagulation reduces swelling in the retina and removes abnormal blood vessels.\n",
                                          "3. Vitrectomy: If you have advanced diabetic retinopathy, you may need a vitrectomy. This eye surgery addresses problems in the retina and vitreous, the jelly-like substance in the center of the eye. The surgery can remove blood or fluid, scar tissue, and part of the vitreous gel so that light can focus properly on the retina."
                                          ]
                recommendation = {"message": message,
                                  "general_recommendation": general_recommendation}

            data = {"prediction": prediction, "label": label_name,
                    "recommendation": recommendation}
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})
    return "OK"


if __name__ == "__main__":
    app.run(debug=True)
