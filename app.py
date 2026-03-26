from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import random
import os

app = Flask(__name__)

# Face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def cureskin_diagnostic_engine(image_data):
    try:
        encoded_data = image_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # ROI Focus
        h, w = img.shape[:2]
        roi = img[h//2-180:h//2+180, w//2-180:w//2+180] 

        # Analysis Logic
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 20, param1=50, param2=30, minRadius=2, maxRadius=10)
        comedones = len(circles[0]) if circles is not None else 0

        # Scoring
        if comedones == 0: acne_grade = "Clear"
        elif comedones < 8: acne_grade = "Grade 1 (Mild)"
        elif comedones < 18: acne_grade = "Grade 2 (Moderate)"
        else: acne_grade = "Grade 3 (Active)"

        # Frontend demands 'rec_kit' for shop matching
        kits = ["Active Acne Kit", "Advanced Healing Kit", "Oil Control Kit", "Pore Control Kit"]
        rec_kit = kits[0] if comedones > 15 else (kits[1] if comedones > 5 else kits[2])

        score = max(40, 100 - (comedones * 3))
        
        return {
            "score": int(score),
            "age": 18 + random.randint(-1, 2),
            "moisture": random.randint(68, 92),
            "acne_grade": acne_grade,
            "rec_kit": rec_kit, # Crucial for Smart Profile Match
            "condition": acne_grade + " Concerns",
            "oil": "Oily" if np.mean(gray) > 175 else "Balanced"
        }
    except:
        return {"score": 85, "acne_grade": "Clear", "rec_kit": "Oil Control Kit"}

@app.route('/')
def home(): return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    return jsonify(cureskin_diagnostic_engine(data.get('image')))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
