from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

# Face detection model load karein
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def cureskin_diagnostic_engine(img):
    try:
        # Image processing for True Analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Redness Detection (Active Acne)
        mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        red_pixels = cv2.countNonZero(mask1 + mask2)
        red_ratio = (red_pixels / (img.shape[0] * img.shape[1])) * 100

        # 2. Texture & Pores Analysis
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 3. Oiliness (Brightness analysis)
        brightness = np.mean(hsv[:,:,2])

        # --- DYNAMIC RESULTS (NO RANDOMNESS) ---
        score = max(40, min(97, int(100 - (red_ratio * 15) - (laplacian_var / 50))))
        
        acne_grade = "Clear"
        rec_kit = "Oil Control Kit"
        if red_ratio > 1.2: 
            acne_grade = "Grade 3 (Active)"
            rec_kit = "Active Acne Kit"
        elif red_ratio > 0.6: 
            acne_grade = "Grade 2 (Moderate)"
            rec_kit = "Advanced Healing Kit"
        elif laplacian_var > 110:
            acne_grade = "Grade 1 (Mild)"
            rec_kit = "Pore Control Kit"

        return {
            "score": score,
            "acne_grade": acne_grade,
            "rec_kit": rec_kit,
            "moisture": max(55, 95 - int(red_ratio * 5)),
            "oil": "Oily" if brightness > 175 else "Balanced",
            "pores": "Visible" if laplacian_var > 100 else "Refined",
            "pigmentation": "Moderate" if laplacian_var > 140 else "Minimal",
            "diet": "Avoid oily foods and dairy." if red_ratio > 0.5 else "Maintain hydration.",
            "lifestyle": "Use a fresh towel daily." if red_ratio > 0.3 else "Get 8 hours of sleep."
        }
    except Exception as e:
        print(f"Engine Error: {e}")
        return {"score": 80, "acne_grade": "Analysis Error", "rec_kit": "Standard Kit"}

@app.route('/')
def home(): 
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        encoded_data = data.get('image').split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Face Validation
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0: 
            return jsonify({"face_detected": False})
        
        # Check alignment (centering)
        (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])
        img_w = img.shape[1]
        face_center_x = x + (w // 2)
        # Margin of 130 pixels from center
        is_centered = (img_w // 2 - 130) < face_center_x < (img_w // 2 + 130)
        
        # If it's just a precheck (validation) call
        if data.get('precheck'):
            return jsonify({"face_detected": True, "is_centered": is_centered})
        
        # Actual analysis on the face region
        roi = img[y:y+h, x:x+w]
        result = cureskin_diagnostic_engine(roi)
        return jsonify(result)
        
    except Exception as e:
        print(f"Route Error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
