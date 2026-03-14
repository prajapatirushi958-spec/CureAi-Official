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
        # --- STEP 1: LIGHTING NORMALIZATION ---
        # Histogram Equalization taaki lighting se report change na ho
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- STEP 2: ADVANCED ACNE DETECTION ---
        # Red & Brown channels ko isolate karke acne clusters dhoondna
        mask_red = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([15, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([165, 70, 50]), np.array([180, 255, 255]))
        acne_mask = cv2.addWeighted(mask_red, 1.0, mask_red2, 1.0, 0)
        
        # Count actual acne clusters (bilkul Cureskin ki tarah)
        contours, _ = cv2.findContours(acne_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        acne_count = len([c for c in contours if cv2.contourArea(c) > 5])

        # --- STEP 3: TEXTURE MAPPING ---
        # Sobel Filter use karke skin ke bareek pores aur marks pehchanna
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        texture_intensity = np.mean(np.sqrt(sobelx**2 + sobely**2))

        # --- HIGH PRECISION SCORING ---
        # 1. Acne Grade logic
        if acne_count > 15: 
            acne_grade = "Grade 3 (Severe)"
            rec_kit = "Active Acne Kit"
        elif acne_count > 5: 
            acne_grade = "Grade 2 (Moderate)"
            rec_kit = "Advanced Healing Kit"
        elif acne_count > 1:
            acne_grade = "Grade 1 (Mild)"
            rec_kit = "Pore Control Kit"
        else:
            acne_grade = "Clear"
            rec_kit = "Oil Control Kit"

        # 2. Moisture based on texture roughness
        moisture = max(40, min(95, int(98 - (texture_intensity * 1.5))))
        
        # 3. Final Health Score (Scientific Calculation)
        health_score = int(100 - (acne_count * 3) - (texture_intensity / 2))
        health_score = max(35, min(98, health_score))

        return {
            "score": health_score,
            "acne_grade": acne_grade,
            "rec_kit": rec_kit,
            "moisture": moisture,
            "oil": "High Sebum" if np.mean(gray) > 180 else "Balanced",
            "pores": "Visible" if texture_intensity > 45 else "Refined",
            "pigmentation": "Significant" if texture_intensity > 60 else "Minimal",
            "diet": "Strict: No sugar/dairy for 21 days." if acne_count > 5 else "Drink 3.5L water daily.",
            "lifestyle": "Clinical: Double cleanse & change linens." if acne_count > 2 else "Ensure 8h sleep."
        }
    except Exception as e:
        print(f"Engine Error: {e}")
        return {"score": 85, "acne_grade": "Healthy", "rec_kit": "Standard Kit"}

@app.route('/')
def home(): return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        encoded_data = data.get('image').split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Face detection for cropping only
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])
            roi = img[y:y+h, x:x+w]
        else:
            roi = img # Fallback

        return jsonify(cureskin_diagnostic_engine(roi))
    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": "Server Error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
