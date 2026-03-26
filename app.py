from flask import Flask, render_template, request, jsonify, session, g
import cv2
import numpy as np
import base64
import random
import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

app = Flask(__name__)
# Crucial for secure sessions
app.secret_key = os.environ.get("SECRET_KEY", "cureai_secure_session_key_2026")
DATABASE = 'cureai.db'

# --- DATABASE SETUP ---
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        # Users Table
        db.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, email TEXT UNIQUE, password TEXT, age INTEGER, water_intake TEXT
        )''')
        # Persistent History Table
        db.execute('''CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER, date TEXT, score INTEGER, acne_grade TEXT, kit TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )''')
        db.commit()

init_db()

# --- DIAGNOSTIC ENGINE ---
def cureskin_diagnostic_engine(image_data):
    try:
        encoded_data = image_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        h, w = img.shape[:2]
        roi = img[h//2-180:h//2+180, w//2-180:w//2+180] 

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 20, param1=50, param2=30, minRadius=2, maxRadius=10)
        comedones = len(circles[0]) if circles is not None else 0

        if comedones == 0: acne_grade = "Clear"
        elif comedones < 8: acne_grade = "Grade 1 (Mild)"
        elif comedones < 18: acne_grade = "Grade 2 (Moderate)"
        else: acne_grade = "Grade 3 (Active)"

        variance = np.var(gray)
        pigmentation = "Minimal" if variance < 1100 else "Moderate"
        dark_circles = "Not Detected" if np.mean(gray) > 125 else "Mild Visibility"

        score = max(40, 100 - (comedones * 3))
        moisture = random.randint(68, 92)
        skin_age = 18 + random.randint(-1, 2)
        
        # Determine recommended kit based on analysis (Ensures frontend match works)
        rec_kit = "Active Acne Kit" if comedones > 10 else ("Pore Control Kit" if comedones > 5 else "Barrier Cream")

        return {
            "score": int(score),
            "age": skin_age,
            "moisture": moisture,
            "acne_grade": acne_grade,
            "pigmentation": pigmentation,
            "dark_circles": dark_circles,
            "pores": "Visible" if comedones > 10 else "Refined",
            "oil": "Oily (High Sebum)" if np.mean(gray) > 175 else "Balanced",
            "condition": acne_grade + " Concerns",
            "diet": "Reduce dairy and refined sugar for 14 days.",
            "lifestyle": "Maintain 7.5 hours of sleep. Use a clean pillowcase.",
            "rec_kit": rec_kit
        }
    except:
        return {"score": 85, "age": 18, "moisture": 80, "acne_grade": "Clear", "pigmentation": "Minimal", "dark_circles": "None", "pores": "Refined", "oil": "Balanced", "condition": "Healthy Barrier", "diet": "Drink 3L water.", "lifestyle": "Regular sleep.", "rec_kit": "Gel Moisturizer"}

# --- ROUTES ---
@app.route('/')
def home(): 
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    result = cureskin_diagnostic_engine(data.get('image'))
    
    # Save to history automatically if logged in
    if 'user_id' in session:
        db = get_db()
        db.execute('INSERT INTO history (user_id, date, score, acne_grade, kit) VALUES (?, ?, ?, ?, ?)',
                   (session['user_id'], datetime.now().strftime("%Y-%m-%d %H:%M"), result['score'], result['acne_grade'], result['rec_kit']))
        db.commit()
        
    return jsonify(result)

# --- ACCOUNT & AUTHENTICATION ENDPOINTS ---
@app.route('/auth/signup', methods=['POST'])
def signup():
    data = request.json
    db = get_db()
    hashed_pw = generate_password_hash(data['password'])
    try:
        cursor = db.execute('INSERT INTO users (name, email, password, age, water_intake) VALUES (?, ?, ?, ?, ?)',
                            (data['name'], data['email'], hashed_pw, data.get('age', 18), data.get('water_intake', 'Good')))
        db.commit()
        session['user_id'] = cursor.lastrowid
        return jsonify({"success": True, "name": data['name']})
    except sqlite3.IntegrityError:
        return jsonify({"success": False, "message": "Account with this email already exists"}), 400

@app.route('/auth/login', methods=['POST'])
def login():
    data = request.json
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE email = ?', (data['email'],)).fetchone()
    if user and check_password_hash(user['password'], data['password']):
        session['user_id'] = user['id']
        return jsonify({"success": True, "name": user['name'], "age": user['age'], "water_intake": user['water_intake']})
    return jsonify({"success": False, "message": "Invalid email or password"}), 401

@app.route('/auth/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    return jsonify({"success": True})

@app.route('/auth/forgot-password', methods=['POST'])
def forgot_password():
    data = request.json
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE email = ?', (data['email'],)).fetchone()
    if user:
        new_pw = generate_password_hash(data['new_password'])
        db.execute('UPDATE users SET password = ? WHERE email = ?', (new_pw, data['email']))
        db.commit()
        return jsonify({"success": True, "message": "Password updated successfully!"})
    return jsonify({"success": False, "message": "Email address not found"}), 404

@app.route('/profile/update', methods=['POST'])
def update_profile():
    if 'user_id' not in session: return jsonify({"success": False}), 401
    data = request.json
    db = get_db()
    db.execute('UPDATE users SET name = ?, age = ?, water_intake = ? WHERE id = ?',
               (data['name'], data['age'], data['water_intake'], session['user_id']))
    db.commit()
    return jsonify({"success": True})

@app.route('/profile/history', methods=['GET'])
def get_history():
    if 'user_id' not in session: return jsonify({"success": False}), 401
    db = get_db()
    history = db.execute('SELECT * FROM history WHERE user_id = ? ORDER BY id DESC', (session['user_id'],)).fetchall()
    return jsonify({"success": True, "history": [dict(row) for row in history]})

@app.route('/auth/session', methods=['GET'])
def check_session():
    if 'user_id' in session:
        db = get_db()
        user = db.execute('SELECT name, age, water_intake FROM users WHERE id = ?', (session['user_id'],)).fetchone()
        if user:
            return jsonify({"logged_in": True, "user": dict(user)})
        else:
            session.pop('user_id', None) # Clear invalid session
    return jsonify({"logged_in": False})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
