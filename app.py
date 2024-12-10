import os

from flask import Flask, request, jsonify
from main import process_license_plate
from ultralytics import YOLO

app = Flask(__name__)

Crop_License_Plate_model = YOLO("train_Crop_License_Plate/train/weights/best.pt")
Read_License_Plate_model = YOLO("train_Read_License_Plate/train/weights/best.pt")

# กำหนดโฟลเดอร์สำหรับบันทึกรูปภาพชั่วคราว
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# สร้างโฟลเดอร์หากยังไม่มี
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['POST'])
def read_license_plate():
    
    try:
        # ตรวจสอบว่า request มีไฟล์
        if 'image' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['image']

        # ตรวจสอบว่าไฟล์มีชื่อ
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
         # ตรวจสอบชนิดไฟล์
        allowed_content_types = ['image/jpeg', 'image/png', 'image/gif']
        if file.content_type not in allowed_content_types:
            return jsonify({"error": "File type not allowed. Only JPEG, PNG, GIF images are accepted."}), 400

        # บันทึกไฟล์
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], "upload_Photo.jpg")
        file.save(file_path)
        
        result = process_license_plate(Crop_License_Plate_model,Read_License_Plate_model)
        os.remove("uploads/upload_Photo.jpg")
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({"error":e}), 500

if __name__ == '__main__':
    app.run()