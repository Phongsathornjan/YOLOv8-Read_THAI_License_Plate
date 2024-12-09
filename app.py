import sys
import os

# กำหนด Path ของ Virtual Environment
venv_path = os.path.join(os.path.dirname(__file__), 'myenv', 'Lib')

# เพิ่ม path ของ Lib ใน sys.path
sys.path.append(venv_path)

from flask import Flask
from main import *
from ultralytics import YOLO

app = Flask(__name__)

Crop_License_Plate_model = YOLO("train_Crop_License_Plate/train/weights/best.pt")
Read_License_Plate_model = YOLO("train_Read_License_Plate/train/weights/best.pt")

@app.route('/', methods=['POST'])
def read_license_plate():
    return main(Crop_License_Plate_model,Read_License_Plate_model)

if __name__ == '__main__':
    app.run()