import matplotlib.pyplot as plt
import cv2
from flask import jsonify
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import numpy as np

from province_abbreviation_to_name import *
from map_label import *
from plot_license_plate import *
from read_license_plate import *


def CNN_process():
    
    labels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A01', 11: 'A02', 12: 'A04', 13: 'A06', 14: 'A07', 15: 'A08', 16: 'A09', 17: 'A10', 18: 'A12', 19: 'A13', 20: 'A14', 21: 'A16', 22: 'A18', 23: 'A19', 24: 'A20', 25: 'A21', 26: 'A22', 27: 'A23', 28: 'A24', 29: 'A25', 30: 'A26', 31: 'A28', 32: 'A30', 33: 'A31', 34: 'A32', 35: 'A33', 36: 'A34', 37: 'A35', 38: 'A36', 39: 'A37', 40: 'A38', 41: 'A39', 42: 'A40', 43: 'A41', 44: 'A42', 45: 'A43', 46: 'A44'}
    # โหลดโมเดล
    Crop_License_Plate_model = YOLO("YOLO_crop.pt")
    find_letter_model = YOLO("YOLO_find_letter.pt")
    CNN_Read_model = load_model("CNN_read.h5")
    # อ่านภาพต้นฉบับ
    image_path = "uploads/upload_Photo.jpg"
    original_image = cv2.imread(image_path)

    # ตรวจจับกรอบป้ายทะเบียน
    Crop_results = Crop_License_Plate_model.predict(image_path)
    all_boxes = plot_license_plate(Crop_results, Crop_License_Plate_model)
    if(len(all_boxes) == 0):
        return []

    # อ่านข้อมูลจากกรอบที่ครอบแต่ละกรอบ
    result = []
    figure_counter = 1
    # print(labels)
    for i, (x1, y1, x2, y2) in enumerate(all_boxes, start=1):  # เริ่มที่ Figure 2
        haveProvince = 0
        provinceData = []
        xy = []
        cropped_image = original_image[y1+5:y2-5, x1+5:x2-5]  # ตัดภาพตามพิกัด
        resized_image = cv2.resize(cropped_image, (400, 400))
        # ตรวจจับตัวอักษร
        Letter_results = find_letter_model.predict(resized_image)
        letter_boxes = plot_license_plate(Letter_results,find_letter_model)
        # print(letter_boxes)
        if(letter_boxes != []):
            for (x1, y1, x2, y2) in letter_boxes:
                letter_cropped_image = resized_image[y1:y2, x1:x2]  # ตัดภาพ
                resized_letter_image = cv2.resize(letter_cropped_image, (300, 300))  # ปรับขนาดภาพ
                gray_image = cv2.cvtColor(resized_letter_image, cv2.COLOR_BGR2GRAY)  # แปลงเป็น Grayscale
                equalized_image = cv2.equalizeHist(gray_image)  # ทำ Histogram Equalization
                image_rescaled = (equalized_image / 255.0).astype(np.float32)  # Rescale ค่าพิกเซล

                # ทำนายผลด้วยโมเดล CNN
                read_result = CNN_Read_model.predict(image_rescaled.reshape(1, 300, 300, 1))  # แปลงภาพเพื่อใส่โมเดล
                predicted_label_index = read_result.argmax()  # หาค่าดัชนีของผลลัพธ์ที่ค่ามากที่สุด
                class_name = labels[predicted_label_index]  # แมปค่าดัชนีกับเลเบล
                confidence = read_result[0][predicted_label_index]  # ความมั่นใจของผลลัพธ์
                
                if class_name.isalpha():  # ตรวจสอบว่าเป็นตัวอักษรล้วน
                    thai_char = province_abbreviation_to_name(class_name)  # แปลงเป็นชื่อจังหวัด
                    haveProvince += 1
                    provinceData.append({
                    "class_name": thai_char,
                    "confidence": confidence,
                    })
                else:
                    thai_char = map_label(class_name)  # แปลงเป็นพยัญชนะไทย
                xy.append({
                    "coordinates": [x1, y1, x2, y2],
                    "class_name": thai_char,
                    "confidence": confidence
                })
                # print(f"Predicted label: {thai_char}, Confidence: {confidence}")
                # plt.figure(figure_counter)
                # plt.title(f"Predicted label: {class_name}")
                # plt.imshow(cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB))
                figure_counter += 1
            sorted_boxes_byy1 = sorted(xy, key=lambda ob: ob["coordinates"][1])  # ใช้ y1
            provide = {"class_name":"can't detect"}
            sorted_boxes_byx1 = sorted(sorted_boxes_byy1, key=lambda ob: ob["coordinates"][0])
            plate_number = ""
            for s in sorted_boxes_byx1:
                plate_number += s["class_name"]
        result.append({
            "detect_id" : i,
            "coordinates": {
            "x1":x1,
            "y1":x2,
            "x2":y1,
            "y2":y2,
            },
            "plate_number": plate_number,
            "provide": provide['class_name'],
            })
    # plt.show()
    # print(result)
    return result

if __name__ == "__main__":
    CNN_process()