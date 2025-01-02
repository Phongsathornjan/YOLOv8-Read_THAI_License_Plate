import matplotlib.pyplot as plt
import cv2
from flask import jsonify
from ultralytics import YOLO

from province_abbreviation_to_name import *
from map_label import *
from plot_license_plate import *
from read_license_plate import *

def process_license_plate():
    # โหลดโมเดล
    Crop_License_Plate_model = YOLO("YOLO_crop.pt")
    Read_License_Plate_model = YOLO("YOLO_read.pt")     
    # อ่านภาพต้นฉบับ
    image_path = "uploads/upload_Photo.jpg"
    original_image = cv2.imread(image_path)

    # ตรวจจับกรอบป้ายทะเบียน
    Crop_results = Crop_License_Plate_model.predict(original_image)
    all_boxes_plate = plot_license_plate(Crop_results, Crop_License_Plate_model)
    
    # ถ้าไม่พบป้ายทะเบียน ให้ซูมเข้าและตรวจจับอีกครั้ง
    if len(all_boxes_plate) == 0:
        scale = 1.2
        h, w = original_image.shape[:2]
        center_x, center_y = w // 2, h // 2
        new_w, new_h = int(w / scale), int(h / scale)

        cropped_image = original_image[center_y - new_h // 2:center_y + new_h // 2,
                           center_x - new_w // 2:center_x + new_w // 2]
        if cropped_image.size == 0:
            print("Error: Cropped image is empty.")
            return []

        original_image = cropped_image
        Crop_results = Crop_License_Plate_model.predict(original_image)
        all_boxes_plate = plot_license_plate(Crop_results, Crop_License_Plate_model)
        
        
        # ถ้ายังไม่พบป้ายทะเบียน ให้คืนค่าเป็น []
        if len(all_boxes_plate) == 0:
            return []

    # อ่านข้อมูลจากกรอบที่ครอบแต่ละกรอบ
    result = []
    for i, (x1, y1, x2, y2) in enumerate(all_boxes_plate, start=1):  # เริ่มที่ Figure 2
        cropped_image = original_image[y1+5:y2-5, x1+5:x2-5]  # ตัดภาพตามพิกัด
        resized_image = cv2.resize(cropped_image, (400, 400))
        Read_result = Read_License_Plate_model.predict(resized_image)
        result.append(read_license_plate(Read_result, Read_License_Plate_model, i, [x1,y1,x2,y2]))
    #     plt.figure(i)
    #     plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    #     plt.title(f'License Plate {i-1}')
    #     plt.axis('off')

    # # แสดงผลทั้งหมด
    # plt.show()
    return result

if __name__ == "__main__":
    process_license_plate()