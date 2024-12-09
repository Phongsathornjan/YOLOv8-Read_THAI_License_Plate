import matplotlib.pyplot as plt
import cv2
from flask import jsonify

from province_abbreviation_to_name import *
from map_label import *
from plot_license_plate import *
from read_license_plate import *

def process_license_plate(model1,model2):
    # โหลดโมเดล
    Crop_License_Plate_model = model1
    Read_License_Plate_model = model2

    # อ่านภาพต้นฉบับ
    image_path = "uploads/upload_Photo.jpg"
    original_image = cv2.imread(image_path)

    # ตรวจจับกรอบป้ายทะเบียน
    Crop_results = Crop_License_Plate_model.predict(image_path)
    all_boxes = plot_license_plate(Crop_results, Crop_License_Plate_model)

    # อ่านข้อมูลจากกรอบที่ครอบแต่ละกรอบ
    result = []
    for i, (x1, y1, x2, y2) in enumerate(all_boxes, start=1):  # เริ่มที่ Figure 2
        cropped_image = original_image[y1:y2, x1:x2]  # ตัดภาพตามพิกัด
        Read_result = Read_License_Plate_model.predict(cropped_image)
        result.append(read_license_plate(Read_result, Read_License_Plate_model, i, [x1,y1,x2,y2]))
        # plt.figure(i)
        # plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        # plt.title(f'License Plate {i-1}')
        # plt.axis('off')

    # แสดงผลทั้งหมด
    # plt.show()
    return result

if __name__ == "__main__":
    process_license_plate()