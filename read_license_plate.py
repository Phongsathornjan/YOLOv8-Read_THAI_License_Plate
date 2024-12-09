from province_abbreviation_to_name import *
from map_label import *

def read_license_plate(results, model, i,coordinates):
    xy = []  # ใช้ list เพื่อเก็บข้อมูล box ทั้งหมด
    if len(results[0].boxes.xyxy) != 0:
        for box in results[0].boxes:
            # ดึงข้อมูลของ bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_index = int(box.cls[0])  # ดึง index ของ class
            class_name = model.names[class_index]  # ดึงชื่อคลาส
            confidence = float(box.conf[0])  # ความมั่นใจ
                
            # แปลงชื่อคลาส
            if class_name.isalpha():  # ตรวจสอบว่าเป็นตัวอักษรล้วน
                thai_char = province_abbreviation_to_name(class_name)  # แปลงเป็นชื่อจังหวัด
            else:
                thai_char = map_label(class_name)  # แปลงเป็นพยัญชนะไทย
            
            # เพิ่มข้อมูล box ลงใน list
            xy.append({
                "coordinates": [x1, y1, x2, y2],
                "class_name": thai_char,
                "confidence": confidence
            })
            
        # จัดเรียง box ตามค่า y1
        sorted_boxes_byy1 = sorted(xy, key=lambda ob: ob["coordinates"][1])  # ใช้ y1
        provide = sorted_boxes_byy1[len(sorted_boxes_byy1)-1]
        sorted_boxes_byy1.pop(len(sorted_boxes_byy1)-1)
        sorted_boxes_byx1 = sorted(sorted_boxes_byy1, key=lambda ob: ob["coordinates"][0])  # ใช้ x1
        plate_number = ""
        for s in sorted_boxes_byx1:
            plate_number += s["class_name"]
        return {
                "detect_id" : i,
                "coordinates" : {
                    "x1":coordinates[0],
                    "y1":coordinates[1],
                    "x2":coordinates[2],
                    "y2":coordinates[3],
                    },
                "plate_number": plate_number,
                "provide": provide['class_name']
                }
    else:
        return {
                "detect_id" : i,
                "coordinates" : {
                    "x1":coordinates[0],
                    "y1":coordinates[1],
                    "x2":coordinates[2],
                    "y2":coordinates[3],
                    },
                "plate_number": "Can't detect",
                "provide": "Can't detect"
                }
