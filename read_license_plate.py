from province_abbreviation_to_name import *
from map_label import *

def read_license_plate(results, model, i):
    xy = []  # ใช้ list เพื่อเก็บข้อมูล box ทั้งหมด
    for result in results:
        print(f"License Plate {i-1}")
        if len(result.boxes.xyxy) != 0:
            for box in result.boxes:
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
            sorted_boxes = sorted(xy, key=lambda ob: ob["coordinates"][1])  # ใช้ y1
            provide = sorted_boxes[len(sorted_boxes)-1]
            sorted_boxes.pop(len(sorted_boxes)-1)
            final_sorted_boxes = sorted(sorted_boxes, key=lambda ob: ob["coordinates"][0])  # ใช้ x1
            
            # แสดงผลเรียงลำดับ
            for result in final_sorted_boxes:
                print(result["class_name"])
                
            print(provide["class_name"])
        else:
            print("Can't detect label in this License Plate")