from province_abbreviation_to_name import *
from map_label import *

def read_license_plate(results, model, i,coordinates):
    xy = []  # ใช้ list เพื่อเก็บข้อมูล box ทั้งหมด
    if len(results[0].boxes.xyxy) != 0:
        haveProvince = 0
        provinceData = []
        index = 0
        for box in results[0].boxes:
            # ดึงข้อมูลของ bounding box

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_index = int(box.cls[0])  # ดึง index ของ class
            class_name = model.names[class_index]  # ดึงชื่อคลาส
            confidence = float(box.conf[0])  # ความมั่นใจ
            # แปลงชื่อคลาส
            if class_name.isalpha():  # ตรวจสอบว่าเป็นตัวอักษรล้วน
                thai_char = province_abbreviation_to_name(class_name)  # แปลงเป็นชื่อจังหวัด
                haveProvince += 1
                provinceData.append({
                "class_name": thai_char,
                "confidence": confidence,
                "index" : index
                })
            else:
                thai_char = map_label(class_name)  # แปลงเป็นพยัญชนะไทย
            
            # print(thai_char + " " + str(confidence) + "%" + " index = "+str(index))

            # เพิ่มข้อมูล box ลงใน list
            xy.append({
                "coordinates": [x1, y1, x2, y2],
                "class_name": thai_char,
                "confidence": confidence
            })
            index += 1
        
        # for o in provinceData:
        #     print("ProvinceIndex = "+str(o))
        
        # จัดเรียง box ตามค่า y1
        # for o in xy:
        #     print(o["class_name"])

        if haveProvince == 1:
            sorted_boxes_byy1 = sorted(xy, key=lambda ob: ob["coordinates"][1])  # ใช้ y1
            provide = sorted_boxes_byy1[len(sorted_boxes_byy1)-1]
            sorted_boxes_byy1.pop(len(sorted_boxes_byy1)-1)
            sorted_boxes_byx1 = sorted(sorted_boxes_byy1, key=lambda ob: ob["coordinates"][0])  # ใช้ x1
        elif haveProvince == 0:
            sorted_boxes_byy1 = sorted(xy, key=lambda ob: ob["coordinates"][1])  # ใช้ y1
            provide = {"class_name":"can't detect"}
            sorted_boxes_byx1 = sorted(sorted_boxes_byy1, key=lambda ob: ob["coordinates"][0])  # ใช้ x1   
        else:
            provide = provinceData[0]
            for o in provinceData[1:]:
                if provide['confidence'] < o['confidence']:
                    provide = o

            # ดึง index ที่ต้องการลบจาก provinceData
            indices_to_remove = [item["index"] for item in provinceData]

            # เรียง index จากมากไปน้อย
            indices_to_remove.sort(reverse=True)

            # ลบรายการใน xy ตาม index ที่เรียงไว้
            for index in indices_to_remove:
                if 0 <= index < len(xy):  # ตรวจสอบว่า index อยู่ในช่วงที่ถูกต้อง
                    xy.pop(index)
                    
            sorted_boxes_byy1 = sorted(xy, key=lambda ob: ob["coordinates"][1])  # ใช้ y1
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
