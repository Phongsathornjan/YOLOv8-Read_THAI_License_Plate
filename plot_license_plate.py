import matplotlib.pyplot as plt
import cv2

def plot_license_plate(results, model):
    all_boxes = []  # เก็บพิกัดของทุกกรอบ
    for result in results:
        for box in result.boxes:
            # พิกัดกรอบในรูปแบบ [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # ดึงชื่อคลาสจาก index
            class_index = int(box.cls[0])  # ต้องแปลงเป็น int ก่อน
            class_name = model.names[class_index]  # ดึงชื่อคลาส

            # พิมพ์ความมั่นใจและชื่อคลาส
            # print(f"Confidence: {box.conf[0]:.2f}, Class Name: {class_name}")
            
            # เก็บพิกัดเป็น int ใน all_boxes
            if class_name != "province":
                all_boxes.append((int(x1), int(y1), int(x2), int(y2)))
        
        result_img = result.plot()
        # plt.figure(1)
        # plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        # plt.title('Main photo')
        # plt.axis('off')
        # plt.show()
    return all_boxes  # 