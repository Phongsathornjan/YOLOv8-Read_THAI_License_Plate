def map_label(class_name):
    # สร้าง mapping ของพยัญชนะไทย
    thai_chars = [
        "ก", "ข", "ฃ", "ค", "ฅ", "ฆ", "ง", "จ", "ฉ", "ช", "ซ", "ฌ", "ญ", "ฎ", "ฏ", "ฐ", 
        "ฑ", "ฒ", "ณ", "ด", "ต", "ถ", "ท", "ธ", "น", "บ", "ป", "ผ", "ฝ", "พ", "ฟ", "ภ", 
        "ม", "ย", "ร", "ล", "ว", "ศ", "ษ", "ส", "ห", "ฬ", "อ", "ฮ"
    ]
    
    # ตรวจสอบว่าคลาสเป็นรูปแบบ Axx
    if class_name.startswith("A") and len(class_name) > 1:
        try:
            index = int(class_name[1:]) - 1  # แปลง A01-A44 เป็น index
            if 0 <= index < len(thai_chars):
                return thai_chars[index]
        except ValueError:
            pass
    
    return class_name  # หากไม่ตรงเงื่อนไข คืนค่าดั้งเดิม