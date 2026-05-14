import gradio as gr
from ultralytics import YOLO
import google.generativeai as genai
from PIL import Image
import numpy as np
import cv2

# ==========================================
# 1. การตั้งค่าระบบ (Configuration)
# ==========================================

# ใส่ Gemini API Key ที่ได้จาก Google AI Studio
# สมัครได้ที่: https://aistudio.google.com/
GEMINI_API_KEY = "AIzaSyBvZ_cYtsF15YilP0trd12wpJuuUkjoT6c"

genai.configure(api_key=GEMINI_API_KEY)
# ลองใช้ตัว Lite ที่มักจะโควตาเหลือเฟือสำหรับสายฟรี
gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')

# โหลดโมเดล YOLO26n-seg ที่เทรนเสร็จแล้ว
# ตรวจสอบให้แน่ใจว่าไฟล์ best.pt อยู่ในโฟลเดอร์เดียวกัน
try:
    model = YOLO('./Models/best.pt')
except Exception as e:
    print(f"Error: ไม่พบไฟล์ best.pt กรุณาตรวจสอบตำแหน่งไฟล์ ({e})")

# ==========================================
# 2. ฟังก์ชันประมวลผล (Core Logic)
# ==========================================

def analyze_car_damage(input_img):
    if input_img is None:
        return None, "กรุณาอัปโหลดรูปภาพ"

    # --- Step A: YOLO Detection & Segmentation ---
    # ทำนายผลจากรูปภาพ
    results = model.predict(source=input_img, conf=0.25)
    
    # วาดผลลัพธ์ (Masks/Boxes) ลงบนภาพ
    res_plotted = results[0].plot()
    
    # แปลงจาก BGR (OpenCV) เป็น RGB (PIL) เพื่อแสดงผลใน Gradio
    output_image = Image.fromarray(res_plotted[:, :, ::-1])
    
    # นับจำนวนจุดที่ตรวจพบ
    damage_count = len(results[0].boxes)
    
    # --- Step B: Gemini Cost Estimation ---
    # สร้าง Prompt ส่งให้ Gemini
    prompt = f"""
    ในฐานะผู้พัฒนาโปรเจค Vehicle Damage Detection ตรวจพบความเสียหายทั้งหมด {damage_count} จุด บนรถคันนี้
    
    จากภาพที่แนบมา ช่วยวิเคราะห์ดังนี้:
    1. ประเภทของความเสียหายในแต่ละจุด (เช่น รอยบุบ, รอยขีดข่วนลึก)
    2. ประเมินราคาซ่อมรายจุด (หน่วยบาท)
    3. ราคารวมโดยประมาณ
    
    ตอบเป็นภาษาไทย รูปแบบ Markdown ที่อ่านง่าย
    """
    
    try:
        response = gemini_model.generate_content([prompt, input_img])
        analysis_text = response.text
    except Exception as e:
        analysis_text = f"เกิดข้อผิดพลาดในการเรียก Gemini API: {e}"
    
    return output_image, analysis_text

# ==========================================
# 3. ส่วนติดต่อผู้ใช้ (Frontend with Gradio)
# ==========================================

# ==========================================
# 🎨 นำสไตล์จากโปรเจคเก่ามาปรับใช้ (Dark Navy + Orange + Glass Panel)
# ==========================================

custom_css = """
/* พื้นหลังสีน้ำเงินเข้มแบบโปรเจคเก่า */
.gradio-container { 
    background: #0f172a !important; 
    font-family: sans-serif; 
}

/* สร้างกล่องสไตล์ Glass Panel ให้ดูเด่นขึ้นมา */
.glass-panel { 
    background: rgba(255, 255, 255, 0.95) !important; 
    border-radius: 20px !important;
    padding: 25px !important; 
    box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important;
}

/* บังคับตัวอักษรในกล่องให้เป็นสีเข้ม จะได้อ่านง่ายบนพื้นขาว */
.black-text * { 
    color: #0f172a !important; 
}

/* ปรับแต่งปุ่มกดให้เป็นสีส้ม #ff9800 แบบโปรเจคเก่า */
.predict-btn { 
    background: #ff9800 !important; 
    color: #ffffff !important; 
    border: 2px solid #ff9800 !important;
    border-radius: 12px !important; 
    font-size: 1.2rem !important; 
    font-weight: 900 !important; 
    padding: 10px 25px !important;
    transition: all 0.2s ease;
}
.predict-btn:hover {
    background: #e68a00 !important;
    border-color: #e68a00 !important;
    transform: translateY(-2px);
}

/* กล่องแสดงผลข้อความ Gemini ให้ดูเป็นระเบียบคล้าย Metric Box */
.output-markdown { 
    background: #ffffff !important; 
    border: 1px solid #cbd5e1 !important; 
    border-radius: 10px !important; 
    padding: 20px !important;
}
"""

with gr.Blocks(css=custom_css) as demo:
    
    # ใช้ HTML จัดหัวข้อแบบโปรเจคเก่าเป๊ะๆ (ข้อความขาว ไฮไลต์ส้ม)
    gr.HTML("<h1 style='text-align: center; color: white; margin: 20px 0;'>CAR DAMAGE <span style='color: #ff9800;'>ASSESSMENT PRO</span></h1>")
    
    with gr.Row():
        # ฝั่งซ้าย: อัปโหลดรูป (ใส่คลาส glass-panel และ black-text)
        with gr.Column(elem_classes=["glass-panel", "black-text"]):
            gr.Markdown("### 📸 อัปโหลดรูปภาพความเสียหาย")
            input_img = gr.Image(type="pil", label="อัปโหลดรูปรถยนต์")
            btn = gr.Button("🔍 เริ่มการวิเคราะห์ระบบ", elem_classes="predict-btn")
            
        # ฝั่งขวา: แสดงผล YOLO
        with gr.Column(elem_classes=["glass-panel", "black-text"]):
            gr.Markdown("### 🛡️ ผลการตรวจจับ (YOLO)")
            output_img = gr.Image(label="AI Vision")
            
    with gr.Row():
        # ด้านล่าง: รายงานสรุป
        with gr.Column(elem_classes=["glass-panel", "black-text"]):
            gr.Markdown("### 📄 รายงานการประเมินราคา (Gemini AI)")
            output_text = gr.Markdown(
                "*ระบบพร้อมวิเคราะห์... กรุณาอัปโหลดรูปภาพด้านบน*", 
                elem_classes="output-markdown"
            )

    # เชื่อมโยงฟังก์ชัน
    btn.click(fn=analyze_car_damage, inputs=input_img, outputs=[output_img, output_text])

demo.launch(share=True)