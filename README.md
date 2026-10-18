# Big Data Analytics & Machine Learning Pipeline: All-Season NYC Taxi Demand Forecasting

โครงงานพยากรณ์ความต้องการรถแท็กซี่ในมหานครนิวยอร์กครบทุกฤดูกาล (Yellow Taxi Hourly Zone-Level Demand Forecasting Across 4 Seasons) 
วิเคราะห์และสร้างแบบจำลองจากข้อมูลจริงกว่า **13,389,587 แถว** ครอบคลุมทั้ง 4 ฤดูกาลของปี 2024 (ฤดูหนาว ฤดูใบไม้ผลิ ฤดูร้อน และฤดูใบไม้ร่วง)

---

## 🚀 วิธีการเปิดและรันบน Google Colab (Run on Colab Guide)

โน้ตบุ๊ก [`MiniProject_Part2_BigData_ML.ipynb`](./MiniProject_Part2_BigData_ML.ipynb) ได้รับการออกแบบให้เป็นแบบ **Self-Contained & Production-Ready** สามารถเปิดรันบน Google Colab หรือ Kaggle ได้ทันทีโดยไม่ต้องเตรียมไฟล์ข้อมูลล่วงหน้า:

### ขั้นตอนการรันบน Google Colab:
1. เข้าไปที่ [Google Colab](https://colab.research.google.com/)
2. เลือกเมนู **File (ไฟล์) -> Upload notebook (อัปโหลดโน้ตบุ๊ก)** แล้วเลือกไฟล์ `MiniProject_Part2_BigData_ML.ipynb` (หรือเปิดผ่าน GitHub URL)
3. กดเมนูด้านบน **Runtime (รันไทม์) -> Run all (เรียกใช้ทั้งหมด)** หรือกดคีย์ลัด `Ctrl + F9`
4. **ระบบจะทำงานโดยอัตโนมัติทั้งหมด:**
   * ตรวจสอบสภาพแวดล้อมและติดตั้งแพ็กเกจที่จำเป็น (`pyarrow`, `lightgbm`) อัตโนมัติ
   * สร้างโฟลเดอร์ `data/` และดาวน์โหลดชุดข้อมูลทางการ Parquet ทั้ง 4 ฤดูกาล (2024-01, 2024-04, 2024-07, 2024-10) พร้อมตาราง `taxi_zone_lookup.csv` จาก NYC TLC CloudFront โดยอัตโนมัติ
   * รันการประเมินคุณภาพ ทำความสะอาดข้อมูล สร้าง Full Cartesian Grid และฝึกสอนโมเดล Machine Learning จนเสร็จสิ้นทุกเซลล์ (ใช้เวลาประมาณ 3–4 นาที บน Colab Free GPU/CPU)

---

## 📌 จุดเด่นทางเทคนิคและระเบียบวิธีวิจัย (Technical Highlights)

1. **ข้อมูลขนาดใหญ่ครอบคลุมครบ 4 ฤดูกาล (13.38M Rows):**
   * มกราคม 2024 (ฤดูหนาว): 2,964,624 แถว
   * เมษายน 2024 (ฤดูใบไม้ผลิ): 3,514,289 แถว
   * กรกฎาคม 2024 (ฤดูร้อน): 3,076,903 แถว
   * ตุลาคม 2024 (ฤดูใบไม้ร่วง): 3,833,771 แถว
   * ใช้เทคนิค **Memory-Safe Downcasting** (`int16`, `float32`) ประหยัด RAM ไม่เกิดปัญหา Out-of-Memory

2. **การทลายกับดัก Zero-Demand Trap:**
   * สร้าง **Full Cartesian Grid** เต็มรูปแบบ ($263 \text{ โซน} \times 2,952 \text{ ชั่วโมง} = \mathbf{776,376} \text{ แถว}$) และเติมค่า $0$ ในชั่วโมงที่ไม่มีการเรียกรถ ป้องกันความลำเอียงของโมเดลที่มักทำนายช่วงคนน้อยผิดพลาดอย่างรุนแรง

3. **การแบ่งข้อมูลตามลำดับเวลาภายในแต่ละฤดูกาล (Seasonal Stratified Chronological Split):**
   * ป้องกันปัญหาข้อมูลรั่วไหล (Temporal Data Leakage) โดยไม่ใช้ Random Split
   * แบ่งข้อมูลอย่างสมดุล: Train (วันที่ 8–21 รวม 353,472 แถว), Validation (วันที่ 22–26 รวม 126,240 แถว), และ Test (วันที่ 27–สิ้นเดือน รวม 119,928 แถว) ครบทั้ง 4 ฤดูกาล

4. **การป้องกัน Data Leakage อย่างเข้มงวด:**
   * ตัวแปร Lags และ Rolling Statistics ทุกตัวถูกคำนวณหลังเรียก `.shift(1)` เสมอ
   * Preprocessing Pipeline (`ColumnTransformer` OneHotEncoder + StandardScaler) ทำการ `.fit()` เฉพาะบนชุด **Train Set เท่านั้น**

5. **ประสิทธิภาพของแบบจำลองที่ดีที่สุด (LightGBM Regressor):**
   * **Test MAE:** 2.3531 เที่ยว/โซน/ชั่วโมง
   * **Test RMSE:** 7.9157
   * **Test $R^2$:** **0.9726** (ลด Error จาก Dummy Baseline ลงถึง 84.2%)
   * **TimeSeriesSplit CV (5 Folds):** $R^2 = 0.9427 \pm 0.0344$

---

## 🤖 สถาปัตยกรรมตัวแทนอัจฉริยะ (Multi-Agent Debate & Refinement Protocol)

โครงงานนี้ได้รับการกำกับและตรวจสอบเชิงวิชาการอย่างเข้มข้นผ่านการทำงานร่วมกันของ 3 บทบาทตัวแทน:
* **Builder Agent:** ทำหน้าที่ออกแบบโค้ด ไปป์ไลน์ และการคำนวณทางสถิติ
* **Adversarial Critic Agent:** ทำหน้าที่ตั้งคำถาม โต้แย้ง จับผิด Data Leakage และผลักดันการแก้ปัญหา Zero-Demand Trap ด้วย Full Cartesian Grid
* **Quality Judge Agent:** ทำหน้าที่ประเมินเกณฑ์คุณภาพ 100 คะแนนเต็ม ตรวจสอบความถูกต้องของตัวเลขจริง และยืนยันความสมบูรณ์ปราศจากข้อบกพร่อง (Zero Objections Consensus)