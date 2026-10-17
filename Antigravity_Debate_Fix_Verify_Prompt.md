# Antigravity Prompt: Multi-Agent Debate, Fix, Verify Loop

## Project: `MiniProject_Part2_BigData_ML.ipynb`

คุณกำลังทำงานใน Antigravity workspace ที่มีไฟล์ Notebook ชื่อ
`MiniProject_Part2_BigData_ML.ipynb`

ภารกิจคือ **ตรวจสอบข้อผิดพลาด แก้ไข Notebook จริง รันตรวจสอบ และให้ reviewer
อีกบทบาทโต้แย้งผลการแก้ไขอย่างเป็นระบบ** ทำซ้ำจนผ่านเกณฑ์หรือมีข้อจำกัดที่พิสูจน์ได้
นี่ไม่ใช่งานให้เขียนคำแนะนำอย่างเดียว

## 1. กติกาหลัก

1.  เปิดอ่าน Notebook ฉบับจริงใน workspace ทั้ง Markdown, source code, cell
    order, outputs, metadata และ dependencies
2.  อย่าเชื่อ handoff/การวิเคราะห์เดิมโดยอัตโนมัติ ตรวจทุกข้อกับไฟล์จริงและจัดประเภท
    `CONFIRMED`, `NEEDS VERIFICATION`, `NOT APPLICABLE` หรือ `DISPROVEN`
3.  ต้องแก้ไฟล์จริง รันตรวจ และตรวจซ้ำ อย่าหยุดแค่รายงานปัญหา
4.  เก็บต้นฉบับไว้ สร้างไฟล์ใหม่ `MiniProject_Part2_BigData_ML_fixed.ipynb`
5.  ห้ามแต่งผลลัพธ์หรืออ้างว่ารันสำเร็จหากไม่ได้รันจริง
6.  ห้ามแก้เฉพาะตัวเลขใน Markdown หากต้นเหตุอยู่ในโค้ด ต้องแก้โค้ดและรันใหม่ก่อน
7.  รักษาโครงสร้าง ภาษาไทย และส่วนที่ถูกต้องไว้ ไม่เขียนโปรเจกต์ใหม่ทั้งหมดโดยไม่จำเป็น
8.  ห้ามใช้ Test set เลือกโมเดลหรือจูน hyperparameters หลังจากดูผล Test
9.  ทุกข้อสรุปต้องมีหลักฐานจาก cell/code/output/test หรือแหล่งอ้างอิงที่ตรวจจริง
10. หากไม่มีข้อมูลหรือ dependency ให้แยกสิ่งที่ตรวจ static ได้ออกจากสิ่งที่รันยืนยันไม่ได้
    พร้อมระบุข้อจำกัด
11. หากต้องตรวจนิยามฟิลด์ TLC ให้ใช้แหล่งที่เชื่อถือได้จริงและบันทึกแหล่งอ้างอิง
    ห้ามอ้างว่าได้ตรวจเว็บหากไม่ได้ตรวจ
12. อย่าหยุดเมื่อยังมีปัญหา P0 ที่แก้ได้ หากแก้ไม่ได้ให้ระบุเป็น `BLOCKED` พร้อมเหตุผล

## 2. บทบาทที่ต้องใช้ในการโต้แย้ง

หาก Antigravity รองรับ subagents ให้แยกบทบาทเหล่านี้ หากไม่รองรับ ให้ทำ review
pass แยกกันและตรวจ diff จริง ห้ามแสร้งว่ามี independent review ถ้าไม่ได้ทำ

### Agent A --- Data Scientist / Implementer

ตรวจปัญหา เสนอวิธีแก้ แก้ Notebook จริง รันการคำนวณ และอธิบายผลข้างเคียง

### Agent B --- Skeptical Reviewer

พยายามหักล้างข้อสรุปของ Implementer ตรวจ leakage, split contamination,
feature availability, metric validity, edge cases และผลข้างเคียง
ต้องอ้างหลักฐาน ไม่เห็นด้วยเพียงเพื่อให้บทสนทนาดูครบ

### Agent C --- Reproducibility Engineer

ตรวจ cell order, imports, dependencies, paths, random seeds, stale
outputs, reproducibility และความตรงกันของ Markdown/outputs

### Agent D --- Adjudicator / Final Auditor

ตัดสินจากหลักฐาน ไม่ใช่เสียงข้างมาก หากหลักฐานไม่พอให้คงสถานะ unresolved
หรือเลือกแนวทางที่ระมัดระวังที่สุด

## 3. ขั้นตอนทำงาน: Audit → Debate → Fix → Test → Challenge → Fix Again

### Phase 1: Baseline audit

1.  ยืนยันชื่อ/ตำแหน่งไฟล์ต้นฉบับ
2.  อ่านทุกเซลล์และทำ cell map: index, type, purpose, inputs, outputs,
    dependencies
3.  บันทึก split, preprocessing, model pipeline, metrics, dataset paths
    และ errors ที่มี
4.  ตรวจว่า outputs อาจ stale หรือมาจากโค้ดคนละเวอร์ชันหรือไม่
5.  สร้าง issue ledger: Issue ID, cell(s), severity, evidence,
    implementer proposal, reviewer objection, adjudicator decision,
    patch, verification, remaining risk, status

อย่าแก้จากสมมติฐานก่อนทำ baseline audit

### Phase 2: ตรวจประเด็นที่เคยพบ (ต้องตรวจซ้ำกับ Notebook จริงทั้งหมด)

#### P0-A --- Target Encoding Leakage (คาดว่าอยู่ใกล้ Cell 18)

ตรวจ custom transformer `SmoothedTargetEncoder` และทุกจุดที่เรียก `.fit()` /
`.transform()`: - encoder ที่ fit ด้วย `X_train, y_train` ถูกใช้ transform
แถว Train ชุดเดิมหรือไม่? - target ของแถวหนึ่งส่งผลต่อ encoding ที่ใช้ train
แถวนั้นเองหรือไม่? - preprocessing/encoder fit ใหม่จาก fold-train ในแต่ละ CV
fold หรือไม่? - fold-train ถูก transform แบบ in-sample หรือ cross-fitted? -
มีการใช้ Validation/Test target เพื่อ fit encoding หรือไม่?

หากพบ leakage: - ทำ out-of-fold/cross-fitted target encoding ให้แต่ละแถว
Train ไม่ใช้ target ของตัวเอง - Validation/Test ใช้ encoder ที่ fit
จากข้อมูลฝึกที่อนุญาตเท่านั้น - ทุก CV fold fit preprocessing/encoding ใหม่ภายใน
fold ห้าม encode ทั้ง dataset ก่อน CV - เพิ่ม test/assertion ที่พิสูจน์ว่าแถว
held-out ไม่ใช้ target ของตัวเอง - หากทำอย่างปลอดภัยไม่ได้ ให้ใช้ encoder ที่รองรับ
cross-fitting หรือเอา target encoding ออกและเลือก categorical encoding
ที่เหมาะสม - คำนวณ CV/Validation/Test และ feature importance ใหม่หลังแก้

#### P0-B --- การใช้ Test set ซ้ำ (คาดว่าอยู่ใกล้ Cell 24)

ตรวจว่ามีการใช้ `predict(X_test)` เพื่อเปรียบเทียบหลายโมเดลหรือจูนโมเดลหรือไม่ - ใช้
CV/Validation สำหรับเลือกโมเดลและ tuning - ล็อก model, features,
preprocessing และ hyperparameters ก่อนประเมิน Test - ประเมิน Test
เฉพาะโมเดลสุดท้ายหลังล็อกการตัดสินใจ - ถ้า Test ถูกใช้เลือกโมเดลไปแล้ว ให้บันทึกข้อจำกัด
และพิจารณาสร้าง holdout ใหม่ที่ยังไม่ถูกใช้หากทำได้ - อย่าลบ final test evaluation
ที่จำเป็น แต่แยกจากตาราง model selection ให้ชัด

#### P0-C --- Feature Availability / Prediction Timestamp (คาดว่าอยู่ใกล้ Cell 3 และ 14)

โจทย์ระบุว่าพยากรณ์ `trip_duration_min` ณ เวลาเริ่มเดินทาง ตรวจว่า feature
ทุกตัวมีอยู่จริงในเวลานั้นหรือไม่ โดยเฉพาะ: - `trip_distance`: ตรวจ data
dictionary TLC ว่าเป็นระยะทางจากมิเตอร์หรือ planned route distance
ที่ทราบก่อนเดินทาง - `RatecodeID`: ตรวจนิยามและเวลาที่ทราบค่า
ห้ามสมมติว่าพร้อมก่อนเริ่มเดินทาง - `store_and_fwd_flag`: ตรวจว่าใช้ได้ ณ
prediction time หรือไม่ - `is_airport_trip`: ตรวจแหล่งข้อมูลที่ใช้สร้าง feature -
pickup/dropoff zones, `route_pair`, pickup hour/day/season และ derived
features อื่น ๆ

หาก feature ไม่พร้อมใช้ก่อนเดินทาง: - ถ้าโจทย์คือ ETA ก่อนเดินทาง ให้ตัด feature
นั้นหรือแทนด้วยข้อมูลที่พร้อมใช้จริง เช่น planned route distance
จากแหล่งข้อมูลที่ตรวจสอบได้ - ถ้าคง feature จากมิเตอร์ ให้ปรับ use case ให้ชัดว่าเป็น
prediction หลังทราบข้อมูลนั้น ไม่ใช่ ณ เวลาเริ่ม - ห้ามใช้ dropoff timestamp,
fare/total หลังจบการเดินทาง หรือข้อมูลหลังเหตุการณ์เป็น predictor สำหรับ ETA ณ
เวลาเริ่ม - ทำ feature audit table: feature, definition,
available-at-time, evidence, keep/drop/replace

#### P1-A --- CV sampling / folds (คาดว่าอยู่ใกล้ Cell 20)

ตรวจการสุ่มตัวอย่างประมาณ 400,000 แถวด้วย `rng.choice()` และคำอธิบายว่าเป็น
stratified sampling หรือไม่ - ถ้าสุ่มธรรมดา ให้เรียก random sample ไม่ใช่
stratified - หากอ้าง stratified ให้ตรวจ strata และการสุ่มจริง -
รายงานจำนวนแถวที่ใช้ CV จริงและ random seed - สร้าง folds ก่อน fit
preprocessing/target encoding - คำนวณ mean ± SD จาก fold metrics จริง

#### P1-B --- Random split กับ temporal generalization (คาดว่าอยู่ใกล้ Cell 16)

-   Random stratified split ไม่ได้พิสูจน์ generalization ไปอนาคต
-   หาก use case คืออนาคต ให้เพิ่ม time-based holdout: train ช่วงก่อนและ test
    ช่วงหลัง
-   ห้าม fit preprocessing/encoding ด้วยข้อมูลที่เกิดหลังช่วง test ใน temporal
    evaluation
-   ข้อมูลมีเพียง 4 บล็อกสัปดาห์ในปี 2024
    จึงต้องระบุข้อจำกัดเรื่องเวลาและการเป็นตัวแทนทั้งปี
-   หากเพิ่ม temporal split ไม่ได้ ให้บันทึกเหตุผลและจำกัดข้อกล่าวอ้าง

#### P1-C --- Cleaning filters และ population bias

ตรวจเกณฑ์ duration 1--180 นาที, distance 0.1--100 ไมล์, fare/total \> 0,
average speed 1--65 mph, passenger count, RatecodeID และ Unknown/Outside
NYC - รายงานจำนวนแถวที่ถูกตัดในแต่ละขั้นจากการรันจริง - แยก data errors ออกจาก
population filters - พิจารณาว่าการกรอง speed ต่ำอาจตัดเที่ยวรถติดหรือไม่ - ระบุว่า
metrics ใช้กับประชากรหลัง cleaning เท่านั้น - ห้ามเปลี่ยน threshold
เพื่อเพิ่มคะแนนโดยไม่มีเหตุผลจากโดเมน

#### P1-D --- ตัวเลขใน Markdown ไม่ตรงกับ outputs

ค่าที่เคยพบและต้องตรวจซ้ำ ไม่ใช่ค่าที่ให้เลือกใช้โดยอัตโนมัติ: - Mean
`trip_duration_min`: Markdown 16.92 นาที เทียบ output 16.36 นาที - Median
`trip_duration_min`: Markdown 13.48 นาที เทียบ output 12.78 นาที - Skewness
`trip_distance`: Markdown 4.08 เทียบ output 2.66 - Manhattan mean
duration: Markdown 14.95 นาที เทียบ output 14.11 นาที - Queens mean
duration: Markdown 32.27 นาที เทียบ output 36.63 นาที - Winter mean
duration: Markdown 15.61 นาที เทียบ output 14.80 นาที - Spring mean
duration: Markdown 17.13 นาที เทียบ output 16.08 นาที - Fall mean duration:
Markdown 17.65 นาที เทียบ output 17.52 นาที - Features หลัง preprocessing:
Markdown 51 เทียบ output 49 - Tuned LightGBM CV MAE: Markdown
`2.894 ± 0.007` เทียบ output `3.067 ± 0.017` - Weekend share: Markdown
26.90% เทียบ output 28.06%

ตรวจว่าแต่ละค่ามาจาก dataset/filter/metric version เดียวกันหรือไม่
อย่าเลือกตัวเลขด้วยการเดา หากต่างกันเพราะ subset หรือนิยามต่างกัน ให้ระบุเหตุผลและ
subset อย่างชัดเจน อัปเดตจากผลรันล่าสุดเท่านั้น

#### P2-A --- Feature importance และ safety buffer

-   `trip_distance` กับ `log_trip_distance` มีความสัมพันธ์กัน จึงไม่ควรตีความ
    importance แยกเป็นอิทธิพลอิสระ
-   ห้ามตีความ target-encoded feature ก่อนแก้ leakage
-   ตรวจข้อกล่าวอ้างว่า subgroup สนามบิน/เที่ยวไกลมี MAE ราว 6.2 นาทีจาก output
    จริง
-   Safety Buffer `+7 ถึง +10 นาที` ต้องมี subgroup underprediction
    quantile/coverage analysis รองรับ หากไม่มีให้ระบุเป็นสมมติฐานที่ต้องทดสอบ
    ไม่ใช่ข้อสรุปที่พิสูจน์แล้ว

## 4. Debate protocol: ต้องมีการคัดค้านที่มีสาระ

สำหรับทุก issue สำคัญ: 1. Implementer claim: ระบุปัญหาและเสนอวิธีแก้พร้อมหลักฐาน
2. Reviewer challenge: พยายามหักล้างโดยตรวจ edge cases, leakage, metric
validity, feature timing และผลข้างเคียง 3. Implementer response: ตอบด้วย
code evidence หรือ test result ไม่ใช่การยืนยันความเห็นเดิม 4. Adjudicator
decision: ตัดสินตามหลักฐาน อาจปฏิเสธทั้งสองทางเลือก 5. Implementer แก้ไฟล์จริง 6.
Reviewer ตรวจ diff และรันทดสอบ patch 7. หากพบปัญหาใหม่ ให้เปิด issue
ใหม่และแก้ต่อ

ใช้บันทึกต่อ issue รูปแบบนี้:

``` text
Issue ID:
Evidence in original notebook:
Implementer proposal:
Reviewer objection:
Evidence/test resolving the objection:
Adjudicator decision:
Patch applied:
Verification performed:
Remaining risk:
Status: PASS / FAIL / BLOCKED
```

อย่าสร้างบทสนทนาเทียม และอย่าอ้างว่าเป็น independent review
หากไม่ได้ตรวจไฟล์/patch จริง หากไม่มี multi-agent ให้ทำ review pass
แยกกันและเปิดดู diff หลังแก้

## 5. Verification หลังแก้

1.  ตรวจ syntax ทุก code cell, imports, variables, dependencies และ paths
2.  รัน Notebook จากต้นจนจบใน kernel/session ใหม่ หากข้อมูลและ environment
    พร้อม
3.  หากรันเต็มไม่ได้ ให้ทำ smoke test เท่าที่ทำได้และระบุชัดว่าไม่ใช่ full run
4.  ตรวจ warnings, exceptions, skipped cells และ stale outputs
5.  เพิ่ม/รัน test สำหรับ leakage safeguards เท่าที่ทำได้
6.  ตรวจว่า Test set ใช้หลัง model selection เท่านั้น
7.  ตรวจว่า metrics ใน Markdown ตรงกับ outputs ที่รันใหม่
8.  ตรวจ split, sample size, features และ cleaning descriptions
    ให้ตรงกับโค้ด
9.  หากเปลี่ยน features, encoder หรือ split ต้องคำนวณ metrics ใหม่
    ห้ามใช้ผลก่อนแก้เป็นผลสุดท้าย
10. บันทึกหลักฐานการรันจริงและ error log ที่สำคัญ

## 6. วงรอบแก้ไขจนผ่าน

ทำ `Review → Debate → Fix → Run → Independent Review` ซ้ำจน: - ไม่มี P0
ที่ยังแก้ได้ค้างอยู่ - P1 สำคัญถูกแก้หรือมีเหตุผลที่ตรวจสอบได้ว่าทำไมยังแก้ไม่ได้ - Reviewer
ไม่พบข้อผิดพลาดใหม่ที่ทำให้ผลไม่น่าเชื่อถือ - Notebook รันได้ หรือมีข้อจำกัดที่รายงานชัดเจน -
ตัวเลขและคำอธิบายสอดคล้องกัน

หากยังคัดค้านกันไม่ได้ ให้ตั้ง `BLOCKED`/`UNRESOLVED` ไม่ใช่ `PASS`
ระบุหลักฐานที่ขาดและผลกระทบ อย่าหยุดเพียงเพราะแก้รายการเดิมครบ
หากการทดสอบพบปัญหาใหม่ให้ดำเนินการต่อ

## 7. Final acceptance checklist

-   [ ] ตรวจข้อสังเกตเดิมทุกข้อกับ Notebook จริงแล้ว
-   [ ] target encoding ไม่มี self-target leakage
-   [ ] preprocessing/encoding fit ภายในแต่ละ fold/split
-   [ ] Test set ไม่ถูกใช้เลือก/จูนโมเดลซ้ำ
-   [ ] feature audit สอดคล้องกับ prediction timestamp
-   [ ] sampling และ CV อธิบายตรงกับโค้ดจริง
-   [ ] temporal generalization ถูกทดสอบหรือระบุข้อจำกัด
-   [ ] cleaning filters มีเหตุผลและอธิบายผลกระทบ
-   [ ] Markdown, tables, plots และ outputs ใช้ผลชุดล่าสุด
-   [ ] ไม่มีคำกล่าว "100% leakage-free" ที่ไม่มีหลักฐาน
-   [ ] Notebook รันจริง หรือมีข้อจำกัดชัดเจน
-   [ ] reviewer ตรวจ diff หลังแก้แล้ว
-   [ ] มี issue ledger และ changelog
-   [ ] ต้นฉบับยังอยู่ และมีไฟล์ fixed แยกต่างหาก

## 8. ไฟล์ที่ต้องส่งมอบ

สร้างไฟล์ใน workspace: 1. `MiniProject_Part2_BigData_ML_fixed.ipynb` ---
Notebook ที่แก้จริง 2. `MiniProject_Part2_BigData_ML_review.md` --- issue
ledger, debate decisions, evidence, run/test results และข้อจำกัด 3.
`MiniProject_Part2_BigData_ML_changelog.md` --- รายการแก้ไขหลัก แยกตาม
cell/หัวข้อ

รายงานสุดท้ายต้องบอก: - ปัญหาใด confirmed/disproven/blocked -
แก้ส่วนใดและเพราะอะไร - metrics ก่อน/หลังแก้ โดยระบุว่าค่าใดเป็นผลรันจริง -
Notebook รันจากต้นจนจบสำเร็จหรือไม่ พร้อมหลักฐาน - reviewer
คัดค้านอะไรและแก้แล้วหรือยัง - ความเสี่ยงที่ยังเหลือ

**เริ่มทำงานทันที:** ตรวจไฟล์จริง → baseline audit → debate → แก้ไฟล์ → รันตรวจ →
reviewer ตรวจ patch → แก้ซ้ำจนผ่านหรือบันทึก blocker ที่มีหลักฐาน
ห้ามตอบกลับด้วยแผนอย่างเดียวโดยไม่ลงมือแก้ไฟล์ใน workspace
