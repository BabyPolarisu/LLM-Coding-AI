# 💻 AI Coding Assistant — MiniProject
> ระบบ LLM สำหรับช่วยเขียนโค้ด พร้อมหน้าจอ UI รับ Prompt แบบอินเทอร์แอคทีฟ และระบบทดสอบรันโค้ดทันที

---

## 📖 สารบัญ (Table of Contents)
1. [ภาพรวมของโปรเจกต์ (Overview)](#-ภาพรวมของโปรเจกต์-overview)
2. [สถาปัตยกรรมของระบบ (System Architecture)](#-สถาปัตยกรรมของระบบ-system-architecture)
3. [คำอธิบายการทำงานของแต่ละระบบขั้นตอน (Step-by-Step System Breakdown)](#-คำอธิบายการทำงานของแต่ละระบบขั้นตอน)
   - [ระบบที่ 1: Dynamic Question Router (`hybrid_retriever.py -> QuestionRouter`)](#ระบบที่-1-dynamic-question-router-hybrid_retrieverpy---questionrouter)
   - [ระบบที่ 2: Code Knowledge Graph Database (`code_graph_engine.py`)](#ระบบที่-2-code-knowledge-graph-database-code_graph_enginepy)
   - [ระบบที่ 3: Hybrid RAG & Internet Resource Dataset (`dataset_manager.py`)](#ระบบที่-3-hybrid-rag--internet-resource-dataset-hybrid_retrieverpy--dataset_managerpy)
   - [ระบบที่ 4: Prompt Engineering Pipeline (`prompts.py`)](#ระบบที่-4-prompt-engineering-pipeline-promptspy)
   - [ระบบที่ 5: LLM Inference Engine (`llm_engine.py`)](#ระบบที่-5-llm-inference-engine-llm_enginepy)
   - [ระบบที่ 6: Interactive Web UI (`app.py`)](#ระบบที่-6-interactive-web-ui-apppy)
   - [ระบบที่ 7: Safe Code Execution Sandbox (`code_runner.py`)](#ระบบที่-7-safe-code-execution-sandbox-code_runnerpy)
   - [ระบบที่ 8: Configuration & Model Settings (`config.py`)](#ระบบที่-8-configuration--model-settings-configpy)
4. [โครงสร้างโฟลเดอร์ของโปรเจกต์ (Directory Structure)](#-โครงสร้างโฟลเดอร์)
5. [วิธีการติดตั้งและรันโปรแกรม (Installation & How to Run)](#-วิธีการติดตั้งและรันโปรแกรม)
6. [การตั้งค่าโมเดล Local Ollama (qwen2.5-coder)](#-การตั้งค่าโมเดล-local-ollama)

---

## 🚀 ภาพรวมของโปรเจกต์ (Overview)
MiniProject นี้เป็นระบบ **AI Coding Assistant & Graph RAG** ระดับโปรดักชันที่ถูกออกแบบขึ้นเพื่อช่วยเหลือนักพัฒนาในการเขียนโค้ด, วิเคราะห์สถาปัตยกรรม Codebase, ตรวจหาและแก้ไขบั๊ก, อธิบายการทำงานของอัลกอริทึม, สร้างชุดทดสอบ Unit Test ตลอดจนการ Refactor โค้ดให้มีประสิทธิภาพสูงตามหลัก Clean Code

### จุดเด่นหลัก (Key Capabilities):
* **Dynamic Question Router:** คัดแยกประเภทคำถามอัตโนมัติ (เชิงโครงสร้างสัมพันธ์ -> ส่งเข้า Graph DB | เชิงความหมายและคอนเซปต์ -> ส่งเข้า Hybrid Vector RAG)
* **Code Knowledge Graph (Kùzu DB + AST):** สแกน Codebase สร้าง Property Graph (Nodes: Module, Function, Class; Edges: CALLS, IMPORTS, CLASS_DEFINED_IN) ตรวจสอบ Callers/Callees และ Call Hierarchy ได้อย่างแม่นยำ 100%
* **Interactive Mermaid Visualization:** เรนเดอร์แผนผังการเรียกฟังก์ชันและสถาปัตยกรรมแบบกราฟิกสดบนหน้าเว็บ
* **Interactive Cypher Console:** รองรับการเขียนคำสั่ง Cypher Query เพื่อสืบค้นข้อมูลในกราฟได้โดยตรง
* **Hybrid RAG Engine (Sparse BM25 + Dense Semantic):** ผสาน Okapi BM25 เข้ากับ Subword Vector Space และเทคนิค Reciprocal Rank Fusion (RRF) เพื่อดึงอัลกอริทึมจริงจาก GitHub
* **รองรับ Local LLM 100%:** รองรับการเชื่อมต่อกับ Ollama (เช่น `qwen2.5-coder:7b`) รันบนเครื่องตัวเอง ปลอดภัยและฟรี
* **ระบบ Token Streaming & Latency Tracking:** แสดงคำตอบแบบเรียลไทม์ทีละคำ พร้อมจับเวลา Response Latency ทุกข้อความ
* **Live Code Sandbox:** รันและทดสอบโค้ด Python ที่ได้จาก LLM ได้ทันทีในหน้าเว็บ พร้อมตรวจจับเวลาทำงาน และแสดงข้อผิดพลาด (stdout/stderr)
* **Chat History Export:** ดาวน์โหลดประวัติการสนทนาและโค้ดทั้งหมดออกมาเป็นไฟล์ Markdown (.md) ได้ในคลิกเดียว

---

## 🏗️ สถาปัตยกรรมของระบบ (System Architecture)

```
                                  [ ผู้ใช้งาน (User) ]
                                           │
                                    (ป้อน Prompt + เลือกโหมด)
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. STREAMLIT USER INTERFACE (app.py)                                                  │
│    - Tab 1: Coding Assistant & Router (Chat, Model Selector, Latency, Sources Expander) │
│    - Tab 2: Live Code Sandbox (Python Code Editor, Run Trigger, Output Display)        │
│    - Tab 3: Code Knowledge Graph & Graph RAG (Mermaid Visualizer, Callers, Cypher)     │
│    - Tab 4: Internet Resource Dataset & Hybrid Search Playground                       │
│    - Tab 5: System Explanation (เอกสารอธิบายการทำงานแต่ละระบบอย่างละเอียด)           │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. DYNAMIC QUESTION ROUTER (hybrid_retriever.py -> QuestionRouter)                    │
│    - ตรวจสอบคำถามเชิงโครงสร้าง (ใครเรียก, ลำดับการเรียก, caller, callee, import)       │
│    - คัดแยกเส้นทางอัตโนมัติ: [GRAPH MODE] vs [VECTOR / HYBRID MODE]                    │
└────────────────────┬─────────────────────────────────────────────┬─────────────────────┘
                     │                                             │
 (Structural Query)  ▼                                             ▼  (Semantic / Code Gen)
┌──────────────────────────────────────────┐     ┌──────────────────────────────────────────┐
│ 3A. CODE KNOWLEDGE GRAPH (Kùzu DB + AST) │     │ 3B. HYBRID RAG ENGINE (BM25 + Semantic)  │
│  - Property Graph: Module, Function      │     │  - Sparse: Okapi BM25 Lexical Matching   │
│  - Relations: CALLS, IMPORTS             │     │  - Dense: Subword Vector Cosine Sim      │
│  - Call Hierarchy Depth Traversal        │     │  - Fusion: Weighted α / RRF (k=60)       │
│  - Interactive Cypher Query Engine       │     │  - Ground-Truth Code from GitHub         │
└────────────────────┬─────────────────────┘     └─────────────────┬────────────────────────┘
                     │                                             │
                     └─────────────────────┬───────────────────────┘
                                           │ (Combined Ground-Truth Context)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. PROMPT ENGINEERING PIPELINE (prompts.py)                                           │
│    - ผสาน System Persona (Senior Software Engineer) เข้ากับกฎเฉพาะของแต่ละโหมด         │
│    - กำหนดให้โค้ดมี Type Hints, Docstrings, Modular Structure, และ Complexity Analysis  │
│    - รวม Source Code / Error Traceback / Graph Context เข้าเป็น Prompt Bundle เดียวกัน │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 5. LLM INFERENCE ENGINE (llm_engine.py)                                                │
│    ├─ Local Ollama Connector (/api/chat แบบ Streaming)                                 │
│    ├─ Cloud API Connector (OpenAI / DeepSeek / Groq Compatible)                       │
│    └─ Smart Simulation Engine (Fallback Generator เมื่อไม่มี Service ภายนอก)          │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                (ผลลัพธ์โค้ดที่สร้างขึ้น)
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 6. CODE EXTRACTION & SANDBOX EXECUTION (code_runner.py)                                │
│    - ใช้ Regular Expressions กรองเอาเฉพาะ Python Code Block ออกจาก Markdown Text       │
│    - รันโค้ดผ่าน Isolated Subprocess ป้องกันการแครชของเว็บแอป                            │
│    - มี Timeout Guard (10 วินาที) ดักจับ Infinite Loop                                  │
│    - รวบรวม stdout, stderr, execution_time, exit_code ส่งกลับไปแสดงที่หน้า UI          │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔍 คำอธิบายการทำงานของแต่ละระบบขั้นตอน

### ระบบที่ 1: Dynamic Question Router (`hybrid_retriever.py -> QuestionRouter`)
ระบบคัดแยกเจตนาของคำถาม (Query Routing) ตามแนวคิด Router Pattern จาก Session 12:
1. **Rule-Based Structural Pattern:** ดักจับคำสั่งหรือคีย์เวิร์ดที่ถามถึงความเชื่อมโยงเชิงสถาปัตยกรรม (เช่น `ใครเรียก`, `ลำดับการเรียก`, `caller`, `callee`, `hierarchy`, `import`, `โมดูล`, `dependency`)
2. **Dynamic Routing Decision:**
   - **Graph Mode (`graph`):** ส่งคำถามไปค้นหาใน Kùzu Code Knowledge Graph เพื่อตอบโครงสร้างที่แน่นอน 100%
   - **Vector Mode (`vector`):** ส่งคำถามไปค้นหาใน Hybrid Vector RAG เมื่อเป็นคำถามเชิงคอนเซปต์ หรือต้องการสร้างอัลกอริทึม
3. **Transparency:** แสดง Badge `[🔗 Graph Architecture Mode]` หรือ `[🔍 Vector Semantic Mode]` บนหน้าต่างแชทเพื่อให้ผู้ใช้ทราบเส้นทางการประมวลผล

---

### ระบบที่ 2: Code Knowledge Graph Database (`code_graph_engine.py`)
ระบบวิเคราะห์ความสัมพันธ์ของ Codebase ด้วย Graph Database ตามแนวคิด Session 11 & 12:
1. **AST Static Code Parsing:** สแกนไฟล์ Python ในโปรเจกต์ด้วย Python `ast` เพื่อวิเคราะห์ฟังก์ชัน, คลาส, และการนำเข้าโมดูลโดยไม่ต้องสั่งรันโค้ด
2. **Kùzu Embedded Graph Storage:** จัดเก็บลงในฐานข้อมูล Kùzu Graph DB (Property Graph Model)
   - **Nodes:** `Module`, `Function`, `Class`
   - **Edges:** `CALLS` (ฟังก์ชันเรียกฟังก์ชัน), `IMPORTS` (โมดูลนำเข้าโมดูล), `DEFINED_IN`, `CLASS_DEFINED_IN`
3. **Call Hierarchy & Graph Traversal:** สืบสายการเรียกแบบ Recursive Depth Traversal วิเคราะห์ Caller/Callee
4. **Interactive Mermaid Generator:** แปลงผลลัพธ์เป็น Mermaid Diagram สดให้ผู้ใช้ดูบนหน้าเว็บ
5. **Interactive Cypher Console:** เปิดโอกาสให้ผู้ใช้เขียนคำสั่ง Cypher Query เพื่อสำรวจกราฟอย่างอิสระ

---

### ระบบที่ 3: Hybrid RAG & Internet Resource Dataset (`hybrid_retriever.py` & `dataset_manager.py`)
ระบบสืบค้นชุดข้อมูลโค้ดอ้างอิงจากอินเทอร์เน็ต (GitHub Open-Source) แบบ **Hybrid Retrieval-Augmented Generation**:
1. **Sparse Lexical Retrieval (Okapi BM25):** 
   - ใช้อัลกอริทึม Okapi BM25 ($k_1=1.5, b=0.75$) ในการคำนวณคะแนน Term Frequency และ Inverse Document Frequency (IDF)
   - ให้น้ำหนักพิเศษกับ Title ($3\times$), Keywords ($2\times$), Category ($2\times$) เพื่อจับคู่ชื่อฟังก์ชัน, คำย่อ (e.g. `LRU`, `BFS`, `BST`) และ API ตรงตัวได้อย่างสมบูรณ์แบบ
2. **Dense Semantic Retrieval (Vector Space & Cosine Similarity):**
   - แปลงเอกสารและ Query ให้อยู่ใน Subword & Character N-gram Vector Space แบบความหนาแน่นสูง
   - คำนวณ Cosine Similarity ด้วย Matrix Dot Product ระหว่าง Unit Normalized Vectors ช่วยให้เข้าใจบริบท ภาษาไทย (เช่น "ค้นหาทวิภาค", "เรียงลำดับข้อมูล") และคำพ้องความหมาย
3. **Hybrid Fusion & Re-Ranking:**
   - **Weighted Linear Combination:** ผสานคะแนนแบบ $\alpha \cdot Score_{dense} + (1 - \alpha) \cdot Score_{sparse}$ ปรับแต่งค่าน้ำหนัก $\alpha$ ได้อิสระ
   - **Reciprocal Rank Fusion (RRF):** จัดอันดับโดยอาศัยตำแหน่ง $RRF(d) = \sum \frac{1}{60 + Rank_m(d)}$ ตามมาตรฐานอุตสาหกรรม
4. **Data Sync & Fallback:** ดึงโค้ดจริงจาก [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python) และ [tiangolo/fastapi](https://github.com/tiangolo/fastapi) พร้อมระบบแคช Local JSON ปลอดภัยเมื่อไม่มีอินเทอร์เน็ต

---

### ระบบที่ 4: Prompt Engineering Pipeline (`prompts.py`)
ระบบนี้มีหน้าที่แปลงข้อความธรรมดาจากผู้ใช้ให้กลายเป็น Prompt ที่มีความรัดกุม แม่นยำ และชี้นำให้ LLM สร้างโค้ดที่มีคุณภาพสูงสุด:
1. **Base System Persona:** กำหนดบทบาทของ AI ให้เป็น Senior Software Engineer ที่ยึดมั่นใน Clean Code, SOLID Principles และการระบุ Type Annotations
2. **Task-Specific Roles (บทบาทเฉพาะงาน):**
   - **`generate`**: บังคับให้แจกแจงโครงสร้างโค้ด พร้อมตัวอย่างการเรียกใช้งาน (Sample Usage)
   - **`debug`**: กำหนดให้ระบุสาเหตุที่แท้จริงของบั๊ก (Root Cause) อธิบายวิธีแก้ และแสดงโค้ดที่ได้รับการแก้ไขแล้ว
   - **`explain`**: บังคับให้อธิบายแบบ Step-by-Step พร้อมสรุป Big-O Time & Space Complexity
   - **`test`**: กำหนดให้ออกแบบ Unit Test ครอบคลุมทั้งกรณีข้อมูลทั่วไป (Happy Path) และกรณีขอบเขตสุดโต่ง (Edge Cases)
   - **`refactor`**: เน้นวิเคราะห์ Code Smells และปรับปรุงประสิทธิภาพโดยไม่กระทบผลลัพธ์เดิม
3. **Context Injection:** มีการจัดฟอร์แมตแนบโค้ดเดิม, Error Message, Ground-Truth References, และ Graph Context เข้าไปด้วย Markdown block อย่างเป็นระเบียบ

---

### ระบบที่ 5: LLM Inference Engine (`llm_engine.py`)
รับหน้าที่สื่อสารกับโมเดลภาษา โดยมีคุณสมบัติดังนี้:
1. **Ollama Integration:** เชื่อมต่อกับ Ollama ผ่าน REST API `http://localhost:11434/api/chat` ด้วย Python Standard Library (`urllib.request`) โดยตรง จึงไม่ต้องพึ่งพาไลบรารีขนาดใหญ่ภายนอก
2. **Token Streaming:** ใช้ Generator Function (`yield`) เพื่อดึง Token ที่โมเดลสร้างออกมาแสดงผลบนหน้าจอทันทีแบบ Real-time ช่วยลดความรู้สึกในการรอคอยของผู้ใช้
3. **Auto Health Check & Fallback:** มีเมธอด `check_ollama_status()` คอยตรวจสอบว่า Service ของ Ollama เปิดอยู่หรือไม่ และดึงรายการโมเดลที่ดาวน์โหลดไว้มาให้เลือก หากผู้ใช้ยังไม่ได้เปิด Service ระบบจะสลับไปใช้ **Simulation Engine** โดยอัตโนมัติ ทำให้ผู้ใช้สามารถทดสอบฟังก์ชันงานทั้งหมดของ UI ได้ทันทีโดยไม่เจอปัญหาจอขาวหรือ Error แครช

---

### ระบบที่ 6: Interactive Web UI (`app.py`)
หน้าจอส่วนติดต่อผู้ใช้ถูกสร้างด้วย **Streamlit** พร้อมการตกแต่งด้วย CSS สไตล์ Modern Dark Glassmorphism:
1. **Chat & Router Badges:** แสดงผลลัพธ์การคัดแยกเส้นทางอัตโนมัติ พร้อมกล่องขยายดูข้อมูลอ้างอิง (Retrieved Sources)
2. **Code Knowledge Graph Inspector:** แท็บสืบค้นความสัมพันธ์ของฟังก์ชัน, แผนภาพ Mermaid, และคอนโซล Cypher
3. **Session State & Metrics:** ตรวจวัด Query Count, Response Latency, และอัตราส่วนการส่งคำถามไปยัง Graph vs Vector
4. **Chat Export:** ส่งออกประวัติการแชททั้งหมดเป็นไฟล์ Markdown (.md) เพื่อนำไปใช้งานต่อได้ทันที
5. **Sandbox Sync:** ปุ่มส่งโค้ดที่สร้างไปยัง Sandbox ได้ในคลิกเดียว

---

### ระบบที่ 7: Safe Code Execution Sandbox (`code_runner.py`)
ระบบทดสอบการทำงานของโค้ด Python ในสภาพแวดล้อมที่ปลอดภัย:
1. **Regex Code Parser:** ฟังก์ชัน `extract_python_code()` สกัดโค้ดภาษา Python ออกมาจากบล็อก Markdown ` ```python ... ``` ` โดยอัตโนมัติ
2. **Subprocess Isolation:** นำโค้ดไปเขียนลงไฟล์ชั่วคราว (`tempfile`) และสั่งรันผ่าน `subprocess.run()` แยกต่างหาก จึงไม่กระทบต่อ Process หลักของ Streamlit
3. **Timeout Protection:** กำหนดเวลา Timeout สูงสุด 10 วินาที หากโค้ดของผู้ใช้เกิด Infinite Loop ระบบจะตัดการทำงานทันทีและแจ้งเตือนอย่างปลอดภัย
4. **Detailed Telemetry:** แสดงผลทั้งข้อความพิมพ์ออกหน้าจอ (stdout), ข้อความแจ้งเตือนข้อผิดพลาด (stderr), ระยะเวลาที่ใช้ประมวลผล (execution time เป็นวินาที) และ Exit Code

---

### ระบบที่ 8: Configuration & Model Settings (`config.py`)
จุดศูนย์กลางในการตั้งค่าระบบทั้งหมด:
- กำหนด URL และชื่อโมเดลเริ่มต้น เช่น `qwen2.5-coder:7b`
- กำหนดรายการภาษาโปรแกรมที่รองรับ (Python, JavaScript, TypeScript, C++, Java, Go, Rust, SQL ฯลฯ)
- กำหนดค่า Default Hyperparameters (Temperature: 0.2, Max Tokens: 2048)

---

## 📁 โครงสร้างโฟลเดอร์ (Directory Structure)

```
MiniProject/
├── app.py                 # หน้าต่าง Web UI หลัก 5 แท็บ (Assistant, Sandbox, Graph RAG, Dataset, System Docs)
├── code_graph_engine.py   # Code Knowledge Graph Engine (Kùzu DB, AST Parser, Call Hierarchy, Mermaid)
├── hybrid_retriever.py    # Hybrid RAG & QuestionRouter (BM25 + Dense Vector + Query Classifier)
├── dataset_manager.py     # ระบบจัดการคลัง Resource Dataset และเชื่อมต่อ Hybrid RAG
├── config.py              # การตั้งค่าระบบ โมเดล และค่าคงที่
├── dataset/               # โฟลเดอร์จัดเก็บ Local Dataset จาก GitHub
│   └── internet_coding_dataset.json
├── graph_db_storage/      # ไดเรกทอรีเก็บฐานข้อมูล Kùzu Graph Database
├── prompts.py             # Prompt Templates และ Persona ของแต่ละโหมด
├── llm_engine.py          # Engine จัดการการเชื่อมต่อ Ollama / API / Simulation
├── code_runner.py         # Sandbox จำลองการรันโค้ด Python อย่างปลอดภัย
├── requirements.txt       # รายการแพ็กเกจ Python สำหรับ pip
├── pyproject.toml         # ไฟล์กำหนด Configuration สำหรับ uv
├── run.bat                # สคริปต์คลิกเดียวรันโปรแกรมบน Windows
└── README.md              # เอกสารอธิบายระบบและการใช้งานอย่างละเอียด
```

---

## 🛠️ วิธีการติดตั้งและรันโปรแกรม

### วิธีที่ 1: รันด้วย `uv` (แนะนำ - รวดเร็วที่สุด)
เนื่องจากในสภาพแวดล้อมมี `uv` ติดตั้งอยู่แล้ว สามารถรันได้ทันทีโดยไม่ต้องติดตั้ง environment เพิ่มเติม:

```bash
# 1. เข้าไปที่โฟลเดอร์ MiniProject
cd "f:/Selected topic/MiniProject"

# 2. รันแอปพลิเคชัน
uv run streamlit run app.py
```

### วิธีที่ 2: รันด้วย `pip` / Python ปกติ
```bash
# 1. ติดตั้งไลบรารีที่จำเป็น
pip install streamlit python-dotenv

# 2. รันแอปพลิเคชัน
streamlit run app.py
```

### วิธีที่ 3: ดับเบิลคลิกไฟล์ `run.bat`
* สามารถเปิดโฟลเดอร์ `MiniProject` แล้วดับเบิลคลิกที่ไฟล์ **`run.bat`** เพื่อเริ่มโปรแกรมได้ทันที

---

## 🦙 การตั้งค่าโมเดล Local Ollama & ระบบ Auto-run บนหน้าเว็บ

ระบบนี้ถูกพัฒนาให้สามารถ **สั่งรัน `ollama serve` ได้จากหน้าเว็บโดยตรง** โดยไม่ต้องเปิด Terminal:

1. **ระบบ Auto-run 1-Click บนหน้าเว็บ:**
   - หาก Ollama ยังไม่ได้เปิด จะมีปุ่ม **`🚀 สตาร์ท Ollama Server ทันที (Auto-run)`** ทั้งใน Sidebar และในหน้าหลัก Tab 1
   - เมื่อกดปุ่ม ระบบจะส่งสัญญาณรัน `ollama serve` ในเบื้องหลังแบบ Background Detached Process และเชื่อมต่อเซิร์ฟเวอร์ให้อัตโนมัติ
   - มีตัวเลือก Checkbox: **`⚡ Auto-run ทันทีเมื่อเข้าเว็บ`** เพื่อให้เปิด Ollama อัตโนมัติทุกครั้งที่เข้าหน้าเว็บ

2. **ดาวน์โหลดโมเดลสำหรับงานเขียนโค้ด:**
   ```bash
   ollama pull qwen2.5-coder:7b
   ```
   *(หรือโมเดลขนาดเล็กสำหรับเครื่องสเปกประหยัด: `ollama pull qwen2.5-coder:1.5b`)*

3. **สถานะพร้อมใช้งาน:**
   เมื่อ Ollama Server รันแล้ว สถานะจะเปลี่ยนเป็น **🟢 Ollama Online** พร้อมแสดงรายการโมเดลที่ติดตั้งไว้ และสามารถกดสั่ง **🛑 ปิด Server** ได้จาก Sidebar ทันที

