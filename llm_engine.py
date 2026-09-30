"""
llm_engine.py — Core LLM Inference Engine for Coding Assistant
=============================================================
ระบบจัดการการเชื่อมต่อและ Inference ของโมเดล LLM:
1. Local Ollama Engine (เช่น qwen2.5-coder, codellama, deepseek-coder) ผ่าน REST API
2. OpenAI-Compatible API Engine (รองรับ API key จากภายนอก)
3. Offline Simulation Engine (Fallback อัจฉริยะ สำหรับสาธิตและทดสอบได้ทันทีแม้ไม่ได้เปิด Ollama)
"""

import json
import time
import os
import shutil
import subprocess
from typing import Generator, List, Dict, Any, Optional
import urllib.request
import urllib.error

from config import OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL


class LLMEngine:
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip("/")

    @staticmethod
    def is_ollama_installed() -> bool:
        """ตรวจสอบว่ามีโปรแกรม Ollama ติดตั้งอยู่ในเครื่องหรือไม่"""
        if shutil.which("ollama"):
            return True
        common_paths = [
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
            r"C:\Program Files\Ollama\ollama.exe"
        ]
        return any(os.path.exists(p) for p in common_paths)
        
    def check_ollama_status(self) -> Dict[str, Any]:
        """
        ตรวจสอบสถานะการทำงานของ Ollama และดึงรายชื่อโมเดลที่ติดตั้งไว้
        """
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = [m.get("name") for m in data.get("models", [])]
                    return {
                        "online": True,
                        "models": models,
                        "message": f"Ollama Online (พบ {len(models)} โมเดล)"
                    }
        except Exception as e:
            return {
                "online": False,
                "models": [],
                "message": f"Ollama ไม่พร้อมใช้งาน ({str(e)})"
            }

    def start_ollama_service(self, timeout: int = 10) -> Dict[str, Any]:
        """
        สั่งเปิด service 'ollama serve' ในพื้นหลังแบบ Detached Process
        พร้อมรอตรวจสอบจนกว่า Service จะ Online
        """
        initial_status = self.check_ollama_status()
        if initial_status["online"]:
            return {
                "success": True,
                "message": "Ollama Server กำลังทำงานอยู่แล้ว (Port 11434)"
            }

        ollama_bin = shutil.which("ollama")
        if not ollama_bin:
            fallback = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
            if os.path.exists(fallback):
                ollama_bin = fallback
            else:
                return {
                    "success": False,
                    "message": "ไม่พบโปรแกรม Ollama ในระบบ กรุณาติดตั้งจาก https://ollama.com"
                }

        try:
            creationflags = 0
            if os.name == "nt":
                # DETACHED_PROCESS (0x00000008) + CREATE_NO_WINDOW (0x08000000)
                creationflags = 0x00000008 | 0x08000000

            subprocess.Popen(
                [ollama_bin, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
                close_fds=True
            )
        except Exception as e:
            return {
                "success": False,
                "message": f"เกิดข้อผิดพลาดในการเปิด Ollama: {str(e)}"
            }

        start_wait = time.time()
        while time.time() - start_wait < timeout:
            time.sleep(0.5)
            status = self.check_ollama_status()
            if status["online"]:
                return {
                    "success": True,
                    "message": f"เปิด Ollama Server สำเร็จ! ({status['message']})"
                }

        return {
            "success": False,
            "message": f"ส่งคำสั่ง 'ollama serve' แล้ว แต่เซิร์ฟเวอร์ยังไม่ตอบรับภายใน {timeout} วินาที"
        }

    def stop_ollama_service(self) -> Dict[str, Any]:
        """
        สั่งปิด service Ollama (สำหรับ Windows taskkill)
        """
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/IM", "ollama.exe"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False
                )
                subprocess.run(
                    ["taskkill", "/F", "/IM", "ollama_app.exe"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False
                )
            time.sleep(1)
            status = self.check_ollama_status()
            if not status["online"]:
                return {"success": True, "message": "ปิด Ollama Server เรียบร้อยแล้ว"}
            return {"success": False, "message": "Ollama ยังคงทำงานอยู่"}
        except Exception as e:
            return {"success": False, "message": f"ไม่สามารถปิด Ollama ได้: {str(e)}"}


    def generate_stream_ollama(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2048
    ) -> Generator[str, None, None]:
        """
        Stream output จาก Ollama API (/api/chat)
        """
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            },
            "stream": True
        }
        
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for line in resp:
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
        except Exception as e:
            yield f"\n\n⚠️ **เกิดข้อผิดพลาดในการเชื่อมต่อ Ollama:** {str(e)}\n"
            yield "\n*แนะนำ: ตรวจสอบว่า `ollama serve` กำลังทำงานอยู่ และมีโมเดลติดตั้งไว้แล้ว*"

    def generate_stream_openai(
        self,
        api_key: str,
        base_url: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2
    ) -> Generator[str, None, None]:
        """
        Stream output จาก OpenAI-compatible endpoint
        """
        endpoint = f"{base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "stream": True
        }
        
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
        )
        
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                for line in resp:
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: "):
                        data_str = line_str[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield f"\n\n⚠️ **เกิดข้อผิดพลาดในการเรียก API:** {str(e)}"

    def generate_stream_simulation(
        self,
        mode: str,
        user_prompt: str,
        language: str = "Python",
        code_context: Optional[str] = None,
        dataset_references: Optional[list] = None
    ) -> Generator[str, None, None]:
        """
        Simulation Mode: จำลองการสร้างโค้ดและคำอธิบายอย่างสมบูรณ์แบบ
        สำหรับใช้ทดสอบการทำงานของระบบและ UI ในกรณีที่ผู้ใช้ยังไม่ได้เปิด Ollama
        """
        citation_block = ""
        if dataset_references:
            citation_block = "🌐 **แหล่งอ้างอิง Hybrid RAG Resource Dataset (TheAlgorithms / GitHub):**\n"
            for ref in dataset_references:
                h_score = ref.get("hybrid_score")
                bm25_s = ref.get("bm25_score")
                dense_s = ref.get("dense_score")
                m_type = ref.get("match_type", "Hybrid")
                
                score_detail = ""
                if h_score is not None:
                    score_detail = f" — `{m_type}` [Hybrid: **{h_score}** | BM25: **{bm25_s}** | Cosine: **{dense_s}**]"
                
                citation_block += f"- 🔗 [{ref['title']}]({ref['source_url']}){score_detail} *(Category: {ref['category']} | {ref.get('complexity', '')})*\n"
            citation_block += "\n"

        intro = (
            f"💡 *[โหมดจำลองคำตอบ — เพื่อทดสอบระบบ UI ทันที]*\n\n"
            f"{citation_block}"
            f"### 📋 สรุปการวิเคราะห์สำหรับคำสั่ง:\n"
            f"> \"{user_prompt}\"\n\n"
            f"**ภาษาโปรแกรม:** `{language}`  |  **โหมด:** `{mode.upper()}`\n\n"
            f"---\n\n"
        )
        for char in intro:
            yield char
            time.sleep(0.005)

        # จำลองโค้ดตามโหมด
        if mode == "generate":
            body = f"""### 🛠️ การออกแบบและวิธีแก้ปัญหา
1. **ออกแบบ Function / Class**: มีการกำหนด Type Hints และ Docstrings เพื่อความถูกต้อง
2. **จัดการ Edge Cases**: มีการตรวจสอบ Input ไม่ให้เป็นค่าว่างหรือผิดประเภท
3. **ประสิทธิภาพ**: ออกแบบให้ใช้ Memory และเวลาประมวลผลอย่างเหมาะสม

```python
from typing import List, Dict, Any, Optional

class SolutionManager:
    \"\"\"
    โซลูชันสำหรับ: {user_prompt}
    ออกแบบตามหลัก Clean Code และ SOLID Principles
    \"\"\"
    def __init__(self, name: str = "MiniProject CodeEngine"):
        self.name = name
        self._history: List[str] = []

    def execute_task(self, data: Optional[List[int]] = None) -> Dict[str, Any]:
        \"\"\"
        ประมวลผลข้อมูลตามข้อกำหนดที่ระบุ
        \"\"\"
        if data is None:
            data = [10, 20, 30, 40, 50]
            
        # ตัวอย่างการคำนวณและประมวลผล
        total_sum = sum(data)
        average = total_sum / len(data) if data else 0.0
        
        result = {{
            "task": "{user_prompt}",
            "count": len(data),
            "sum": total_sum,
            "average": average,
            "status": "success"
        }}
        
        self._history.append(f"Processed {{len(data)}} items")
        return result

# ─── ตัวอย่างการเรียกใช้งาน (Sample Usage) ───
if __name__ == "__main__":
    solver = SolutionManager()
    output = solver.execute_task([12, 45, 68, 23, 89])
    print("ผลลัพธ์การทำงาน:")
    for key, val in output.items():
        print(f"  • {{key}}: {{val}}")
```

### ⏱️ การวิเคราะห์ความซับซ้อน (Complexity Analysis)
* **Time Complexity:** $O(n)$ เนื่องจากทำการวนรอบข้อมูล 1 ครั้ง
* **Space Complexity:** $O(1)$ ใช้หน่วยความจำคงที่ ไม่มีการสร้าง Array สำเนาใหม่
"""
        elif mode == "debug":
            body = f"""### 🔍 การระบุปัญหาและสาเหตุของบั๊ก
1. **จุดที่พบข้อผิดพลาด**: ตรวจพบการอ้างอิง Index นอกช่วงขอบเขต (IndexError) หรือการเข้าถึงตัวแปรที่เป็น `None`
2. **สาเหตุ**: ขาดการตรวจสอบความถูกต้องของพารามิเตอร์ขาเข้า (Input Validation) ก่อนการประมวลผล

### 💡 โค้ดที่ได้รับการแก้ไขแล้ว (Corrected Code):
```python
def fixed_function(items: list) -> list:
    \"\"\"
    ฟังก์ชันที่แก้ไขข้อผิดพลาดแล้ว ปลอดภัยต่อค่า None และ Empty List
    \"\"\"
    if not items:
        return []
    
    # ดำเนินการอย่างปลอดภัย
    return [x * 2 for x in items if isinstance(x, (int, float))]

if __name__ == "__main__":
    test_cases = [[1, 2, 3], [], None]
    for tc in test_cases:
        print(f"Input: {{tc}} -> Output: {{fixed_function(tc)}}")
```
"""
        elif mode == "test":
            body = f"""### 🧪 ชุดการทดสอบ (Unit Test Suite)
```python
import unittest

class TestSolution(unittest.TestCase):
    def test_standard_input(self):
        \"\"\"ทดสอบกรณีทั่วไป (Happy Path)\"\"\"
        sample = [1, 2, 3]
        self.assertEqual(len(sample), 3)

    def test_empty_input(self):
        \"\"\"ทดสอบกรณีข้อมูลว่าง (Edge Case)\"\"\"
        empty = []
        self.assertEqual(len(empty), 0)

    def test_boundary_values(self):
        \"\"\"ทดสอบค่าขอบเขต (Boundary Value Analysis)\"\"\"
        self.assertTrue(100 > 0)

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
```
"""
        else:
            body = f"""### 📖 คำอธิบายระบบและการทำงาน
1. โค้ดนี้ถูกออกแบบมาเพื่อรองรับคำสั่ง **{user_prompt}**
2. มีโครงสร้างเป็นระเบียบ แบ่งหน้าที่ตาม Single Responsibility Principle
3. สามารถนำไปรันบนแท็บ **'⚡ ทดสอบรันโค้ด'** ได้ทันที
"""

        # ทำ streaming เลียนแบบ token-by-token
        chunk_size = 6
        for i in range(0, len(body), chunk_size):
            yield body[i:i + chunk_size]
            time.sleep(0.01)
