"""
code_runner.py — Safe Code Execution Sandbox for Python
========================================================
ระบบทดสอบและรันโค้ด Python ที่สร้างขึ้น:
- รันผ่าน isolated subprocess
- มีระบบ Timeout ป้องกัน infinite loop
- ดักจับ stdout, stderr, execution time และ exit code
- แยกโค้ดออกจาก Markdown codeblock อัตโนมัติ
"""

import sys
import subprocess
import tempfile
import time
import re
from typing import Dict, Any, Optional
from pathlib import Path


def extract_python_code(markdown_text: str) -> str:
    """
    ดึงเฉพาะบล็อกโค้ด Python จากคำตอบที่เป็น Markdown
    รองรับทั้ง ```python, ```py, บล็อกปิด, บล็อกเปิดค้าง (unclosed), และโค้ด Python โดยตรง
    """
    if not markdown_text:
        return ""

    text = str(markdown_text).strip()

    # 1. ค้นหาบล็อกที่มีระบุภาษา python หรือ py
    py_pattern = re.compile(r"```(?:python|py)\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)
    matches = py_pattern.findall(text)
    if matches:
        candidates = [m.strip() for m in matches if m.strip()]
        if candidates:
            return max(candidates, key=len)

    # 2. ค้นหาบล็อกโค้ดทั่วไป ``` ... ```
    general_pattern = re.compile(r"```(?:\w*)\s*\n?(.*?)\n?```", re.DOTALL)
    matches = general_pattern.findall(text)
    if matches:
        candidates = [m.strip() for m in matches if m.strip()]
        if candidates:
            return max(candidates, key=len)

    # 3. จัดการกรณีบล็อกเปิดค้าง (Unclosed code block e.g. streaming cutoff)
    unclosed_py = re.compile(r"```(?:python|py)\s*\n?(.*)", re.DOTALL | re.IGNORECASE)
    match_unclosed = unclosed_py.search(text)
    if match_unclosed and match_unclosed.group(1).strip():
        return match_unclosed.group(1).strip()

    unclosed_gen = re.compile(r"```(?:\w*)\s*\n?(.*)", re.DOTALL)
    match_unclosed = unclosed_gen.search(text)
    if match_unclosed and match_unclosed.group(1).strip():
        return match_unclosed.group(1).strip()

    # 4. หากข้อความมีลักษณะเป็นโค้ด Python โดยตรง
    lines = text.splitlines()
    code_keywords = ("def ", "import ", "from ", "class ", "print(", "if __name__", "#", "async def ")
    if any(any(line.strip().startswith(kw) for kw in code_keywords) for line in lines):
        return text

    return ""


def run_python_code(code: str, timeout_seconds: int = 10) -> Dict[str, Any]:
    """
    รันโค้ด Python ใน subprocess ชั่วคราว และดักจับผลลัพธ์
    
    Args:
        code: โค้ด Python ที่ต้องการรัน
        timeout_seconds: เวลาสูงสุดที่อนุญาตให้รัน (default: 10 วินาที)
        
    Returns:
        Dict:
            - success: bool
            - stdout: str
            - stderr: str
            - execution_time: float (seconds)
            - exit_code: int
    """
    cleaned_code = extract_python_code(code)
    if not cleaned_code:
        cleaned_code = code.strip()

    
    # สร้าง temp file เพื่อรัน
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as temp_file:
        temp_file.write("# -*- coding: utf-8 -*-\n")
        temp_file.write(cleaned_code)
        temp_file_path = temp_file.name

    start_time = time.perf_counter()
    try:
        process = subprocess.run(
            [sys.executable, temp_file_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            encoding="utf-8",
            errors="replace"
        )
        elapsed_time = time.perf_counter() - start_time
        
        return {
            "success": process.returncode == 0,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "execution_time": round(elapsed_time, 3),
            "exit_code": process.returncode,
            "cleaned_code": cleaned_code
        }
    except subprocess.TimeoutExpired:
        elapsed_time = time.perf_counter() - start_time
        return {
            "success": False,
            "stdout": "",
            "stderr": f"⏱️ Execution Timeout: โค้ดใช้เวลาทำงานเกิน {timeout_seconds} วินาที (อาจมี Infinite Loop)",
            "execution_time": round(elapsed_time, 3),
            "exit_code": -1,
            "cleaned_code": cleaned_code
        }
    except Exception as e:
        elapsed_time = time.perf_counter() - start_time
        return {
            "success": False,
            "stdout": "",
            "stderr": f"❌ Error running script: {str(e)}",
            "execution_time": round(elapsed_time, 3),
            "exit_code": -1,
            "cleaned_code": cleaned_code
        }
    finally:
        # ลบไฟล์ temp เมื่อเสร็จ
        try:
            Path(temp_file_path).unlink(missing_ok=True)
        except Exception:
            pass
