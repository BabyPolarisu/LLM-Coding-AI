"""
config.py — Configuration Settings for MiniProject Coding Assistant
===================================================================
ระบบตั้งค่าส่วนกลางสำหรับ Coding Assistant:
- โมเดล LLM เริ่มต้น (Ollama, OpenAI, Fallback)
- พารามิเตอร์เริ่มต้นสำหรับการ Generate (Temperature, Max Tokens)
- รายการภาษาโปรแกรมที่รองรับ
"""

import os
from pathlib import Path

# พาธโปรเจกต์
BASE_DIR = Path(__file__).resolve().parent

# ตั้งค่า Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL", "qwen2.5-coder:7b")

# รายชื่อ Local Model ยอดนิยมสำหรับงาน Coding
POPULAR_MODELS = [
    "qwen2.5-coder:14b",
    "qwen2.5-coder:7b",
    "qwen2.5-coder:1.5b",
    "qwen2.5-coder:32b",
    "deepseek-coder:14b",
    "deepseek-coder:6.7b",
    "codellama:13b",
    "codellama:7b",
    "llama3.2:3b",
    "mistral:7b"
]

# ภาษาโปรแกรมที่รองรับในโหมดต่างๆ
SUPPORTED_LANGUAGES = [
    "Python",
    "JavaScript",
    "TypeScript",
    "HTML / CSS",
    "C++",
    "C#",
    "Java",
    "Go",
    "Rust",
    "SQL",
    "PHP",
    "Bash / Shell"
]

# โหมดการทำงานของ Coding Assistant
ASSISTANT_MODES = {
    "generate": "✨ สร้างโค้ดใหม่ (Code Generator)",
    "explain": "📖 อธิบายการทำงานโค้ด (Code Explainer)",
    "debug": "🐞 แก้ไขบั๊ก & ตรวจจับข้อผิดพลาด (Bug Fixer)",
    "test": "🧪 สร้างชุดทดสอบ (Unit Test Generator)",
    "refactor": "⚡ ปรับปรุงโค้ดให้คลีนและเร็วขึ้น (Code Refactoring)"
}

# พารามิเตอร์เริ่มต้น
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 2048
