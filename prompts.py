"""
prompts.py — Prompt Engineering System for Coding Assistant
============================================================
ระบบจัดการ Template และ System Prompts เฉพาะทางสำหรับงานพัฒนาโปรแกรม:
1. Code Generator (สร้างโค้ดตาม Requirement)
2. Code Explainer (อธิบาย Logic, Data Structure และ Big-O)
3. Bug Fixer (วิเคราะห์ Error, Root Cause, และเสนอโค้ดที่แก้แล้ว)
4. Unit Test Generator (สร้าง Test Cases และ Edge Cases)
5. Code Refactorer (Clean Code, PEP8, Performance Optimization)
"""

from typing import Dict, Any, Optional

# System Persona หลักสำหรับ AI Software Engineer
BASE_SYSTEM_PROMPT = """You are an expert Senior Full-Stack Software Engineer and AI Coding Assistant.
Your core principles:
1. Provide accurate, clean, robust, and idiomatic code according to best practices and industry standards.
2. Always write code with proper type annotations, clear docstrings, and meaningful variable names.
3. Structure your explanations in a clear, step-by-step manner (ภาษาไทย หรือ ภาษาอังกฤษตามคำถามของผู้ใช้).
4. Enclose all code blocks with appropriate markdown syntax highlighting (e.g. ```python, ```javascript).
5. Highlight potential edge cases, security considerations, and computational complexity (Time & Space Complexity) where relevant.
6. When generating Mermaid diagrams (```mermaid), ALWAYS wrap node labels containing parentheses, arithmetic operators, or special symbols in double quotes (e.g., node["mid = (left + right) // 2"] instead of unquoted brackets) to guarantee valid syntax.
"""

# Specialized System Prompts ตามโหมดการทำงาน
MODE_SYSTEM_PROMPTS = {
    "generate": """You are an expert Code Generator.
Given the user's task or requirement:
1. Clarify the solution design briefly.
2. Provide the complete, working, bug-free implementation code in the target programming language.
3. Include clear comments, type hints, error handling, and modular structure.
4. Provide a sample usage demonstration showing input and expected output.
""",

    "explain": """You are an expert Code Tutor and Explainer.
Given the user's code snippet:
1. Summarize the high-level purpose of the code.
2. Explain the code step-by-step (line-by-line or function-by-function).
3. Analyze Time Complexity and Space Complexity (Big-O notation).
4. Point out any notable design patterns or algorithms used.
5. Provide helpful tips or potential improvements.
""",

    "debug": """You are an expert Debugger and Code Reviewer.
Given the buggy code and/or error message:
1. Identify the root cause of the error or bug clearly.
2. Explain why the bug occurred and what went wrong.
3. Provide the corrected, working code solution.
4. Explain the key changes made to prevent this error in the future.
""",

    "test": """You are a Quality Assurance and Test Automation Specialist.
Given the target code:
1. Design comprehensive unit tests covering:
   - Happy paths (typical valid cases)
   - Edge cases (null, empty, negative, boundary values)
   - Failure / Exception handling paths
2. Use standard testing frameworks (e.g., pytest for Python, Jest for JS, Google Test for C++).
3. Include clear test descriptions and assertions.
""",

    "refactor": """You are a Software Architect and Clean Code Specialist.
Given the existing code:
1. Analyze code smells, redundancy, performance bottlenecks, or readability issues.
2. Refactor the code adhering to Clean Code, SOLID principles, and language conventions (e.g. PEP 8).
3. Provide the modernized, optimized code.
4. List the exact refactoring improvements made and their benefits (e.g., speed, memory, maintainability).
"""
}


def build_prompt_bundle(
    mode: str,
    user_prompt: str,
    language: str = "Python",
    code_context: Optional[str] = None,
    error_message: Optional[str] = None,
    dataset_references: Optional[list] = None
) -> Dict[str, str]:
    """
    สร้าง System Prompt และ Human Prompt แบบสมบูรณ์ตามโหมดและข้อมูลที่ผู้ใช้ระบุ
    พร้อมรองรับการแทรก Reference จาก Resource Dataset ที่ดึงจากอินเทอร์เน็ต
    """
    system_text = BASE_SYSTEM_PROMPT + "\n" + MODE_SYSTEM_PROMPTS.get(mode, MODE_SYSTEM_PROMPTS["generate"])
    
    # ประกอบ User Message
    user_parts = [
        f"**Target Programming Language:** {language}",
        f"**User Requirement / Question:**\n{user_prompt.strip()}"
    ]

    # แทรก Context จาก Hybrid RAG Resource Dataset (ถ้ามี)
    if dataset_references:
        ref_text = "### 📚 Retrieved Ground-Truth Code Reference (จาก Hybrid RAG Dataset):\n"
        for ref in dataset_references:
            h_score = ref.get("hybrid_score", "")
            m_type = ref.get("match_type", "")
            meta_str = f" [Match: {m_type}, Score: {h_score}]" if h_score != "" else ""
            ref_text += f"- **Algorithm:** {ref.get('title')}{meta_str} ({ref.get('complexity', 'N/A')})\n"
            ref_text += f"  **Source URL:** {ref.get('source_url')}\n"
            sample_code = ref.get('code', '')[:1200]
            ref_text += f"  ```python\n{sample_code}\n  ```\n"
        ref_text += "\n*Please use the above reference as algorithmic inspiration while fulfilling the user's specific request.*"
        user_parts.append(ref_text)
    
    if code_context and code_context.strip():
        user_parts.append(f"\n**Source Code Context:**\n```{language.lower()}\n{code_context.strip()}\n```")
        
    if error_message and error_message.strip():
        user_parts.append(f"\n**Error Message / Stack Trace:**\n```text\n{error_message.strip()}\n```")
        
    full_user_prompt = "\n\n".join(user_parts)
    
    return {
        "system": system_text,
        "user": full_user_prompt
    }

