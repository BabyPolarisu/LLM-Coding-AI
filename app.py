"""
app.py — Main Streamlit Application for MiniProject Coding Assistant
===================================================================
เว็บแอปพลิเคชัน Coding Assistant ที่มี UI รองรับการรับ Prompt สำหรับงานเขียนโค้ด:
1. แท็บ '💬 Coding Assistant': รับ Prompt, เลือกโหมด, พร้อมระบบ RAG ดึง Resource Dataset จาก Internet
2. แท็บ '⚡ ทดสอบรันโค้ด (Sandbox)': รันโค้ด Python ที่ได้จาก LLM ทันที พร้อมแสดง stdout/stderr
3. แท็บ '🌐 คลัง Resource Dataset': ดู Source Code และอัลกอริทึมจริงที่ดึงมาจาก Internet/GitHub
4. แท็บ '📚 คำอธิบายระบบขั้นตอน': อธิบายสถาปัตยกรรมและการสร้างแต่ละขั้นตอนอย่างละเอียด
"""

import re
import streamlit as st
import streamlit.components.v1 as components
import time
from typing import Optional

from config import (
    DEFAULT_OLLAMA_MODEL,
    POPULAR_MODELS,
    SUPPORTED_LANGUAGES,
    ASSISTANT_MODES,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    OLLAMA_BASE_URL
)
from prompts import build_prompt_bundle
from llm_engine import LLMEngine
from code_runner import run_python_code, extract_python_code
from dataset_manager import DatasetManager
from code_graph_engine import CodeGraphEngine
from hybrid_retriever import QuestionRouter


def extract_mermaid_code(markdown_text: str) -> Optional[str]:
    """ดึงโค้ด Mermaid จากคำตอบของ LLM เพื่อนำไปเรนเดอร์เป็นภาพไดอะแกรมสด"""
    if not markdown_text:
        return None
    pattern = re.compile(r"```(?:mermaid)\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)
    match = pattern.search(str(markdown_text))
    if match and match.group(1).strip():
        return match.group(1).strip()
    return None


def sanitize_mermaid_code(code: str) -> str:
    """
    แก้ไข Syntax ที่พบบ่อยใน Mermaid โดยเฉพาะวงเล็บหรือเครื่องหมายพิเศษภายใน Node:
    เช่น C[mid = (left + right) // 2] -> C["mid = (left + right) // 2"]
    """
    if not code:
        return ""
    lines = []
    for line in code.splitlines():
        # แก้ไข Node [ ... ] ที่มีวงเล็บหรือสัญลักษณ์ข้างในแต่ไม่ได้ใส่คำพูดครอบ
        def fix_brackets(m):
            node_id = m.group(1)
            content = m.group(2).strip()
            if content.startswith('"') and content.endswith('"'):
                return m.group(0)
            if any(ch in content for ch in "()[]{}/<>*&%$#@!,;:"):
                content = content.replace('"', "'")
                return f'{node_id}["{content}"]'
            return m.group(0)

        # แก้ไข Node { ... } (Diamond Decision) ที่มีเครื่องหมายพิเศษ
        def fix_braces(m):
            node_id = m.group(1)
            content = m.group(2).strip()
            if content.startswith('"') and content.endswith('"'):
                return m.group(0)
            if any(ch in content for ch in "()[]{}/<>*&%$#@!,;:"):
                content = content.replace('"', "'")
                return f'{node_id}{{"{content}"}}'
            return m.group(0)

        line = re.sub(r'([A-Za-z0-9_]+)\[([^\]\n]+)\]', fix_brackets, line)
        line = re.sub(r'([A-Za-z0-9_]+)\{([^\}\n]+)\}', fix_braces, line)
        lines.append(line)
    return "\n".join(lines)


def render_mermaid(code: str, height: int = 360):
    """เรนเดอร์ Mermaid Diagram เป็นกราฟฟิก Interactive SVG แบบสดๆ พร้อมระบบ Auto-Fix Syntax และ Error Catching"""
    import html
    sanitized_code = sanitize_mermaid_code(code.strip())
    escaped_code = html.escape(sanitized_code)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ 
                startOnLoad: false, 
                theme: 'neutral',
                securityLevel: 'loose',
                flowchart: {{ useMaxWidth: true, htmlLabels: true, curve: 'basis' }}
            }});

            async function renderDiagram() {{
                const target = document.getElementById('mermaid-target');
                const rawCode = document.getElementById('raw-mermaid-data').value;
                try {{
                    const uniqueId = 'mermaid_' + Math.random().toString(36).substring(2, 9);
                    const {{ svg }} = await mermaid.render(uniqueId, rawCode);
                    target.innerHTML = svg;
                }} catch (err) {{
                    console.warn("Mermaid render error:", err);
                    target.innerHTML = `
                        <div style="background: #fff1f2; border: 1px solid #fecdd3; border-radius: 8px; padding: 12px; font-size: 13px; color: #9f1239; text-align: left; max-width: 95%; margin: auto;">
                            <b>⚠️ ไม่สามารถวาดไดอะแกรมได้เนื่องจากรูปแบบ Mermaid ผิดพลาด (Syntax Error):</b><br>
                            <code>${{err.message || err}}</code><br><br>
                            <span style="color: #475569;">💡 <b>วิธีแก้ไข:</b> หากในชื่อกล่องมีเครื่องหมายวงเล็บ <code>( )</code> หรือเครื่องหมายคำนวณ ให้ใส่เครื่องหมายคำพูดครอบ เช่น <code>C["mid = (left + right) // 2"]</code></span>
                        </div>
                    `;
                }}
            }}
            window.addEventListener('DOMContentLoaded', renderDiagram);
        </script>
        <style>
            body {{
                margin: 0;
                padding: 10px;
                background-color: transparent;
                display: flex;
                justify-content: center;
                align-items: center;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }}
            .mermaid-target {{
                width: 100%;
                text-align: center;
                overflow: auto;
            }}
            svg {{
                max-width: 100% !important;
                height: auto !important;
            }}
        </style>
    </head>
    <body>
        <textarea id="raw-mermaid-data" style="display:none;">{escaped_code}</textarea>
        <div id="mermaid-target" class="mermaid-target">
            <span style="color: #64748b; font-size: 13px;">⏳ กำลังวาดแผนภาพ...</span>
        </div>
    </body>
    </html>
    """
    components.html(html_content, height=height, scrolling=True)


def generate_chat_markdown(chat_history: list) -> str:
    """แปลงประวัติการสนทนาเป็นไฟล์ Markdown เพื่อให้ผู้ใช้ดาวน์โหลดได้สะดวก"""
    lines = [
        "# 📝 AI Coding Assistant — บันทึกประวัติการสนทนา (Chat Export)",
        f"- **วันที่ส่งออก:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **จำนวนข้อความ:** {len(chat_history)} รายการ",
        "\n---\n"
    ]
    for idx, msg in enumerate(chat_history, 1):
        role_label = "🧑‍💻 ผู้ใช้งาน (User)" if msg["role"] == "user" else "🤖 AI Assistant"
        lines.append(f"### ข้อความที่ {idx}: {role_label}")
        lines.append(f"\n{msg['content']}\n")
        if msg.get("code_snippet"):
            lines.append("```python")
            lines.append(msg["code_snippet"])
            lines.append("```\n")
        if msg.get("latency"):
            lines.append(f"> ⏱️ เวลาตอบสนอง: `{msg['latency']:.2f} วินาที`\n")
        lines.append("\n---\n")
    return "\n".join(lines)


def set_sandbox_code(code_str: str):
    """อัปเดตโค้ดไปยัง Sandbox State และ Editor Key ให้ตรงกันเสมอ พร้อมแสดงผลทันที"""
    code_clean = str(code_str).strip()
    st.session_state["sandbox_code"] = code_clean
    st.session_state["sandbox_editor"] = code_clean


# ─── ตั้งค่าหน้าเว็บ Streamlit ────────────────────────────────────────────────

st.set_page_config(
    page_title="Coding Assistant LLM — MiniProject",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS เพื่อความสวยงามและทันสมัย ──────────────────────────────────────
st.markdown("""
<style>
    /* Gradient Header */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    /* Feature Badge */
    .status-badge {
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-online { background-color: #10b98122; color: #10b981; border: 1px solid #10b98155; }
    .status-offline { background-color: #f59e0b22; color: #f59e0b; border: 1px solid #f59e0b55; }
    
    /* Dataset Tag */
    .dataset-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #3b82f633;
        color: #60a5fa;
        border: 1px solid #3b82f655;
        margin-right: 6px;
    }
    
    /* Hybrid RAG Score Pills */
    .hybrid-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .pill-hybrid { background-color: #8b5cf622; color: #c084fc; border: 1px solid #8b5cf655; }
    .pill-bm25 { background-color: #3b82f622; color: #60a5fa; border: 1px solid #3b82f655; }
    .pill-dense { background-color: #10b98122; color: #34d399; border: 1px solid #10b98155; }
    .pill-match { background-color: #f59e0b22; color: #fbbf24; border: 1px solid #f59e0b55; }

    /* System Card */
    .system-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Hybrid Card */
    .hybrid-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #475569;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── เริ่มต้น Engine & Auto-run Ollama Serve ตั้งแต่เปิดเว็บ ──────────────────
engine = LLMEngine(base_url=OLLAMA_BASE_URL)
dataset_mgr = DatasetManager()

# ตรวจสอบและสั่งรัน 'ollama serve' อัตโนมัติทันทีตั้งแต่เปิดเว็บ
if "ollama_startup_checked" not in st.session_state:
    st.session_state.ollama_startup_checked = True
    initial_ollama_status = engine.check_ollama_status()
    if not initial_ollama_status["online"]:
        engine.start_ollama_service(timeout=8)

# ─── Session State Management ─────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "sandbox_code" not in st.session_state:
    st.session_state.sandbox_code = (
        "# พิมพ์หรือวางโค้ด Python ที่นี่เพื่อทดสอบรัน\n"
        "def greet(name: str) -> str:\n"
        "    return f'สวัสดี, {name}! ยินดีต้อนรับสู่ Coding Assistant'\n\n"
        "if __name__ == '__main__':\n"
        "    print(greet('Developer'))\n"
    )
if "sandbox_editor" not in st.session_state:
    st.session_state.sandbox_editor = st.session_state.sandbox_code
if "prefilled_prompt" not in st.session_state:
    st.session_state.prefilled_prompt = ""

@st.cache_resource
def get_graph_engine() -> CodeGraphEngine:
    ge = CodeGraphEngine()
    try:
        stats = ge.get_stats()
        if stats.get("functions", 0) == 0:
            ge.index_codebase(".")
    except Exception:
        pass
    return ge

@st.cache_resource
def get_question_router() -> QuestionRouter:
    return QuestionRouter()

if "graph_engine" not in st.session_state:
    st.session_state.graph_engine = get_graph_engine()
if "question_router" not in st.session_state:
    st.session_state.question_router = get_question_router()
if "session_stats" not in st.session_state:
    st.session_state.session_stats = {
        "total_queries": 0,
        "total_time": 0.0,
        "graph_routed": 0,
        "vector_routed": 0
    }


# ─── Sidebar: ตั้งค่าและสถานะระบบ ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ การตั้งค่าระบบ LLM")
    
    # Provider Selection (เริ่มต้นเป็น Ollama Local LLM)
    provider = st.radio(
        "เลือก Engine ในการประมวลผล:",
        ["Ollama (Local LLM)", "OpenAI / Custom API", "จำลองคำตอบ (Simulation Demo)"],
        index=0
    )
    
    selected_model = DEFAULT_OLLAMA_MODEL
    api_key = ""
    api_base = "https://api.openai.com/v1"
    
    if provider == "Ollama (Local LLM)":
        status_info = engine.check_ollama_status()
        
        # หากยังออฟไลน์ ให้สั่งสตาร์ทให้อัตโนมัติทันที
        if not status_info["online"]:
            with st.spinner("⚡ กำลังเชื่อมต่อและเริ่มรัน Ollama Server อัตโนมัติ..."):
                start_auto = engine.start_ollama_service(timeout=6)
                if start_auto["success"]:
                    status_info = engine.check_ollama_status()

        if status_info["online"]:
            c_badge, c_stop = st.columns([2, 1])
            with c_badge:
                st.markdown(f'<span class="status-badge status-online">🟢 {status_info["message"]} (Auto-run)</span>', unsafe_allow_html=True)
            with c_stop:
                if st.button("🛑 ปิด Server", key="stop_ollama_sidebar_btn", help="สั่งหยุดการทำงานของ Ollama Service", use_container_width=True):
                    res_stop = engine.stop_ollama_service()
                    if res_stop["success"]:
                        st.toast("🔴 ปิด Ollama Server แล้ว")
                    st.rerun()
                    
            # กรองโมเดลสำหรับ Embedding ออกจากรายการโมเดลเขียนโค้ด
            installed_raw = [m for m in status_info["models"] if "embed" not in m.lower()]
            
            # รวบรวมโมเดลที่ติดตั้งแล้ว และโมเดลยอดนิยม
            model_options = []
            for m in installed_raw:
                clean_m = m.replace(":latest", "")
                model_options.append(f"🟢 {clean_m} (ติดตั้งแล้ว)")
                
            for m in POPULAR_MODELS:
                clean_pop = m.replace(":latest", "")
                if not any(clean_pop in opt for opt in model_options):
                    model_options.append(f"📥 {clean_pop}")
                    
            model_options.append("✏️ พิมพ์ชื่อโมเดลเอง (Custom)...")
            
            # กำหนด index เริ่มต้น โดยเลือกรุ่นที่ติดตั้งแล้วในเครื่องเป็นอันดับแรก (เช่น qwen2.5-coder:7b)
            default_idx = 0
            for i, opt in enumerate(model_options):
                if "ติดตั้งแล้ว" in opt:
                    default_idx = i
                    break
                    
            choice = st.selectbox("เลือกโมเดล Ollama:", model_options, index=default_idx)
            if choice.startswith("✏️"):
                selected_model = st.text_input("พิมพ์ชื่อโมเดล เช่น qwen2.5-coder:7b:", value="qwen2.5-coder:7b")
            else:
                selected_model = choice.split(" ")[1]
                
            # ตรวจสอบว่าโมเดลนี้ติดตั้งในเครื่องแล้วหรือไม่
            is_installed = any(selected_model in m for m in status_info["models"])
            if not is_installed:
                st.info(f"💡 โมเดล `{selected_model}` ยังไม่ได้ติดตั้งใน Ollama\n\nคำสั่งดาวน์โหลดผ่าน Terminal:\n```bash\nollama run {selected_model}\n```\n*(หากเรียกใช้ระบบจะสลับโหมดจำลองให้อัตโนมัติ)*")
        else:
            st.markdown(f'<span class="status-badge status-offline">🔴 Ollama ยังไม่ตอบรับ</span>', unsafe_allow_html=True)
            st.caption("ระบบกำลังพยายามเริ่มการทำงานในเบื้องหลัง หรือคลิกปุ่มด้านล่างเพื่อเริ่มใหม่:")
            
            # ปุ่ม Autorun Ollama Serve เมื่อเกิดกรณีจำเป็น
            if st.button("🚀 สั่งรัน Ollama Server ทันที", type="primary", key="autorun_ollama_sidebar_btn", use_container_width=True):
                with st.spinner("กำลังสั่งรัน 'ollama serve' ในเบื้องหลังและเชื่อมต่อระบบ..."):
                    start_res = engine.start_ollama_service(timeout=8)
                    if start_res["success"]:
                        st.toast("🟢 " + start_res["message"])
                        st.success(start_res["message"])
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(start_res["message"])

            selected_model = st.selectbox("เลือกโมเดลเป้าหมาย (จะสลับโหมดจำลองหากออฟไลน์):", POPULAR_MODELS, index=0)

    elif provider == "OpenAI / Custom API":
        api_key = st.text_input("API Key:", type="password", placeholder="sk-...")
        api_base = st.text_input("Base URL:", value="https://api.openai.com/v1")
        selected_model = st.text_input("Model Name:", value="gpt-4o-mini")

        
    st.divider()
    st.markdown("### 🔀 Hybrid RAG (สืบค้นชุดข้อมูล)")
    use_dataset_rag = st.checkbox(
        "เปิดใช้ Hybrid RAG จาก GitHub Dataset",
        value=True,
        help="ผสาน Sparse BM25 (คำค้นตรงตัว) + Dense Semantic Vector Space (ความหมายคล้ายคลึง)"
    )
    
    rag_strategy = "weighted"
    alpha_weight = 0.5
    top_k_rag = 2
    
    if use_dataset_rag:
        strategy_display = st.selectbox(
            "กลยุทธ์ Hybrid Retrieval:",
            [
                "🔀 Hybrid Fusion (BM25 + Semantic)",
                "🏆 Reciprocal Rank Fusion (RRF)",
                "🔤 Sparse Only (BM25 Lexical)",
                "🧠 Dense Only (Semantic Vector)"
            ],
            index=0,
            help="เลือกวิธีคำนวณและรวมผลการค้นหา"
        )
        if "Hybrid Fusion" in strategy_display:
            rag_strategy = "weighted"
            alpha_weight = st.slider(
                "ค่าน้ำหนัก Semantic (Alpha α):",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="0.0 = BM25 ล้วน (Keyword), 1.0 = Semantic ล้วน (Cosine Sim), 0.5 = สมดุลเท่ากัน"
            )
        elif "RRF" in strategy_display:
            rag_strategy = "rrf"
        elif "Sparse" in strategy_display:
            rag_strategy = "sparse_only"
        else:
            rag_strategy = "dense_only"
            
        top_k_rag = st.slider("จำนวน Context อ้างอิง (Top-K):", min_value=1, max_value=4, value=2)
        st.caption(f"📦 ชุดข้อมูลพร้อมใช้: **{len(dataset_mgr.data)} รายการ** (TheAlgorithms/Python)")
    
    st.divider()
    st.markdown("### 🎛️ พารามิเตอร์การสร้างโค้ด")
    temperature = st.slider("Temperature (ความสุ่ม):", min_value=0.0, max_value=1.0, value=DEFAULT_TEMPERATURE, step=0.05,
                            help="ค่าต่ำ (0.0-0.2) เหมาะสำหรับงานโค้ดเพื่อให้ผลลัพธ์แน่นอนและถูกต้อง แม่นยำ")
    max_tokens = st.slider("Max Tokens:", min_value=256, max_value=4096, value=DEFAULT_MAX_TOKENS, step=256)
    
    st.divider()
    st.markdown("### 💡 ตัวอย่าง Prompt ด่วน")
    example_prompts = [
        "สร้าง Binary Search พร้อม Type Hints และตัวอย่าง",
        "เขียน Quick Sort Algorithm พร้อมคำนวณ Time Complexity",
        "สร้างฟังก์ชัน Fibonacci แบบ Dynamic Programming",
        "เขียน REST API ด้วย FastAPI สำหรับระบบสินค้า",
        "สร้าง Binary Search Tree พร้อมเมธอด insert และ search",
        "ใครเรียกใช้ฟังก์ชัน run_python_code บ้าง และส่งค่าอะไรเข้าไป",
        "ฟังก์ชัน retrieve_relevant_resources มีการเรียกใช้ฟังก์ชันอะไรต่อบ้าง",
        "อธิบายโครงสร้างสถาปัตยกรรมและความสัมพันธ์ของโมดูลใน Codebase นี้"
    ]
    for ex in example_prompts:
        if st.button(f"📌 {ex}", use_container_width=True, key=f"btn_{ex}"):
            st.session_state.prefilled_prompt = ex
            st.rerun()

    st.divider()
    st.markdown("### 🕸️ Code Knowledge Graph (Kùzu DB)")
    graph_stats = st.session_state.graph_engine.get_stats()
    st.caption(f"📊 โหนดในฐานข้อมูล: **{graph_stats['functions']} Functions**, **{graph_stats['modules']} Modules**, **{graph_stats['calls']} Calls**")
    if st.button("🔄 สแกน Codebase สร้าง Graph DB", key="reindex_graph_sidebar_btn", use_container_width=True):
        with st.spinner("กำลังวิเคราะห์ AST และสร้าง Kùzu Knowledge Graph..."):
            res = st.session_state.graph_engine.index_codebase(".")
            st.toast(f"✅ สแกนสำเร็จ! {res['functions']} ฟังก์ชัน, {res['calls']} calls")
            st.rerun()

    st.divider()
    st.markdown("### 📊 ประสิทธิภาพ & การส่งออก")
    q_count = st.session_state.session_stats["total_queries"]
    avg_latency = (st.session_state.session_stats["total_time"] / q_count) if q_count > 0 else 0.0
    c_stat1, c_stat2 = st.columns(2)
    c_stat1.metric("คำขอทั้งหมด", q_count)
    c_stat2.metric("เฉลี่ย/คำตอบ", f"{avg_latency:.2f}s")
    st.caption(f"🔗 Graph: **{st.session_state.session_stats['graph_routed']}** | 🔍 Vector: **{st.session_state.session_stats['vector_routed']}**")

    # ปุ่ม Export Chat History
    chat_export_md = generate_chat_markdown(st.session_state.chat_history)
    st.download_button(
        label="📥 ส่งออกประวัติแชท (.md)",
        data=chat_export_md,
        file_name=f"coding_assistant_chat_{time.strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True,
        help="บันทึกประวัติการแชท โค้ดที่สร้าง และผลลัพธ์เป็นไฟล์ Markdown"
    )

    if st.button("🗑️ ล้างประวัติการสนทนาทั้งหมด", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.session_stats = {"total_queries": 0, "total_time": 0.0, "graph_routed": 0, "vector_routed": 0}
        st.rerun()

# ─── ส่วนหัวหลักของหน้าเว็บ ──────────────────────────────────────────────────
st.markdown('<div class="main-title">💻 AI Coding Assistant & Code Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">ระบบ LLM ผู้ช่วยเขียนโค้ดอัจฉริยะ พร้อม Code Knowledge Graph (Kùzu), Hybrid RAG และระบบรันโค้ดสด</div>', unsafe_allow_html=True)

# ─── Navigation Tabs ─────────────────────────────────────────────────────────
tab_assistant, tab_sandbox, tab_graph, tab_dataset, tab_system_docs = st.tabs([
    "💬 ผู้ช่วยเขียนโค้ด (Coding Assistant & Router)",
    "⚡ ทดสอบรันโค้ด (Sandbox)",
    "🕸️ โครงสร้างโค้ด & Graph RAG (Code Knowledge Graph)",
    "🌐 คลัง Resource Dataset (Internet)",
    "📚 อธิบายการทำงานแต่ละระบบ (System Explanation)"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: CODING ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
with tab_assistant:
    col_mode, col_lang = st.columns([2, 1])
    with col_mode:
        selected_mode_key = st.selectbox(
            "🎯 เลือกโหมดการทำงาน:",
            options=list(ASSISTANT_MODES.keys()),
            format_func=lambda k: ASSISTANT_MODES[k],
            index=0
        )
    with col_lang:
        selected_language = st.selectbox("🌐 ภาษาโปรแกรมเป้าหมาย:", SUPPORTED_LANGUAGES, index=0)

    # ฟิลด์เพิ่มเติมสำหรับโหมดที่ต้องส่งโค้ดต้นทางมาวิเคราะห์
    code_context = ""
    error_context = ""
    if selected_mode_key in ["explain", "debug", "test", "refactor"]:
        with st.expander("📝 แนบ Source Code ที่ต้องการวิเคราะห์ / แก้ไข", expanded=True):
            code_context = st.text_area(
                "วางโค้ดเดิมที่นี่:",
                height=150,
                placeholder="วางซอร์สโค้ดของคุณตรงนี้...",
                key="code_context_input"
            )
            if selected_mode_key == "debug":
                error_context = st.text_area(
                    "ระบุข้อความ Error / Traceback (ถ้ามี):",
                    height=80,
                    placeholder="e.g. IndexError: list index out of range at line 14",
                    key="error_context_input"
                )

    # แสดงสถานะ Local LLM ให้ผู้ใช้มั่นใจว่าเชื่อมต่ออยู่ตลอดเวลา
    if provider == "Ollama (Local LLM)":
        cur_ollama_status = engine.check_ollama_status()
        if cur_ollama_status["online"]:
            st.caption(f"⚡ **กำลังใช้งาน Local Model:** `{selected_model}` | สถานะ: `🟢 เชื่อมต่อ Ollama สำเร็จ (Auto-run พร้อมใช้งาน)`")
        else:
            with st.container():
                st.warning("⚠️ **ตรวจพบ Ollama Server ปิดอยู่** — ระบบกำลังสั่งเปิดใช้งานในเบื้องหลัง หรือคลิกปุ่มด้านล่าง:")
                col_t1_a, col_t1_b = st.columns([2, 1])
                with col_t1_a:
                    st.caption("💡 เริ่มรัน `ollama serve` ในเบื้องหลังทันทีโดยไม่ต้องเปิด Terminal")
                with col_t1_b:
                    if st.button("🚀 สตาร์ท Ollama ทันที", key="tab1_autorun_btn", type="primary", use_container_width=True):
                        with st.spinner("กำลังสั่งรัน 'ollama serve' ในเบื้องหลังและเชื่อมต่อระบบ..."):
                            t1_res = engine.start_ollama_service(timeout=8)
                            if t1_res["success"]:
                                st.toast("🟢 " + t1_res["message"])
                                st.success(t1_res["message"])
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(t1_res["message"])

    st.markdown("#### 💬 ประวัติการสนทนา & ผลลัพธ์")

    
    # แสดงประวัติการสนทนา
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            
            # แสดง Source References / Context ที่ใช้ (Session 13 feature)
            if msg.get("sources"):
                with st.expander("📚 ข้อมูลอ้างอิงและบริบทที่ใช้ประมวลผล (Retrieved Sources)", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(src)
                        
            # แสดงข้อมูลความเร็วและเส้นทาง Router
            if msg.get("latency") is not None:
                route_tag = msg.get("route_badge", "🔍 Vector Search")
                st.caption(f"⚡ เวลาตอบสนอง: `{msg['latency']:.2f} วินาที` | เส้นทาง: {route_tag}")

            if msg["role"] == "assistant":
                candidate_code = msg.get("code_snippet") or extract_python_code(msg.get("content", ""))
                if candidate_code and candidate_code.strip():
                    msg["code_snippet"] = candidate_code.strip()
                    code_lines = len(candidate_code.strip().splitlines())
                    
                    c_send1, c_send2 = st.columns([1.6, 2])
                    with c_send1:
                        btn_label = f"🚀 ส่งโค้ดนี้ไปยัง Sandbox ({code_lines} บรรทัด)"
                        if st.button(btn_label, key=f"send_sandbox_{msg.get('id', time.time())}", type="primary", use_container_width=True):
                            set_sandbox_code(candidate_code)
                            st.toast("✅ ส่งโค้ดเข้า Sandbox สำเร็จ! สามารถคลิกแท็บ '⚡ ทดสอบรันโค้ด (Sandbox)' ด้านบนเพื่อทดสอบรันได้เลย")
                            time.sleep(0.3)
                            st.rerun()

                # ตรวจจับและเรนเดอร์ Mermaid Diagram สดทันทีถ้ามีในข้อความ
                mermaid_diag = extract_mermaid_code(msg.get("content", ""))
                if mermaid_diag:
                    with st.expander("📊 ดูแผนภาพกราฟสด (Live Interactive Diagram)", expanded=True):
                        render_mermaid(mermaid_diag, height=360)

    # รับ Input Prompt จากผู้ใช้
    user_prompt_input = st.chat_input(
        placeholder=f"พิมพ์ความต้องการของคุณสำหรับ {ASSISTANT_MODES[selected_mode_key]} เช่น 'ใครเรียกใช้ run_python_code' หรือ 'เขียน Binary Search'..."
    )

    # จัดการกรณีที่กดปุ่มตัวอย่างด่วนจาก Sidebar
    active_prompt = None
    if user_prompt_input:
        active_prompt = user_prompt_input
    elif st.session_state.prefilled_prompt:
        active_prompt = st.session_state.prefilled_prompt
        st.session_state.prefilled_prompt = ""

    if active_prompt:
        st.session_state.session_stats["total_queries"] += 1
        
        # 1. วิเคราะห์และคัดแยกเส้นทางด้วย QuestionRouter (Session 12 Router Pattern)
        route_decision = st.session_state.question_router.classify_with_reason(active_prompt)
        current_route = route_decision["route"]
        route_badge = route_decision["mode_badge"]
        
        retrieved_refs = []
        graph_context_text = ""
        context_sources = []

        if current_route == "graph":
            st.session_state.session_stats["graph_routed"] += 1
            # ดึงข้อมูลเชิงโครงสร้างจาก Code Knowledge Graph (Kùzu DB)
            all_known_funcs = st.session_state.graph_engine.get_all_function_names()
            matched_funcs = [fn for fn in all_known_funcs if fn.lower() in active_prompt.lower()]
            
            if matched_funcs:
                target_fn = matched_funcs[0]
                callers = st.session_state.graph_engine.get_callers(target_fn)
                callees = st.session_state.graph_engine.get_callees(target_fn)
                caller_names = [c["caller_name"] for c in callers]
                callee_names = [c["callee_name"] for c in callees]
                
                graph_context_text = (
                    f"\n\n[Code Knowledge Graph Context for `{target_fn}`]:\n"
                    f"- Callers (ฟังก์ชันที่เรียกใช้ {target_fn}): {', '.join(caller_names) if caller_names else 'ไม่มีใครเรียก'}\n"
                    f"- Callees (ฟังก์ชันที่ {target_fn} เรียกใช้ต่อ): {', '.join(callee_names) if callee_names else 'ไม่มีการเรียกฟังก์ชันอื่น'}\n"
                )
                context_sources.append(f"🕸️ **Code Graph (Kùzu):** ฟังก์ชัน `{target_fn}` (Callers: {len(caller_names)}, Callees: {len(callee_names)})")
            else:
                stats = st.session_state.graph_engine.get_stats()
                graph_context_text = (
                    f"\n\n[Code Knowledge Graph Overview]: "
                    f"Codebase นี้มี {stats['functions']} Functions, {stats['modules']} Modules, และ {stats['calls']} Call Relations ใน Kùzu DB\n"
                )
                context_sources.append("🕸️ **Code Graph (Kùzu):** Architectural structural context")
        else:
            st.session_state.session_stats["vector_routed"] += 1
            # 2. ค้นหา Resource Dataset จาก Internet ด้วย Hybrid RAG (Session 12)
            if use_dataset_rag:
                retrieved_refs = dataset_mgr.retrieve_relevant_resources(
                    query=active_prompt,
                    top_k=top_k_rag,
                    strategy=rag_strategy,
                    alpha=alpha_weight
                )
                for r in retrieved_refs:
                    m_type = r.get('match_type', 'Hybrid')
                    h_score = r.get('hybrid_score', 0.0)
                    context_sources.append(f"🔗 **[{r['title']}]({r['source_url']})** `[{m_type}]` (Score: **{h_score}**)")

        # 3. จัดแสดงข้อความผู้ใช้พร้อม Badge แสดงผลลัพธ์ของ Router
        user_display = f"**[{ASSISTANT_MODES[selected_mode_key]}] ภาษา: {selected_language}** — `{route_badge}`\n\n{active_prompt}"
        if context_sources:
            user_display += "\n\n" + "\n".join([f"- {s}" for s in context_sources])

        st.session_state.chat_history.append({"role": "user", "content": user_display, "id": time.time()})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_display)

        # 4. ประกอบ Prompt รวม Context โครงสร้างกราฟ (ถ้ามี)
        final_code_context = (code_context + graph_context_text).strip()
        prompt_bundle = build_prompt_bundle(
            mode=selected_mode_key,
            user_prompt=active_prompt,
            language=selected_language,
            code_context=final_code_context,
            error_message=error_context,
            dataset_references=retrieved_refs if use_dataset_rag else None
        )

        # 5. จับเวลาและ Stream คำตอบจาก LLM Engine
        start_eval_time = time.time()
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            full_response = ""

            status_check = engine.check_ollama_status()
            
            if provider == "Ollama (Local LLM)" and status_check["online"]:
                stream_gen = engine.generate_stream_ollama(
                    model=selected_model,
                    system_prompt=prompt_bundle["system"],
                    user_prompt=prompt_bundle["user"],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            elif provider == "OpenAI / Custom API" and api_key:
                stream_gen = engine.generate_stream_openai(
                    api_key=api_key,
                    base_url=api_base,
                    model=selected_model,
                    system_prompt=prompt_bundle["system"],
                    user_prompt=prompt_bundle["user"],
                    temperature=temperature
                )
            else:
                stream_gen = engine.generate_stream_simulation(
                    mode=selected_mode_key,
                    user_prompt=active_prompt + (f" (Graph Context: {graph_context_text})" if graph_context_text else ""),
                    language=selected_language,
                    code_context=final_code_context,
                    dataset_references=retrieved_refs if use_dataset_rag else None
                )

            for chunk in stream_gen:
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)

        latency = time.time() - start_eval_time
        st.session_state.session_stats["total_time"] += latency

        # 6. บันทึกและดึงโค้ดออกมาเก็บไว้ในประวัติ (สกัดโค้ด Python หากตรวจพบ)
        extracted_code = extract_python_code(full_response)
        msg_id = time.time()
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": full_response,
            "code_snippet": extracted_code,
            "sources": context_sources,
            "latency": latency,
            "route_badge": route_badge,
            "id": msg_id
        })
        
        # อัปเดต sandbox อัตโนมัติหากได้โค้ดใหม่
        if extracted_code:
            set_sandbox_code(extracted_code)
            st.toast("⚡ โค้ดถูกส่งเข้า Sandbox พร้อมทดสอบรันแล้ว!")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: LIVE CODE SANDBOX
# ══════════════════════════════════════════════════════════════════════════════
with tab_sandbox:
    st.markdown("### ⚡ ทดสอบและรันโค้ด Python ใน Sandbox")
    st.caption("ระบบจะสร้างสภาพแวดล้อมจำลอง (Subprocess) เพื่อรันโค้ดและดักจับผลลัพธ์ stdout, stderr พร้อมจับเวลาการทำงาน")
    
    col_code, col_output = st.columns([1.2, 1])
    
    with col_code:
        st.markdown("**✏️ Python Code Editor:**")
        editor_code = st.text_area(
            "Code Editor",
            value=st.session_state.sandbox_code,
            height=380,
            key="sandbox_editor",
            label_visibility="collapsed"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1.2, 1.4, 1])
        with col_btn1:
            run_btn = st.button("▶️ รันโค้ดทันที (Execute)", type="primary", use_container_width=True)
        with col_btn2:
            latest_chat_code = None
            for prev_m in reversed(st.session_state.chat_history):
                if prev_m.get("role") == "assistant":
                    c_found = prev_m.get("code_snippet") or extract_python_code(prev_m.get("content", ""))
                    if c_found and c_found.strip():
                        latest_chat_code = c_found.strip()
                        break
            
            pull_btn = st.button(
                "📥 ดึงโค้ดล่าสุดจากแชท",
                use_container_width=True,
                disabled=(latest_chat_code is None),
                help="ดึงโค้ด Python ล่าสุดจากประวัติการสนทนามาใส่ใน Editor นี้ทันที"
            )
            if pull_btn and latest_chat_code:
                set_sandbox_code(latest_chat_code)
                st.toast("✅ ดึงโค้ดล่าสุดจากแชทเข้า Sandbox เรียบร้อยแล้ว!")
                st.rerun()
        with col_btn3:
            if st.button("🧹 ล้างโค้ด", use_container_width=True):
                set_sandbox_code("")
                st.toast("🧹 ล้างโค้ดใน Editor เรียบร้อยแล้ว")
                st.rerun()

    with col_output:
        st.markdown("**📊 ผลลัพธ์การทำงาน (Output & Metrics):**")
        if run_btn:
            with st.spinner("กำลังรันโค้ดอย่างปลอดภัย..."):
                exec_result = run_python_code(editor_code, timeout_seconds=10)
                
                # Metrics Row
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Status", "SUCCESS" if exec_result["success"] else "FAILED")
                m_col2.metric("Execution Time", f"{exec_result['execution_time']} s")
                m_col3.metric("Exit Code", exec_result["exit_code"])
                
                if exec_result["stdout"]:
                    st.markdown("**Standard Output (stdout):**")
                    st.code(exec_result["stdout"], language="text")
                else:
                    st.info("ไม่มีการพิมพ์ข้อความออกทาง stdout (print)")
                    
                if exec_result["stderr"]:
                    st.markdown("**Errors / Traceback (stderr):**")
                    st.error(exec_result["stderr"])
        else:
            st.info("👈 กดปุ่ม **'▶️ รันโค้ดทันที'** เพื่อดูผลลัพธ์การรันโปรแกรม")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: CODE KNOWLEDGE GRAPH & GRAPH RAG (KÙZU DB)
# ══════════════════════════════════════════════════════════════════════════════
with tab_graph:
    st.markdown("### 🕸️ โครงสร้างโค้ด & Graph RAG (Code Knowledge Graph)")
    st.markdown(
        "ระบบวิเคราะห์โครงสร้าง Codebase ด้วย **Kùzu Embedded Graph Database** และ **Python AST (Abstract Syntax Tree)** "
        "ช่วยให้ผู้ช่วยเขียนโค้ดเข้าใจลำดับการเรียก (Call Hierarchy), ความสัมพันธ์ของฟังก์ชัน (Callers/Callees) "
        "และการเชื่อมโยงของโมดูล (Cross-module Dependencies) เพื่อตอบคำถามเชิงสถาปัตยกรรมได้อย่างแม่นยำ 100%"
    )

    # สรุปสถานะ Graph Database
    g_stats = st.session_state.graph_engine.get_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 โมดูลที่สแกนพบ", g_stats["modules"])
    c2.metric("⚡ ฟังก์ชันในระบบ", g_stats["functions"])
    c3.metric("🔗 ความสัมพันธ์การเรียก (CALLS)", g_stats["calls"])
    c4.metric("🔄 การ Import (IMPORTS)", g_stats["imports"])

    col_btn_reindex, col_info_reindex = st.columns([1, 3])
    with col_btn_reindex:
        if st.button("🔄 สแกน & อัปเดต Graph DB ทันที", key="btn_reindex_tab_graph", use_container_width=True):
            with st.spinner("กำลังวิเคราะห์ AST และสร้างกราฟใน Kùzu DB..."):
                idx_res = st.session_state.graph_engine.index_codebase(".")
                st.toast(f"✅ สแกนสำเร็จ: {idx_res['functions']} ฟังก์ชัน, {idx_res['calls']} calls")
                st.rerun()
    with col_info_reindex:
        st.caption("💡 ฐานข้อมูล Kùzu เก็บอยู่ในเครื่อง (`graph_db_storage/`) ไม่ต้องเปิดเซิร์ฟเวอร์ภายนอกและประมวลผล Cypher ได้อย่างรวดเร็วระดับ Microsecond")

    st.divider()

    # Sub-tabs สำหรับการใช้งาน Graph RAG
    subtab_func, subtab_mod, subtab_cypher, subtab_router, subtab_visualizer = st.tabs([
        "🔍 วิเคราะห์ความสัมพันธ์ฟังก์ชัน (Functions & Calls)",
        "📦 ภาพรวมโมดูล (Module Inspector)",
        "⚡ ทดสอบคำสั่ง Cypher Query สด",
        "🔀 จำลองการทำงานของ QuestionRouter",
        "🎨 ตัวแสดงผลกราฟ Mermaid สด (Live Visualizer)"
    ])

    # Sub-tab 1: Function Analyzer
    with subtab_func:
        st.markdown("#### 🎯 วิเคราะห์เส้นทางการเรียกฟังก์ชัน (Call Hierarchy & Callers)")
        all_funcs = st.session_state.graph_engine.get_all_function_names()
        
        if not all_funcs:
            st.warning("⚠️ ยังไม่พบฟังก์ชันในฐานข้อมูล กรุณากดปุ่ม '🔄 สแกน & อัปเดต Graph DB ทันที' ด้านบน")
        else:
            col_f_sel, col_f_action = st.columns([2, 1])
            with col_f_sel:
                default_fn_idx = all_funcs.index("retrieve_relevant_resources") if "retrieve_relevant_resources" in all_funcs else 0
                selected_func = st.selectbox(
                    "เลือกฟังก์ชันเป้าหมาย:",
                    options=all_funcs,
                    index=default_fn_idx
                )
            with col_f_action:
                graph_action = st.selectbox(
                    "มุมมองการวิเคราะห์:",
                    ["แผนภาพ Callers & Callees (Mermaid)", "ใครเรียกใช้ฟังก์ชันนี้? (Callers)", "ฟังก์ชันนี้เรียกใครบ้าง? (Callees)", "สืบสายลำดับการเรียกต่อเนื่อง (Hierarchy)"]
                )

            # แสดงผล
            if graph_action == "แผนภาพ Callers & Callees (Mermaid)":
                st.markdown("**📊 แผนภาพความสัมพันธ์สด (Live Interactive Diagram):**")
                mermaid_code = st.session_state.graph_engine.generate_function_mermaid(selected_func)
                render_mermaid(mermaid_code, height=360)
                with st.expander("📝 ดูโค้ด Mermaid Syntax"):
                    st.code(mermaid_code, language="mermaid")
                
            elif graph_action == "ใครเรียกใช้ฟังก์ชันนี้? (Callers)":
                callers = st.session_state.graph_engine.get_callers(selected_func)
                st.markdown(f"**⚡ ฟังก์ชันที่เรียกใช้ `{selected_func}` (พบ {len(callers)} แห่ง):**")
                if callers:
                    st.dataframe(callers, use_container_width=True)
                else:
                    st.info(f"ไม่มีฟังก์ชันอื่นเรียกใช้ `{selected_func}` (อาจเป็น Root Entrypoint หรือ Event Handler)")
                    
            elif graph_action == "ฟังก์ชันนี้เรียกใครบ้าง? (Callees)":
                callees = st.session_state.graph_engine.get_callees(selected_func)
                st.markdown(f"**📦 ฟังก์ชันที่ `{selected_func}` เรียกใช้ต่อ (พบ {len(callees)} แห่ง):**")
                if callees:
                    st.dataframe(callees, use_container_width=True)
                else:
                    st.info(f"`{selected_func}` ไม่ได้เรียกใช้ฟังก์ชันอื่น (Leaf Function)")
                    
            else:
                st.markdown(f"**🌳 ลำดับสายการเรียกต่อเนื่อง (Call Hierarchy Tree) ของ `{selected_func}`:**")
                tree = st.session_state.graph_engine.trace_call_hierarchy(selected_func, max_depth=3)
                st.json(tree)

            # ปุ่มให้ AI อธิบายสถาปัตยกรรม
            st.write("")
            if st.button(f"🤖 ให้ AI ช่วยวิเคราะห์สถาปัตยกรรมของ `{selected_func}`", key=f"ai_explain_graph_{selected_func}"):
                with st.spinner("AI กำลังวิเคราะห์ Graph Context และโครงสร้างโค้ด..."):
                    callers_list = [c["caller_name"] for c in st.session_state.graph_engine.get_callers(selected_func)]
                    callees_list = [c["callee_name"] for c in st.session_state.graph_engine.get_callees(selected_func)]
                    ai_prompt = (
                        f"วิเคราะห์บทบาทและสถาปัตยกรรมของฟังก์ชัน '{selected_func}' ในฐานข้อมูลโค้ด:\n"
                        f"- ผู้เรียกใช้ (Callers): {callers_list}\n"
                        f"- ฟังก์ชันที่ถูกเรียก (Callees): {callees_list}\n"
                        f"กรุณาอธิบาย: 1. ความรับผิดชอบหลัก (Single Responsibility) 2. ผลกระทบหากมีการแก้ไข (Impact Analysis) 3. คำแนะนำในการเขียน Unit Test"
                    )
                    ai_res = ""
                    status_check = engine.check_ollama_status()
                    if status_check["online"]:
                        gen = engine.generate_stream_ollama(
                            model=selected_model,
                            system_prompt="คุณคือ Software Architect ผู้เชี่ยวชาญด้าน Graph RAG และ Code Analysis",
                            user_prompt=ai_prompt,
                            temperature=0.2
                        )
                    else:
                        gen = [f"### 🏛️ การวิเคราะห์สถาปัตยกรรมฟังก์ชัน `{selected_func}`\n\n"
                               f"- **บทบาทในระบบ:** ทำหน้าที่เป็นตัวเชื่อมโยงในระบบ โดยถูกเรียกโดย {callers_list or 'Entrypoint'}\n"
                               f"- **การพึ่งพา:** มีการกระจายงานต่อไปยัง {callees_list or 'ไม่มี'}\n"
                               f"- **ผลกระทบจากการแก้ไข:** ควรระมัดระวังเรื่อง Backward Compatibility ของพารามิเตอร์ส่งออก\n"
                               f"- **คำแนะนำ Test:** ควรทำ Mocking ฟังก์ชัน Callee เพื่อทดสอบแบบ Isolated Unit Test"]
                    
                    st.markdown("#### 🤖 ผลการวิเคราะห์จาก AI Software Architect:")
                    for chunk in gen:
                        ai_res += chunk
                    st.markdown(ai_res)

    # Sub-tab 2: Module Inspector
    with subtab_mod:
        st.markdown("#### 📦 ตรวจสอบโครงสร้างและการเชื่อมโยงของโมดูล (Module Inspector)")
        all_mods = st.session_state.graph_engine.get_all_module_names()
        if not all_mods:
            st.info("ไม่พบโมดูลในฐานข้อมูล กรุณาสแกน Codebase ก่อน")
        else:
            sel_mod = st.selectbox("เลือกโมดูลที่ต้องการตรวจสอบ:", options=all_mods)
            mod_summary = st.session_state.graph_engine.get_module_summary(sel_mod)
            
            c_m1, c_m2 = st.columns(2)
            with c_m1:
                st.markdown(f"**⚡ ฟังก์ชันที่ประกาศในโมดูล `{sel_mod}` ({len(mod_summary['functions'])} ฟังก์ชัน):**")
                if mod_summary["functions"]:
                    st.dataframe(mod_summary["functions"], use_container_width=True)
                else:
                    st.caption("ไม่มีการประกาศฟังก์ชันโดยตรงในโมดูลนี้")
            with c_m2:
                st.markdown(f"**🔄 โมดูลที่ถูก Import ใน `{sel_mod}` ({len(mod_summary['imports'])} โมดูล):**")
                if mod_summary["imports"]:
                    for imp in mod_summary["imports"]:
                        st.markdown(f"- 📦 `{imp}`")
                else:
                    st.caption("ไม่มีการ Import โมดูลภายในโปรเจกต์นี้")

    # Sub-tab 3: Cypher Query Console
    with subtab_cypher:
        st.markdown("#### ⚡ คอนโซลรันคำสั่ง Cypher Query สด (Kùzu Property Graph)")
        st.caption("ทดสอบเขียนภาษา Cypher Query เพื่อสืบค้น Node & Relationship ภายในฐานข้อมูลได้โดยตรง")
        
        presets = {
            "แสดงการเรียกฟังก์ชัน (CALLS) 15 รายการแรก": "MATCH (caller:Function)-[:CALLS]->(callee:Function) RETURN caller.name, callee.name LIMIT 15",
            "แสดงฟังก์ชันทั้งหมดเรียงตามชื่อ": "MATCH (f:Function) WHERE f.file <> '' RETURN f.name, f.file, f.line ORDER BY f.name LIMIT 20",
            "แสดงการ Import ข้ามโมดูล (IMPORTS)": "MATCH (m1:Module)-[:IMPORTS]->(m2:Module) RETURN m1.name, m2.name",
            "แสดงคลาสและการสังกัดโมดูล": "MATCH (c:Class)-[:CLASS_DEFINED_IN]->(m:Module) RETURN c.name, m.name"
        }
        sel_preset = st.selectbox("เลือกคำสั่งตัวอย่าง:", list(presets.keys()))
        cypher_text = st.text_area("คำสั่ง Cypher Query:", value=presets[sel_preset], height=90)
        
        if st.button("▶️ รัน Cypher Query", key="run_cypher_btn", type="primary"):
            res_df, msg = st.session_state.graph_engine.execute_cypher(cypher_text)
            if msg == "success" and res_df is not None:
                st.success(f"✅ สำเร็จ! พบข้อมูลทั้งหมด {len(res_df)} รายการ")
                st.dataframe(res_df, use_container_width=True)
            else:
                st.error(f"❌ เกิดข้อผิดพลาดในการรันคำสั่ง Cypher: {msg}")

    # Sub-tab 4: Router Playground
    with subtab_router:
        st.markdown("#### 🔀 ห้องทดลองระบบ QuestionRouter (Session 12 Router Pattern)")
        st.markdown(
            "ทดสอบป้อนข้อความคำถามเพื่อดูว่าระบบจะตัดสินใจส่งต่อไปยัง **Graph Database (เชิงโครงสร้าง)** "
            "หรือ **Vector/Hybrid RAG (เชิงความหมายและคอนเซปต์)** พร้อมแสดงเหตุผลและคีย์เวิร์ดที่ตรวจพบ"
        )
        sample_q = st.text_input(
            "พิมพ์คำถามทดสอบ Router:",
            value="ใครเรียกใช้ run_python_code ในระบบบ้าง",
            placeholder="เช่น ใครเรียกใช้ ..., ลำดับการเรียก ..., เขียนโค้ด binary search, อธิบายการทำงาน ..."
        )
        if sample_q:
            r_result = st.session_state.question_router.classify_with_reason(sample_q)
            r_col1, r_col2 = st.columns([1, 2])
            with r_col1:
                if r_result["route"] == "graph":
                    st.success(f"### 🔗 {r_result['mode_badge']}")
                else:
                    st.info(f"### 🔍 {r_result['mode_badge']}")
            with r_col2:
                st.markdown(f"**เหตุผลการตัดสินใจ:** {r_result['reason']}")
                if r_result["matched_keywords"]:
                    st.markdown("**คีย์เวิร์ดที่ตรวจพบ:** " + " ".join([f"`{k}`" for k in r_result["matched_keywords"]]))
                st.caption(f"เส้นทาง: `{r_result['route'].upper()}` -> ระบบจะดึงข้อมูลจาก `{ 'Kùzu Graph DB' if r_result['route'] == 'graph' else 'Hybrid BM25 + Dense Vector' }`")

    # Sub-tab 5: Live Mermaid Visualizer
    with subtab_visualizer:
        st.markdown("#### 🎨 ตัวแสดงผลกราฟ Mermaid สด (Live Interactive Visualizer)")
        st.markdown("พิมพ์หรือวางโค้ด Mermaid ที่ได้จาก AI ในกล่องข้อความด้านล่าง เพื่อดูกราฟฟิก SVG แสดงโหนดและเส้นเชื่อมโยงแบบสดๆ ได้ทันที")

        default_mermaid_demo = (
            "graph LR\n"
            "  UI[\"🖥️ Streamlit Frontend\"] --> Router{\"🔀 QuestionRouter\"}\n"
            "  Router -->|'graph'| Kuzu[\"🕸️ Kùzu Graph DB (AST)\"]\n"
            "  Router -->|'vector'| RAG[\"🧠 Hybrid RAG (BM25+Dense)\"]\n"
            "  Kuzu --> Context[\"📄 Augmented Context\"]\n"
            "  RAG --> Context\n"
            "  Context --> LLM[\"🤖 LLM Engine (Ollama)\"]\n"
            "  LLM --> Sandbox[\"⚡ Subprocess Sandbox\"]\n"
            "  style UI fill:#3b82f6,stroke:#1d4ed8,color:#fff\n"
            "  style Kuzu fill:#8b5cf6,stroke:#6d28d9,color:#fff\n"
            "  style RAG fill:#10b981,stroke:#047857,color:#fff\n"
            "  style LLM fill:#f59e0b,stroke:#b45309,color:#fff\n"
        )

        c_v1, c_v2 = st.columns([1, 1.3])
        with c_v1:
            st.markdown("**📝 โค้ด Mermaid (แก้ไขได้อิสระ):**")
            mermaid_input = st.text_area(
                "Mermaid Code Editor",
                value=default_mermaid_demo,
                height=340,
                key="custom_mermaid_editor",
                label_visibility="collapsed"
            )
            col_mv1, col_mv2 = st.columns(2)
            with col_mv1:
                if st.button("🔄 อัปเดตกราฟ (Render)", use_container_width=True, type="primary"):
                    st.rerun()
            with col_mv2:
                if st.button("📋 ใช้ตัวอย่าง Architecture", use_container_width=True):
                    st.session_state["custom_mermaid_editor"] = default_mermaid_demo
                    st.rerun()

        with c_v2:
            st.markdown("**📊 แผนภาพกราฟฟิกสด (Interactive Rendered SVG):**")
            if mermaid_input and mermaid_input.strip():
                render_mermaid(mermaid_input, height=360)
            else:
                st.info("กรุณาวางโค้ด Mermaid ในช่องซ้ายมือเพื่อเรนเดอร์กราฟ")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: INTERNET RESOURCE DATASET & HYBRID RAG PLAYGROUND
# ══════════════════════════════════════════════════════════════════════════════
with tab_dataset:

    st.markdown("### 🌐 คลัง Resource Dataset & ห้องทดลองสืบค้น Hybrid RAG")
    st.markdown(
        "ชุดข้อมูล Source Code จริงที่ดึงมาจากคลัง Open Source ระดับโลก ([TheAlgorithms/Python](https://github.com/TheAlgorithms/Python) และ [tiangolo/fastapi](https://github.com/tiangolo/fastapi)) "
        "พร้อมทดสอบการทำงานของ **Hybrid Search Engine (Sparse BM25 + Dense Semantic Vector Space)** แบบ Interactive สด"
    )
    
    col_ds1, col_ds2 = st.columns([3, 1])
    with col_ds1:
        search_kw = st.text_input(
            "🔍 พิมพ์คำค้นหาเพื่อทดสอบระบบ Hybrid Search:",
            placeholder="เช่น binary search, ทวิภาค, ค้นหา, จัดเรียง, merge sort, knapsack, graph, lru cache, fastapi api...",
            key="tab3_search_input"
        )
    with col_ds2:
        st.write("")
        st.write("")
        if st.button("🔄 ซิงค์ Dataset ล่าสุดจาก GitHub", key="sync_dataset_btn", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            def update_progress(cur, total, name):
                progress_bar.progress(cur / total)
                status_text.caption(f"กำลังดาวน์โหลด ({cur}/{total}): {name}")
            count = dataset_mgr.fetch_dataset_from_internet(progress_callback=update_progress)
            st.success(f"✅ ซิงค์สำเร็จ! อัปเดตและทำ Indexing ข้อมูล {count} รายการ")
            st.rerun()

    # ควบคุมพารามิเตอร์การทดลองค้นหาในแท็บนี้
    col_st1, col_st2 = st.columns([1.5, 1])
    with col_st1:
        tab3_strat_label = st.selectbox(
            "🔬 เลือกกลยุทธ์การค้นหา (Strategy):",
            [
                "🔀 Hybrid Fusion (BM25 + Semantic)",
                "🏆 Reciprocal Rank Fusion (RRF)",
                "🔤 Sparse Only (BM25 Lexical)",
                "🧠 Dense Only (Semantic Vector)"
            ],
            key="tab3_strategy_sel"
        )
    
    tab3_strat_code = "weighted"
    tab3_alpha = 0.5
    if "Hybrid Fusion" in tab3_strat_label:
        tab3_strat_code = "weighted"
        with col_st2:
            tab3_alpha = st.slider("ค่าน้ำหนัก Semantic (Alpha α):", 0.0, 1.0, 0.5, step=0.05, key="tab3_alpha_slider",
                                  help="0.0 = BM25 ล้วน | 1.0 = Semantic ล้วน | 0.5 = สมดุลเท่ากัน")
    elif "RRF" in tab3_strat_label:
        tab3_strat_code = "rrf"
        with col_st2:
            st.caption("🏆 **Reciprocal Rank Fusion (k=60)** ผสานอันดับแบบไม่ขึ้นกับสเกลคะแนน")
    elif "Sparse" in tab3_strat_label:
        tab3_strat_code = "sparse_only"
        with col_st2:
            st.caption("🔤 **Okapi BM25:** จับคู่คำตรงตัว (Exact Term & Token Matching)")
    else:
        tab3_strat_code = "dense_only"
        with col_st2:
            st.caption("🧠 **Dense Vector:** Cosine Similarity บน Subword Vector Space")

    # Filtered / Retrieved Items
    if search_kw.strip():
        items = dataset_mgr.retrieve_relevant_resources(
            query=search_kw.strip(),
            top_k=len(dataset_mgr.data),
            strategy=tab3_strat_code,
            alpha=tab3_alpha
        )
        st.markdown(f"**🎯 ผลการสืบค้นด้วย `{tab3_strat_label}` (พบ {len(items)} รายการ เรียงตามคะแนน):**")
    else:
        items = dataset_mgr.data
        st.markdown(f"**📦 รายการ Source Code ทั้งหมดใน Dataset ({len(items)} รายการ):**")

    for item in items:
        h_score = item.get("hybrid_score")
        bm25_s = item.get("bm25_score")
        dense_s = item.get("dense_score")
        bm25_r = item.get("bm25_rank")
        dense_r = item.get("dense_rank")
        m_type = item.get("match_type", "Reference")
        matched_kw = item.get("matched_terms", [])

        title_header = f"📌 {item['title']} [{item['category']}]"
        if h_score is not None and search_kw.strip():
            title_header += f" — Score: {h_score:.4f}"

        with st.expander(title_header):
            # Metric badges row
            if h_score is not None and search_kw.strip():
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("🏆 Combined Score", f"{h_score:.4f}")
                m2.metric("🔤 BM25 Lexical", f"{bm25_s:.4f}" if bm25_s is not None else "N/A", f"Rank #{bm25_r}" if bm25_r else "")
                m3.metric("🧠 Dense Cosine", f"{dense_s:.4f}" if dense_s is not None else "N/A", f"Rank #{dense_r}" if dense_r else "")
                m4.metric("🎯 Match Type", m_type)

                if matched_kw:
                    st.markdown("**🏷️ คำที่ตรงกับ Query:** " + " ".join([f"`{t}`" for t in matched_kw]))

            st.markdown(f"**🔗 แหล่งที่มาบน GitHub (Source URL):** [{item['source_url']}]({item['source_url']})")
            st.markdown(f"**⏱️ ความซับซ้อน (Complexity):** `{item.get('complexity', 'N/A')}` | **จำนวนบรรทัด:** `{item.get('lines_count', 0)} lines`")
            
            st.code(item["code"], language="python")
            
            if st.button("🚀 ส่งโค้ดนี้ไปยัง Sandbox เพื่อทดสอบรัน", key=f"ds_btn_{item['id']}", type="primary"):
                set_sandbox_code(item["code"])
                st.toast(f"✅ ส่งโค้ด '{item['title']}' เข้า Sandbox เรียบร้อยแล้ว! สามารถสลับไปที่แท็บ '⚡ ทดสอบรันโค้ด (Sandbox)' ได้เลย")
                time.sleep(0.3)
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: SYSTEM EXPLANATION & ARCHITECTURE (อธิบายการทำงานแต่ละระบบขั้นตอน)
# ══════════════════════════════════════════════════════════════════════════════
with tab_system_docs:
    st.markdown("## 📚 สถาปัตยกรรมระบบและการทำงานแบบ Hybrid RAG (System Architecture & Pipeline)")
    st.markdown("""
    ระบบ **Coding Assistant LLM** นี้ใช้สถาปัตยกรรม **Hybrid RAG (Retrieval-Augmented Generation)** ระดับโปรดักชัน 
    โดยผสานการค้นหาแบบ **Sparse (Okapi BM25)** และ **Dense (Semantic Vector Space)** เข้าด้วยกัน 
    เพื่อแก้ปัญหาคลาสสิกของการค้นหาโค้ดโปรแกรม
    """)
    
    st.markdown("""
    ```
    ┌────────────────────────────────────────────────────────────────────────────────────────┐
    │                                  USER INTERFACE (UI)                                   │
    │         รับ Prompt ข้อกำหนดการเขียนโค้ด / ภาษาเป้าหมาย / เลือกโหมดการทำงาน            │
    └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                                │
                                                ▼
    ┌────────────────────────────────────────────────────────────────────────────────────────┐
    │ 0. DYNAMIC QUESTION ROUTER (hybrid_retriever.py -> QuestionRouter)                    │
    │    - วิเคราะห์คำถามเชิงโครงสร้าง (Callers, Callees, Hierarchy, Dependencies)          │
    │    - จำแนกเส้นทางแบบอัตโนมัติ: [GRAPH MODE] vs [VECTOR / HYBRID MODE]                  │
    └─────────────────────┬─────────────────────────────────────────────┬────────────────────┘
                          │                                             │
      (Structural Query)  ▼                                             ▼  (Semantic / Code Gen)
    ┌───────────────────────────────────────┐     ┌──────────────────────────────────────────┐
    │ 1. CODE KNOWLEDGE GRAPH (Kùzu DB)     │     │ 2. HYBRID RAG ENGINE (BM25 + Semantic)   │
    │  - Nodes: Module, Function, Class     │     │  - Branch A: Okapi BM25 Lexical Search   │
    │  - Edges: CALLS, IMPORTS, CONTAINS    │     │  - Branch B: Dense Subword Vector Cosine │
    │  - Call Hierarchy Depth Traversal     │     │  - Fusion: Weighted Combination / RRF    │
    │  - Cypher Query Graph Engine          │     │  - Real-world GitHub Ground-Truth Code   │
    └───────────────────┬───────────────────┘     └─────────────────────┬────────────────────┘
                        │                                               │
                        └───────────────────────┬───────────────────────┘
                                                │ (Combined Structural + Reference Context)
                                                ▼
    ┌────────────────────────────────────────────────────────────────────────────────────────┐
    │ 3. PROMPT ENGINEERING PIPELINE (prompts.py)                                           │
    │    - ผสาน System Persona + Domain Rules + Ground-Truth Reference + Graph Structure     │
    │    - จัดโครงสร้าง Output ให้มี Type Hints, Docstrings, Complexity และ Markdown Syntax  │
    └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                                │
                                                ▼
    ┌────────────────────────────────────────────────────────────────────────────────────────┐
    │ 4. LLM INFERENCE ENGINE (llm_engine.py)                                                │
    │    - เชื่อมต่อ Local Ollama (qwen2.5-coder) หรือ API ภายนอก                            │
    │    - Token Streaming แบบ Real-Time แสดงผลบนหน้าเว็บทันที                               │
    │    - Intelligent Simulation Fallback ทำงานได้ทันทีแม้ออฟไลน์                          │
    └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                                │
                                                ▼
    ┌────────────────────────────────────────────────────────────────────────────────────────┐
    │ 5. CODE EXTRACTION & POST-PROCESSING (code_runner.py)                                 │
    │    - สกัดโค้ด Python จาก Markdown codeblock ด้วย Regular Expression อัตโนมัติ          │
    │    - ซิงค์โค้ดไปยัง Code Sandbox พร้อมกดรันได้ทันที                                    │
    └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                                │
                                                ▼
    ┌────────────────────────────────────────────────────────────────────────────────────────┐
    │ 6. SAFE EXECUTION SANDBOX (code_runner.py)                                             │
    │    - รัน Python Subprocess ในสภาพแวดล้อมที่จำกัดเวลา (Timeout Guard 10 วินาที)         │
    │    - ดักจับ stdout, stderr, Execution Time และ Exit Code ครบถ้วน                       │
    └────────────────────────────────────────────────────────────────────────────────────────┘
    ```
    """)
    
    st.divider()

    st.markdown("### 🔹 ขั้นตอนที่ 1: เจาะลึกการทำงานของ Hybrid RAG (`hybrid_retriever.py`)")
    st.markdown("""
    #### 💡 ทำไมต้องทำ RAG แบบ Hybrid สำหรับงานเขียนโค้ด?
    ในการสืบค้นชุดข้อมูลโค้ด การค้นหาแบบวิธีเดียว (Single-Method Retrieval) มักเกิดจุดอ่อนร้ายแรง:
    """)

    st.markdown("""
    | คุณสมบัติ | Sparse Only (BM25) | Dense Only (Vector Embedding) | 🌟 Hybrid RAG (BM25 + Dense) |
    | :--- | :--- | :--- | :--- |
    | **การจับคู่ชื่อฟังก์ชัน/API ตัวย่อ** (e.g. `LRU`, `BFS`, `BST`, `FastAPI`) | 🟢 แม่นยำสูงมาก 100% | 🟡 มักคลาดเคลื่อนเนื่องจากตัวย่อมี Embedding ใกล้เคียงคำอื่น | 🟢 **แม่นยำสูง (ได้คะแนน BM25 ดันขึ้น)** |
    | **การเข้าใจความหมาย/ภาษาไทย** (e.g. `การค้นหาแบบทวิภาค`, `การจัดกระเป๋า`) | 🔴 ไม่พบผลลัพธ์ถ้าคำไม่ตรงตัว | 🟢 เข้าใจความหมายและจับคู่คอนเซปต์ได้ | 🟢 **เข้าใจบริบท (ได้คะแนน Dense Vector ดันขึ้น)** |
    | **ความทนทานต่อคำผิด/Synonyms** | 🔴 ต่ำมาก | 🟢 สูงมาก | 🟢 **สมบูรณ์แบบทั้งสองมิติ** |
    | **ความเร็วและความต้องการระบบ** | ⚡ เร็วมาก ไม่ต้องใช้ GPU | 🐢 ต้องคำนวณเวกเตอร์ | ⚡ **เร็วระดับมิลลิวินาที (Numpy Vectorization)** |
    """)

    st.markdown(r"""
    #### 📐 คณิตศาสตร์เบื้องหลังระบบ Hybrid RAG:
    
    **1. Okapi BM25 (Sparse Retrieval):**
    $$Score_{BM25}(D, Q) = \sum_{q_i \in Q} IDF(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{avgdl}\right)}$$
    * โดย $k_1 = 1.5$ (Term Frequency Saturation) และ $b = 0.75$ (Document Length Normalization)
    * มีการกำหนดค่าน้ำหนักพิเศษ: Title ($3\times$), Keywords ($2\times$), Category ($2\times$) และ Code Body ($1\times$)

    **2. Dense Semantic Vector Space:**
    * สร้างเวกเตอร์ความหมาย Subword & Character N-Grams (3-gram, 4-gram)
    * คำนวณ Cosine Similarity ด้วย Matrix Dot Product ระหว่าง Unit Vectors:
    $$Similarity(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \|\vec{d}\|}$$

    **3. กลยุทธ์การผสานผลลัพธ์ (Fusion Strategies):**
    * **Weighted Linear Combination:**
      $$Score_{hybrid} = \alpha \cdot Score_{dense} + (1 - \alpha) \cdot Score_{sparse}$$
      *(ผู้ใช้สามารถปรับค่า $\alpha$ บนหน้าจอได้อย่างอิสระ)*
    * **Reciprocal Rank Fusion (RRF, $k=60$):**
      $$RRF(d) = \frac{1}{60 + Rank_{BM25}(d)} + \frac{1}{60 + Rank_{Dense}(d)}$$
      *(มาตรฐานอุตสาหกรรมที่ใช้ในระบบค้นหาระดับโลก)*
    """)

    st.markdown("### 🔹 ขั้นตอนที่ 2: ระบบ Prompt Engineering Pipeline (`prompts.py`)")
    st.markdown("""
    * ผสาน System Persona เข้ากับชุด Ground-Truth Code Reference ที่ได้จาก Hybrid RAG
    * บังคับโครงสร้าง Output ให้มี Type Annotations, Docstrings, ตัวอย่างการเรียกใช้, และ Big-O Analysis
    """)

    st.markdown("### 🔹 ขั้นตอนที่ 3: ระบบ LLM Inference Engine (`llm_engine.py`)")
    st.markdown("""
    * รองรับ Ollama Local API, OpenAI-compatible API, และ Intelligent Simulation Engine
    * ใช้ Generator Token Streaming เพื่อประสบการณ์ใช้งานที่ลื่นไหลแบบ Real-Time
    """)

    st.markdown("### 🔹 ขั้นตอนที่ 4: ระบบทดสอบรันโค้ด Sandbox (`code_runner.py`)")
    st.markdown("""
    * แยกโค้ดออกจากคำตอบของ AI ด้วย Regex แล้วส่งต่อไปยัง Code Editor ใน Sandbox ทันที
    * รันใน Isolated Subprocess พร้อมจับเวลา Execution Time และดักจับ Timeout (10 วินาที)
    """)

    st.markdown("### 🔹 ขั้นตอนที่ 5: ระบบ Code Knowledge Graph Database (`code_graph_engine.py`)")
    st.markdown("""
    * **Property Graph Data Model บน Kùzu DB:**
      - **Nodes:** `Module` (ชื่อโมดูลและที่อยู่ไฟล์), `Function` (ชื่อฟังก์ชัน, บรรทัด, Docstring, พารามิเตอร์), `Class` (ชื่อคลาส, บรรทัด)
      - **Relationships:** `(Function)-[:CALLS]->(Function)`, `(Function)-[:DEFINED_IN]->(Module)`, `(Module)-[:IMPORTS]->(Module)`
    * **การวิเคราะห์ AST (Abstract Syntax Tree):** ใช้ Python Standard Library `ast` สแกนโครงสร้างโค้ดแบบ Static Analysis โดยไม่ต้องรันโค้ดจริง
    * **การสืบค้นแบบ Cypher:** รองรับคำสั่ง Cypher Query มาตรฐานสำหรับการ Traversal กราฟ (เช่น `MATCH (caller)-[:CALLS]->(target) WHERE target.name='foo' RETURN caller.name`)
    * **การแสดงผล Mermaid:** แปลงกราฟความสัมพันธ์ออกมาเป็น Mermaid Syntax เพื่อเรนเดอร์แผนภาพแบบ Interactive ใน Streamlit
    """)

    st.markdown("### 🔹 ขั้นตอนที่ 6: ระบบ Dynamic Query Router Pattern (`QuestionRouter`)")
    st.markdown("""
    * **การคัดแยกคำถามอัตโนมัติ (Query Classification):**
      - **Graph Route:** ตรวจพบคีย์เวิร์ดเชิงโครงสร้าง เช่น `ใครเรียก`, `ลำดับการเรียก`, `callers`, `callees`, `hierarchy`, `dependency`, `นำเข้าโมดูล`
      - **Vector/Hybrid Route:** คำถามเชิงคอนเซปต์ หรือต้องการสร้างโค้ดใหม่ เช่น `เขียนโค้ด`, `อัลกอริทึม`, `สร้าง API`, `binary search`
    * **ประโยชน์:** ป้องกันไม่ให้ Vector Search ต้องประมวลผลคำถามเชิงโครงสร้างที่ Vector ตอบได้ไม่ดี และลด Context ที่เกินความจำเป็นส่งไปยัง LLM
    """)


