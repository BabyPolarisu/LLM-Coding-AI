# 💻 AI Coding Assistant — MiniProject


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
