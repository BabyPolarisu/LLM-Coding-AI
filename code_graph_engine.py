# -*- coding: utf-8 -*-
"""
code_graph_engine.py — Production Code Knowledge Graph Engine (Kùzu DB)
========================================================================
สถาปัตยกรรม Graph Database สำหรับวิเคราะห์โครงสร้าง Codebase (Session 11-12):
1. Node Tables: Function, Module, Class
2. Relationship Tables: CALLS, DEFINED_IN, CLASS_DEFINED_IN, IMPORTS
3. High-level Architecture Analysis:
   - Call Hierarchy Tracing (Caller / Callee)
   - Cross-module Dependency Tracking
   - Mermaid Graph Visualization Generator
   - Cypher Query Execution Interface
"""

import os
import ast
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import kuzu
import networkx as nx

GRAPH_STORAGE_DIR = Path(__file__).resolve().parent / "graph_db_storage"
_GLOBAL_KUZU_DB: Optional[kuzu.Database] = None


class CodeGraphEngine:
    """
    ระบบวิเคราะห์และจัดเก็บ Code Knowledge Graph บน Kùzu Graph Database
    """

    def __init__(self, db_path: Path = GRAPH_STORAGE_DIR):
        self.db_path = db_path
        self.db: Optional[kuzu.Database] = None
        self.conn: Optional[kuzu.Connection] = None
        self._init_connection()

    def _init_connection(self):
        """เชื่อมต่อฐานข้อมูล Kùzu หรือสร้างใหม่ถ้ายังไม่มี (ใช้ Shared Instance เพื่อป้องกัน file lock)"""
        global _GLOBAL_KUZU_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if _GLOBAL_KUZU_DB is None:
            try:
                _GLOBAL_KUZU_DB = kuzu.Database(str(self.db_path))
            except Exception as err:
                # หากมี Process อื่นเปิดใช้งานอยู่ หรือติด file lock บน Windows
                # ให้ fallback ไปใช้ In-Memory Graph Database ทันที เพื่อให้เว็บทำงานต่อเนื่องได้ 100%
                print(f"[Warning] Kuzu storage lock or corruption ({err}). Recreating database.")
                try:
                    # Clean up corrupted wal/db if invalid
                    if self.db_path.exists():
                        if self.db_path.is_dir():
                            shutil.rmtree(self.db_path, ignore_errors=True)
                        else:
                            self.db_path.unlink(missing_ok=True)
                    wal_file = self.db_path.parent / (self.db_path.name + ".wal")
                    if wal_file.exists():
                        wal_file.unlink(missing_ok=True)
                    _GLOBAL_KUZU_DB = kuzu.Database(str(self.db_path))
                except Exception:
                    _GLOBAL_KUZU_DB = kuzu.Database("")
        self.db = _GLOBAL_KUZU_DB
        self.conn = kuzu.Connection(self.db)
        self._create_schema()

    def _create_schema(self):
        """สร้าง Table Schemas สำหรับ Nodes และ Relationships"""
        try:
            # Nodes
            self.conn.execute(
                "CREATE NODE TABLE IF NOT EXISTS Function ("
                "name STRING, file STRING, line INT64, docstring STRING, args_count INT64, "
                "PRIMARY KEY (name))"
            )
            self.conn.execute(
                "CREATE NODE TABLE IF NOT EXISTS Module ("
                "name STRING, filepath STRING, "
                "PRIMARY KEY (name))"
            )
            self.conn.execute(
                "CREATE NODE TABLE IF NOT EXISTS Class ("
                "name STRING, file STRING, line INT64, "
                "PRIMARY KEY (name))"
            )
            # Relationships
            self.conn.execute("CREATE REL TABLE IF NOT EXISTS CALLS (FROM Function TO Function)")
            self.conn.execute("CREATE REL TABLE IF NOT EXISTS DEFINED_IN (FROM Function TO Module)")
            self.conn.execute("CREATE REL TABLE IF NOT EXISTS CLASS_DEFINED_IN (FROM Class TO Module)")
            self.conn.execute("CREATE REL TABLE IF NOT EXISTS IMPORTS (FROM Module TO Module)")
        except Exception:
            pass

    def clear_database(self):
        """ล้างฐานข้อมูลกราฟเพื่อ Index ใหม่"""
        try:
            del self.conn
            del self.db
            self.conn = None
            self.db = None
            if self.db_path.exists():
                if self.db_path.is_dir():
                    shutil.rmtree(self.db_path)
                else:
                    self.db_path.unlink()
        except Exception:
            pass
        self._init_connection()

    def index_codebase(self, directory_path: str) -> Dict[str, Any]:
        """
        สแกนโค้ด Python ในโฟลเดอร์ แปลงเป็น AST และสร้าง Knowledge Graph ใน Kùzu
        """
        self.clear_database()
        base_dir = Path(directory_path).resolve()
        if not base_dir.exists():
            return {"error": f"Directory not found: {directory_path}"}

        skip_dirs = {".venv", "__pycache__", ".git", ".mypy_cache", "node_modules", "dist", "build"}
        py_files = []
        for root, dirs, files in os.walk(base_dir):
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
            for f in files:
                if f.endswith(".py"):
                    py_files.append(Path(root) / f)

        total_functions = 0
        total_classes = 0
        total_calls = 0
        total_imports = 0

        # Step 1: สร้าง Modules, Classes, และ Functions ทั้งหมดก่อน
        defined_functions = set()
        file_ast_cache = {}

        for fpath in py_files:
            try:
                src = fpath.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(src, filename=str(fpath))
                file_ast_cache[fpath] = (src, tree)
                mod_name = fpath.stem

                # Insert Module
                self.conn.execute(
                    "MERGE (:Module {name: $n, filepath: $fp})",
                    {"n": mod_name, "fp": str(fpath.relative_to(base_dir))}
                )

                lines = src.splitlines()
                for node in ast.walk(tree):
                    # Classes
                    if isinstance(node, ast.ClassDef):
                        self.conn.execute(
                            "MERGE (:Class {name: $n, file: $f, line: $l})",
                            {"n": node.name, "f": fpath.name, "l": node.lineno}
                        )
                        self.conn.execute(
                            "MATCH (c:Class),(m:Module) WHERE c.name=$cn AND m.name=$mn "
                            "MERGE (c)-[:CLASS_DEFINED_IN]->(m)",
                            {"cn": node.name, "mn": mod_name}
                        )
                        total_classes += 1

                    # Functions
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        fn_name = node.name
                        defined_functions.add(fn_name)
                        doc = ast.get_docstring(node) or ""
                        doc_snippet = doc[:200].replace("\n", " ").strip()
                        args_count = len(node.args.args)

                        self.conn.execute(
                            "MERGE (:Function {name: $n, file: $f, line: $l, docstring: $d, args_count: $a})",
                            {"n": fn_name, "f": fpath.name, "l": node.lineno, "d": doc_snippet, "a": args_count}
                        )
                        self.conn.execute(
                            "MATCH (f:Function),(m:Module) WHERE f.name=$fn AND m.name=$mn "
                            "MERGE (f)-[:DEFINED_IN]->(m)",
                            {"fn": fn_name, "mn": mod_name}
                        )
                        total_functions += 1

            except Exception:
                continue

        # Step 2: สร้างความสัมพันธ์ CALLS และ IMPORTS
        for fpath, (src, tree) in file_ast_cache.items():
            mod_name = fpath.stem

            for node in ast.walk(tree):
                # Imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imp_name = alias.name.split(".")[0]
                        chk_mod = self.conn.execute("MATCH (m:Module) WHERE m.name=$n RETURN count(m) AS cnt", {"n": imp_name})
                        if chk_mod.get_as_df()["cnt"][0] > 0:
                            self.conn.execute(
                                "MATCH (m1:Module),(m2:Module) WHERE m1.name=$m1 AND m2.name=$m2 MERGE (m1)-[:IMPORTS]->(m2)",
                                {"m1": mod_name, "m2": imp_name}
                            )
                            total_imports += 1

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imp_name = node.module.split(".")[0]
                        chk_mod = self.conn.execute("MATCH (m:Module) WHERE m.name=$n RETURN count(m) AS cnt", {"n": imp_name})
                        if chk_mod.get_as_df()["cnt"][0] > 0:
                            self.conn.execute(
                                "MATCH (m1:Module),(m2:Module) WHERE m1.name=$m1 AND m2.name=$m2 MERGE (m1)-[:IMPORTS]->(m2)",
                                {"m1": mod_name, "m2": imp_name}
                            )
                            total_imports += 1

                # Function Calls
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    caller_name = node.name
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            callee = None
                            if isinstance(child.func, ast.Name):
                                callee = child.func.id
                            elif isinstance(child.func, ast.Attribute):
                                callee = child.func.attr

                            if callee and callee != caller_name:
                                # Ensure node exists
                                chk = self.conn.execute("MATCH (f:Function) WHERE f.name=$n RETURN count(f) AS cnt", {"n": callee})
                                if chk.get_as_df()["cnt"][0] == 0:
                                    self.conn.execute(
                                        "CREATE (:Function {name: $n, file: '', line: 0, docstring: '', args_count: 0})",
                                        {"n": callee}
                                    )
                                self.conn.execute(
                                    "MATCH (a:Function),(b:Function) WHERE a.name=$a AND b.name=$b MERGE (a)-[:CALLS]->(b)",
                                    {"a": caller_name, "b": callee}
                                )
                                total_calls += 1

        return {
            "status": "success",
            "base_dir": str(base_dir),
            "files_scanned": len(py_files),
            "functions": total_functions,
            "classes": total_classes,
            "calls": total_calls,
            "imports": total_imports
        }

    def get_stats(self) -> Dict[str, int]:
        """ดึงสถิติจำนวน Nodes และ Edges ในฐานข้อมูลกราฟ"""
        try:
            fn_res = self.conn.execute("MATCH (f:Function) WHERE f.file <> '' RETURN count(f) AS cnt").get_as_df()
            fn_count = int(fn_res["cnt"][0]) if not fn_res.empty else 0

            mod_res = self.conn.execute("MATCH (m:Module) RETURN count(m) AS cnt").get_as_df()
            mod_count = int(mod_res["cnt"][0]) if not mod_res.empty else 0

            call_res = self.conn.execute("MATCH ()-[r:CALLS]->() RETURN count(r) AS cnt").get_as_df()
            call_count = int(call_res["cnt"][0]) if not call_res.empty else 0

            imp_res = self.conn.execute("MATCH ()-[r:IMPORTS]->() RETURN count(r) AS cnt").get_as_df()
            imp_count = int(imp_res["cnt"][0]) if not imp_res.empty else 0

            return {
                "functions": fn_count,
                "modules": mod_count,
                "calls": call_count,
                "imports": imp_count
            }
        except Exception:
            return {"functions": 0, "modules": 0, "calls": 0, "imports": 0}

    def get_callers(self, function_name: str) -> List[Dict[str, Any]]:
        """ใครเรียกฟังก์ชันนี้บ้าง? (Find all Callers of a Function)"""
        cypher = (
            "MATCH (caller:Function)-[:CALLS]->(target:Function) "
            "WHERE target.name=$name "
            "RETURN caller.name AS caller_name, caller.file AS file, caller.line AS line"
        )
        try:
            df = self.conn.execute(cypher, {"name": function_name}).get_as_df()
            return df.to_dict(orient="records")
        except Exception:
            return []

    def get_callees(self, function_name: str) -> List[Dict[str, Any]]:
        """ฟังก์ชันนี้เรียกฟังก์ชันอื่นตัวใดบ้าง? (Find all Callee functions)"""
        cypher = (
            "MATCH (caller:Function)-[:CALLS]->(callee:Function) "
            "WHERE caller.name=$name "
            "RETURN callee.name AS callee_name, callee.file AS file, callee.line AS line"
        )
        try:
            df = self.conn.execute(cypher, {"name": function_name}).get_as_df()
            return df.to_dict(orient="records")
        except Exception:
            return []

    def get_module_summary(self, module_name: str) -> Dict[str, Any]:
        """สรุปข้อมูลฟังก์ชันและการ Import ของโมดูล"""
        cypher_fn = (
            "MATCH (f:Function)-[:DEFINED_IN]->(m:Module) "
            "WHERE m.name=$name "
            "RETURN f.name AS function_name, f.line AS line, f.docstring AS docstring"
        )
        cypher_imp = (
            "MATCH (m:Module)-[:IMPORTS]->(dep:Module) "
            "WHERE m.name=$name "
            "RETURN dep.name AS imported_module"
        )
        try:
            funcs = self.conn.execute(cypher_fn, {"name": module_name}).get_as_df().to_dict(orient="records")
            deps = self.conn.execute(cypher_imp, {"name": module_name}).get_as_df().to_dict(orient="records")
            return {"functions": funcs, "imports": [d["imported_module"] for d in deps]}
        except Exception:
            return {"functions": [], "imports": []}

    def trace_call_hierarchy(self, entry_func: str, max_depth: int = 3) -> Dict[str, Any]:
        """วิเคราะห์ลำดับการเรียกต่อเนื่อง (Call Hierarchy Tree) หลายชั้น"""
        visited = set()
        tree = {"name": entry_func, "children": []}

        def _dfs(current: str, node: dict, depth: int):
            if depth >= max_depth or current in visited:
                return
            visited.add(current)
            callees = self.get_callees(current)
            for c in callees:
                c_name = c["callee_name"]
                child_node = {"name": c_name, "file": c.get("file", ""), "children": []}
                node["children"].append(child_node)
                _dfs(c_name, child_node, depth + 1)

        _dfs(entry_func, tree, 0)
        return tree

    def generate_mermaid_diagram(self, max_edges: int = 30) -> str:
        """สร้าง Mermaid Graph Definition เพื่อเรนเดอร์ภาพความสัมพันธ์ใน Streamlit"""
        cypher = (
            "MATCH (a:Function)-[:CALLS]->(b:Function) "
            "WHERE a.file <> '' "
            "RETURN a.name AS src, b.name AS dst LIMIT $lim"
        )
        try:
            df = self.conn.execute(cypher, {"lim": max_edges}).get_as_df()
            if df.empty:
                return "graph LR\n  A[No Relationships Indexed Yet] --> B[Run Indexer]"

            lines = ["graph LR"]
            for _, row in df.iterrows():
                src = row["src"]
                dst = row["dst"]
                lines.append(f"  {src}([{src}]) --> {dst}([{dst}])")
            return "\n".join(lines)
        except Exception as e:
            return f"graph LR\n  Error[{str(e)}]"

    def get_all_function_names(self) -> List[str]:
        """ดึงรายชื่อฟังก์ชันทั้งหมดที่มีการประกาศในโค้ด (มีไฟล์ระบุ)"""
        try:
            df = self.conn.execute("MATCH (f:Function) WHERE f.file <> '' RETURN f.name AS name ORDER BY f.name").get_as_df()
            return df["name"].tolist() if not df.empty else []
        except Exception:
            return []

    def get_all_module_names(self) -> List[str]:
        """ดึงรายชื่อโมดูลทั้งหมดที่สแกนพบ"""
        try:
            df = self.conn.execute("MATCH (m:Module) RETURN m.name AS name ORDER BY m.name").get_as_df()
            return df["name"].tolist() if not df.empty else []
        except Exception:
            return []

    def generate_function_mermaid(self, func_name: str) -> str:
        """สร้างไดอะแกรม Mermaid เฉพาะเจาะจงสำหรับฟังก์ชันที่เลือก (แสดง Callers และ Callees)"""
        callers = self.get_callers(func_name)
        callees = self.get_callees(func_name)

        if not callers and not callees:
            return f"graph TD\n  target[\"🎯 {func_name} (ไม่มีความสัมพันธ์เชื่อมโยง)\"]"

        lines = ["graph TD"]
        # กำหนดสไตล์ให้ target function เด่นชัด
        lines.append(f"  style target fill:#3b82f6,stroke:#1d4ed8,stroke-width:3px,color:#ffffff,font-weight:bold")
        lines.append(f"  target[\"🎯 {func_name}\"]")

        for idx, c in enumerate(callers):
            c_name = c["caller_name"]
            c_id = f"caller_{idx}"
            lines.append(f"  {c_id}[\"⚡ {c_name}\"] -->|calls| target")

        for idx, c in enumerate(callees):
            c_name = c["callee_name"]
            c_id = f"callee_{idx}"
            lines.append(f"  target -->|calls| {c_id}[\"📦 {c_name}\"]")

        return "\n".join(lines)

    def execute_cypher(self, query: str) -> Tuple[Optional[Any], str]:
        """รันคำสั่ง Cypher อิสระ"""
        try:
            res = self.conn.execute(query)
            df = res.get_as_df()
            return df, "success"
        except Exception as e:
            return None, str(e)

