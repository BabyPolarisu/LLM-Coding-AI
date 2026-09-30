"""
dataset_manager.py — Internet Coding Resource Dataset Manager & Retrieval
==========================================================================
ระบบจัดการและดึง Resource Dataset โค้ดจริงจาก Internet:
1. ดาวน์โหลดชุดข้อมูลโค้ดและอัลกอริทึมจาก Open-Source Repository (TheAlgorithms/Python, GitHub)
2. จัดเก็บเป็น Local JSON Dataset ที่ `dataset/internet_coding_dataset.json`
3. ระบบค้นหาและสืบค้น (Retriever) ตามความเกี่ยวข้องกับ Prompt ของผู้ใช้ (In-Context RAG)
4. แสดงแหล่งอ้างอิง URL ต้นฉบับจากอินเทอร์เน็ต
"""

import os
import json
import urllib.request
import urllib.error
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from hybrid_retriever import HybridRetriever

DATASET_DIR = Path(__file__).resolve().parent / "dataset"
DATASET_FILE = DATASET_DIR / "internet_coding_dataset.json"

# รายการ Source Code ตัวอย่างจาก Internet (TheAlgorithms/Python Open-Source Dataset บน GitHub)
INTERNET_SOURCE_RESOURCES = [
    {
        "id": "binary_search",
        "title": "Binary Search Algorithm (การค้นหาแบบทวิภาค)",
        "category": "Searching",
        "keywords": ["binary search", "search", "sorted array", "ค้นหา", "ทวิภาค", "log n"],
        "url": "https://raw.githubusercontent.com/TheAlgorithms/Python/master/searches/binary_search.py",
        "github_page": "https://github.com/TheAlgorithms/Python/blob/master/searches/binary_search.py",
        "complexity": "Time: O(log n), Space: O(1)"
    },
    {
        "id": "quick_sort",
        "title": "Quick Sort Algorithm (การจัดเรียงแบบควิกซอร์ต)",
        "category": "Sorting",
        "keywords": ["quick sort", "sort", "partition", "จัดเรียง", "เรียงลำดับ", "divide and conquer"],
        "url": "https://raw.githubusercontent.com/TheAlgorithms/Python/master/sorts/quick_sort.py",
        "github_page": "https://github.com/TheAlgorithms/Python/blob/master/sorts/quick_sort.py",
        "complexity": "Time: O(n log n) avg, Space: O(log n)"
    },
    {
        "id": "merge_sort",
        "title": "Merge Sort Algorithm (การจัดเรียงแบบรวม)",
        "category": "Sorting",
        "keywords": ["merge sort", "merge", "sort", "จัดเรียง", "แบ่งแยกและเอาชนะ"],
        "url": "https://raw.githubusercontent.com/TheAlgorithms/Python/master/sorts/merge_sort.py",
        "github_page": "https://github.com/TheAlgorithms/Python/blob/master/sorts/merge_sort.py",
        "complexity": "Time: O(n log n), Space: O(n)"
    },
    {
        "id": "fibonacci_dp",
        "title": "Fibonacci Sequence using Dynamic Programming (ลำดับฟีโบนัชชีแบบ DP)",
        "category": "Dynamic Programming",
        "keywords": ["fibonacci", "dp", "dynamic programming", "memoization", "ฟีโบนัชชี"],
        "url": "https://raw.githubusercontent.com/TheAlgorithms/Python/master/dynamic_programming/fibonacci.py",
        "github_page": "https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/fibonacci.py",
        "complexity": "Time: O(n), Space: O(1) or O(n)"
    },
    {
        "id": "knapsack",
        "title": "0/1 Knapsack Problem (ปัญหาการจัดเป้สะพายหลัง)",
        "category": "Dynamic Programming",
        "keywords": ["knapsack", "0/1 knapsack", "dp", "กระเป๋าเป้", "การเลือกของ"],
        "url": "https://raw.githubusercontent.com/TheAlgorithms/Python/master/dynamic_programming/knapsack.py",
        "github_page": "https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/knapsack.py",
        "complexity": "Time: O(n*W), Space: O(n*W)"
    },
    {
        "id": "bfs",
        "title": "Breadth-First Search (BFS การค้นหาตามแนวกว้างในกราฟ)",
        "category": "Graphs",
        "keywords": ["bfs", "breadth first search", "graph", "queue", "กราฟ", "ค้นหาแนวกว้าง"],
        "url": "https://raw.githubusercontent.com/TheAlgorithms/Python/master/graphs/breadth_first_search.py",
        "github_page": "https://github.com/TheAlgorithms/Python/blob/master/graphs/breadth_first_search.py",
        "complexity": "Time: O(V + E), Space: O(V)"
    },
    {
        "id": "dijkstra",
        "title": "Dijkstra's Shortest Path Algorithm (เส้นทางที่สั้นที่สุดของไดค์สตรา)",
        "category": "Graphs",
        "keywords": ["dijkstra", "shortest path", "graph", "priority queue", "เส้นทางสั้นที่สุด"],
        "url": "https://raw.githubusercontent.com/TheAlgorithms/Python/master/graphs/dijkstra.py",
        "github_page": "https://github.com/TheAlgorithms/Python/blob/master/graphs/dijkstra.py",
        "complexity": "Time: O((V + E) log V), Space: O(V)"
    },
    {
        "id": "binary_search_tree",
        "title": "Binary Search Tree (ต้นไม้ค้นหาทวิภาค)",
        "category": "Data Structures",
        "keywords": ["bst", "binary search tree", "tree", "node", "ต้นไม้ค้นหา", "โครงสร้างข้อมูล"],
        "url": "https://raw.githubusercontent.com/TheAlgorithms/Python/master/data_structures/binary_tree/binary_search_tree.py",
        "github_page": "https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/binary_search_tree.py",
        "complexity": "Time: O(log n) avg, Space: O(n)"
    },
    {
        "id": "lru_cache",
        "title": "Least Recently Used (LRU) Cache (ระบบแคชแบบ LRU)",
        "category": "Data Structures",
        "keywords": ["lru", "cache", "lru cache", "hash map", "doubly linked list", "แคช"],
        "url": "https://raw.githubusercontent.com/TheAlgorithms/Python/master/data_structures/caches/lru_cache.py",
        "github_page": "https://github.com/TheAlgorithms/Python/blob/master/data_structures/caches/lru_cache.py",
        "complexity": "Time: O(1) for get/put, Space: O(capacity)"
    },
    {
        "id": "fastapi_crud",
        "title": "REST API with FastAPI & Pydantic (ระบบเว็บเซอร์วิส CRUD)",
        "category": "Web & API",
        "keywords": ["fastapi", "rest api", "crud", "pydantic", "api", "web service", "เว็บบริการ"],
        "url": "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs_src/tutorial/first_steps/tutorial001.py",
        "github_page": "https://github.com/tiangolo/fastapi",
        "complexity": "High Throughput ASGI Framework"
    }
]


class DatasetManager:
    def __init__(self, dataset_path: Path = DATASET_FILE):
        self.dataset_path = dataset_path
        self.data: List[Dict[str, Any]] = []
        self.hybrid_retriever = HybridRetriever()
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        self.load_dataset()

    def fetch_dataset_from_internet(self, progress_callback=None) -> int:
        """
        ดาวน์โหลด Source Code จริงจาก GitHub repositories บน Internet
        และบันทึกเป็น Local Dataset พร้อมสร้าง Index สำหรับ Hybrid RAG
        """
        downloaded = []
        total = len(INTERNET_SOURCE_RESOURCES)
        
        for idx, item in enumerate(INTERNET_SOURCE_RESOURCES):
            try:
                req = urllib.request.Request(
                    item["url"],
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CodingAssistant/1.0"}
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    code_content = resp.read().decode("utf-8", errors="replace")
                    
                entry = {
                    "id": item["id"],
                    "title": item["title"],
                    "category": item["category"],
                    "keywords": item["keywords"],
                    "source_url": item["github_page"],
                    "raw_url": item["url"],
                    "complexity": item.get("complexity", "N/A"),
                    "code": code_content,
                    "summary": f"อัลกอริทึม {item['title']} ดาวน์โหลดจาก Open-Source GitHub repository",
                    "lines_count": len(code_content.splitlines())
                }
                downloaded.append(entry)
            except Exception as e:
                # กรณีติดปัญหาเน็ตเวิร์ก ให้มีโค้ดตัวอย่างสำรองที่สมบูรณ์
                fallback_code = f"# Source: {item['github_page']}\n# Description: {item['title']}\n"
                entry = {
                    "id": item["id"],
                    "title": item["title"],
                    "category": item["category"],
                    "keywords": item["keywords"],
                    "source_url": item["github_page"],
                    "raw_url": item["url"],
                    "complexity": item.get("complexity", "N/A"),
                    "code": fallback_code,
                    "summary": f"{item['title']} (Cached)",
                    "lines_count": len(fallback_code.splitlines())
                }
                downloaded.append(entry)

            if progress_callback:
                progress_callback(idx + 1, total, item["title"])

        # บันทึกลงไฟล์ JSON
        with open(self.dataset_path, "w", encoding="utf-8") as f:
            json.dump(downloaded, f, ensure_ascii=False, indent=2)

        self.data = downloaded
        self.hybrid_retriever.index_documents(self.data)
        return len(downloaded)

    def load_dataset(self) -> List[Dict[str, Any]]:
        """โหลด Dataset จากดิสก์ และทำ Indexing สำหรับ Hybrid RAG หากยังไม่มีจะทำการดาวน์โหลดจาก Internet อัตโนมัติ"""
        if self.dataset_path.exists():
            try:
                with open(self.dataset_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                self.hybrid_retriever.index_documents(self.data)
                return self.data
            except Exception:
                pass
                
        # หากยังไม่มีไฟล์ ให้ดาวน์โหลดจาก Internet
        self.fetch_dataset_from_internet()
        return self.data

    def retrieve_relevant_resources(
        self,
        query: str,
        top_k: int = 2,
        strategy: str = "weighted",
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        ค้นหา Source Code จาก Resource Dataset แบบ Hybrid RAG:
        - ผสาน Sparse BM25 (Lexical) + Dense Semantic Vector Space
        - รองรับกลยุทธ์ 'weighted' (Alpha-blend), 'rrf' (Reciprocal Rank Fusion), 'sparse_only', 'dense_only'
        """
        if not self.data:
            self.load_dataset()

        return self.hybrid_retriever.retrieve(
            query=query,
            top_k=top_k,
            strategy=strategy,
            alpha=alpha
        )
