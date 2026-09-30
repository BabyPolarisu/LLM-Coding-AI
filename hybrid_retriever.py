"""
hybrid_retriever.py — Production-Grade Hybrid RAG Retrieval Engine
==================================================================
ระบบ Hybrid Search & Retrieval-Augmented Generation (Hybrid RAG):
1. Sparse Retrieval: Okapi BM25 (Lexical / Keyword matching with IDF and length normalization)
2. Dense Retrieval: Semantic Vector Space (Cosine Similarity with subword n-gram embeddings & API support)
3. Hybrid Fusion:
   - Weighted Score Combination (alpha * dense + (1 - alpha) * sparse)
   - Reciprocal Rank Fusion (RRF with standard k=60)
4. Transparent Metrics: BM25 score, Semantic similarity, RRF rank, and matched keywords
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


def tokenize_text(text: str) -> List[str]:
    """
    ตัดคำและสกัด Tokens รองรับทั้งภาษาอังกฤษ โค้ดโปรแกรม (camelCase, snake_case) และภาษาไทย
    """
    if not text:
        return []
    
    # แยก camelCase เช่น binarySearch -> binary Search
    s1 = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # แปลง snake_case และเครื่องหมายวรรคตอนเป็นช่องว่าง
    s2 = re.sub(r'[_\-\./\\:()\[\]{},"\'`<>+*=;?!#@$%^&~|]', ' ', s1)
    
    # สกัด words
    tokens = [t.lower().strip() for t in re.findall(r'[\w]+', s2) if len(t.strip()) > 0]
    
    # กรอง stop words ทั่วไปในภาษาอังกฤษ
    stop_words = {
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by",
        "and", "or", "is", "are", "was", "were", "it", "this", "that"
    }
    return [t for t in tokens if t not in stop_words and len(t) > 1]


class BM25Retriever:
    """
    ระบบค้นหาแบบ Sparse / Lexical โดยใช้อัลกอริทึม Okapi BM25
    - คำนวณ Inverse Document Frequency (IDF)
    - ปรับความยาวเอกสาร (Document Length Normalization)
    - กำหนดน้ำหนักพิเศษให้ Title, Keywords, Category และ Code
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[Dict[str, Any]] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_count: int = 0
        self.idf: Dict[str, float] = {}

    def index(self, documents: List[Dict[str, Any]]) -> None:
        """สร้างดัชนี BM25 จากชุดเอกสาร"""
        self.corpus = documents
        self.doc_count = len(documents)
        self.doc_tokens = []
        self.doc_lengths = []

        if self.doc_count == 0:
            return

        # คำนวณ Document Frequency สำหรับคำนวณ IDF
        df: Dict[str, int] = {}

        for doc in documents:
            # รวมข้อความจากส่วนต่างๆ โดยให้น้ำหนัก Title และ Keywords สูงกว่า
            title_tokens = tokenize_text(doc.get("title", "")) * 3
            kw_tokens = tokenize_text(" ".join(doc.get("keywords", []))) * 2
            cat_tokens = tokenize_text(doc.get("category", "")) * 2
            id_tokens = tokenize_text(doc.get("id", "")) * 3
            summary_tokens = tokenize_text(doc.get("summary", ""))
            code_sample = doc.get("code", "")[:1200]
            code_tokens = tokenize_text(code_sample)

            combined_tokens = title_tokens + kw_tokens + cat_tokens + id_tokens + summary_tokens + code_tokens
            self.doc_tokens.append(combined_tokens)
            self.doc_lengths.append(len(combined_tokens))

            # เก็บ unique tokens ของเอกสารนี้เพื่อคำนวณ df
            unique_terms = set(combined_tokens)
            for term in unique_terms:
                df[term] = df.get(term, 0) + 1

        self.avg_doc_len = sum(self.doc_lengths) / max(1, self.doc_count)

        # คำนวณ Okapi BM25 IDF: ln((N - n + 0.5) / (n + 0.5) + 1)
        self.idf = {}
        for term, freq in df.items():
            idf_score = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[term] = max(0.1, idf_score)

    def score(self, query: str) -> List[Tuple[float, Dict[str, Any], List[str]]]:
        """
        คำนวณคะแนน BM25 สำหรับคำค้นหา
        คืนค่า: รายการของ tuple (bm25_score_normalized, document, matched_terms)
        """
        if not self.corpus:
            return []

        q_tokens = tokenize_text(query)
        if not q_tokens:
            return [(0.0, doc, []) for doc in self.corpus]

        raw_scores = []
        doc_matched_terms = []

        for idx, doc_toks in enumerate(self.doc_tokens):
            score = 0.0
            matched = []
            doc_len = self.doc_lengths[idx]
            
            # นับความถี่ของคำในเอกสารนี้
            term_freqs: Dict[str, int] = {}
            for t in doc_toks:
                term_freqs[t] = term_freqs.get(t, 0) + 1

            for qt in q_tokens:
                if qt in term_freqs:
                    matched.append(qt)
                    tf = term_freqs[qt]
                    idf_val = self.idf.get(qt, 0.5)
                    # สูตร Okapi BM25
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self.avg_doc_len)))
                    term_score = idf_val * (tf * (self.k1 + 1.0)) / denom
                    score += term_score

            raw_scores.append(score)
            doc_matched_terms.append(list(set(matched)))

        # Normalize คะแนน BM25 เป็น [0, 1] สำหรับการทำ Hybrid Fusion
        max_score = max(raw_scores) if raw_scores else 0.0
        normalized_scores = []
        for s in raw_scores:
            if max_score > 0:
                normalized_scores.append(round(s / max_score, 4))
            else:
                normalized_scores.append(0.0)

        results = []
        for i in range(len(self.corpus)):
            results.append((normalized_scores[i], self.corpus[i], doc_matched_terms[i]))
        return results


class DenseRetriever:
    """
    ระบบค้นหาแบบ Dense / Semantic Vector Space:
    - แปลงเอกสารและข้อความ Query เป็น Dense Vector Representations
    - ใช้ Subword & Character N-gram TF-IDF Embedding Space (ทำงานออฟไลน์ได้ 100% เร็ว ไม่ต้องพึ่งโมเดลภายนอก)
    - รองรับการคำนวณ Cosine Similarity บน Normalized Vectors
    """
    def __init__(self, n_features: int = 1024):
        self.n_features = n_features
        self.corpus: List[Dict[str, Any]] = []
        self.doc_vectors: Optional[np.ndarray] = None
        self.vocab: Dict[str, int] = {}
        self.idf_vector: Optional[np.ndarray] = None

    def _extract_ngrams(self, text: str) -> List[str]:
        """สกัดทั้ง Words และ Character 3-4 Grams เพื่อจับความหมายและรากคำ"""
        tokens = tokenize_text(text)
        features = list(tokens)
        
        # เพิ่ม subword character n-grams (3-gram, 4-gram)
        clean = re.sub(r'\s+', '', text.lower())
        for n in (3, 4):
            if len(clean) >= n:
                for i in range(len(clean) - n + 1):
                    features.append(clean[i:i+n])
        return features

    def index(self, documents: List[Dict[str, Any]]) -> None:
        """สร้าง Dense Vector Index จากชุดเอกสาร"""
        self.corpus = documents
        if not documents:
            self.doc_vectors = None
            return

        # 1. รวบรวม Vocabulary จากเอกสารทั้งหมด
        doc_features_list = []
        feature_doc_freq: Dict[str, int] = {}

        for doc in documents:
            text = f"{doc.get('title', '')} {doc.get('category', '')} {' '.join(doc.get('keywords', []))} {doc.get('summary', '')} {doc.get('code', '')[:800]}"
            feats = self._extract_ngrams(text)
            doc_features_list.append(feats)
            
            for f in set(feats):
                feature_doc_freq[f] = feature_doc_freq.get(f, 0) + 1

        # คัดเลือก top features เข้า vocab
        sorted_features = sorted(feature_doc_freq.items(), key=lambda x: x[1], reverse=True)[:self.n_features]
        self.vocab = {feat: idx for idx, (feat, _) in enumerate(sorted_features)}

        # คำนวณ IDF vector
        n_docs = len(documents)
        self.idf_vector = np.zeros(len(self.vocab), dtype=np.float32)
        for feat, idx in self.vocab.items():
            df = feature_doc_freq.get(feat, 1)
            self.idf_vector[idx] = math.log((n_docs + 1) / (df + 1)) + 1.0

        # สร้าง Document Vectors (TF-IDF + L2 Unit Normalization)
        matrix = np.zeros((n_docs, len(self.vocab)), dtype=np.float32)
        for doc_idx, feats in enumerate(doc_features_list):
            for f in feats:
                if f in self.vocab:
                    feat_idx = self.vocab[f]
                    matrix[doc_idx, feat_idx] += 1.0

        # คูณด้วย IDF
        matrix = matrix * self.idf_vector

        # Normalize แต่ละแถวให้มี L2 norm = 1
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.doc_vectors = matrix / norms

    def embed_query(self, query: str) -> np.ndarray:
        """แปลง Query เป็น Unit Vector"""
        if not self.vocab or self.idf_vector is None:
            return np.zeros((1, 1), dtype=np.float32)

        q_feats = self._extract_ngrams(query)
        vec = np.zeros((1, len(self.vocab)), dtype=np.float32)
        for f in q_feats:
            if f in self.vocab:
                vec[0, self.vocab[f]] += 1.0

        vec = vec * self.idf_vector
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def score(self, query: str) -> List[Tuple[float, Dict[str, Any]]]:
        """
        คำนวณ Cosine Similarity ระหว่าง Query Vector กับ Document Vectors
        คืนค่า: รายการของ tuple (cosine_similarity, document)
        """
        if self.doc_vectors is None or len(self.corpus) == 0:
            return []

        q_vec = self.embed_query(query)
        # Cosine similarity = Q . D^T (เมื่อทั้งสองเป็น Unit Vectors)
        sims = np.dot(self.doc_vectors, q_vec.T).flatten()
        # แปลงค่าเป็นช่วง [0, 1]
        sims = np.clip(sims, 0.0, 1.0)

        results = []
        for i, doc in enumerate(self.corpus):
            results.append((round(float(sims[i]), 4), doc))
        return results


class HybridRetriever:
    """
    ระบบ Hybrid RAG ที่ผสานการทำงานระหว่าง:
    - Sparse Lexical Search (BM25)
    - Dense Semantic Vector Search (Cosine Similarity)
    พร้อมกลยุทธ์การฟิวชัน 2 รูปแบบ:
    1. Weighted Score Combination: alpha * Dense + (1 - alpha) * BM25
    2. Reciprocal Rank Fusion (RRF): 1 / (k + rank_sparse) + 1 / (k + rank_dense)
    """
    def __init__(self, k_rrf: int = 60):
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()
        self.k_rrf = k_rrf
        self.documents: List[Dict[str, Any]] = []

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """สร้างดัชนีทั้ง BM25 และ Dense Vector พร้อมกัน"""
        self.documents = documents
        self.bm25.index(documents)
        self.dense.index(documents)

    def retrieve(
        self,
        query: str,
        top_k: int = 2,
        strategy: str = "weighted",  # "weighted", "rrf", "sparse_only", "dense_only"
        alpha: float = 0.5          # น้ำหนักของ Dense (0.0 = BM25 ล้วน, 1.0 = Semantic ล้วน)
    ) -> List[Dict[str, Any]]:
        """
        ค้นหาเอกสารแบบ Hybrid RAG และส่งคืนข้อมูลพร้อมคะแนนอย่างโปร่งใส
        """
        if not self.documents:
            return []

        # 1. รัน BM25 Sparse Search
        bm25_results = self.bm25.score(query)
        # 2. รัน Dense Semantic Search
        dense_results = self.dense.score(query)

        # สร้าง Dictionary แมป doc_id -> ข้อมูลคะแนน
        doc_map: Dict[str, Dict[str, Any]] = {}
        for doc in self.documents:
            doc_id = doc["id"]
            doc_map[doc_id] = {
                "doc": doc,
                "bm25_score": 0.0,
                "dense_score": 0.0,
                "matched_terms": [],
                "bm25_rank": len(self.documents),
                "dense_rank": len(self.documents)
            }

        # เติมคะแนน BM25
        # จัดเรียงเพื่อหาลำดับ BM25 Rank
        sorted_bm25 = sorted(bm25_results, key=lambda x: x[0], reverse=True)
        for rank, (score, doc, matched) in enumerate(sorted_bm25):
            d_id = doc["id"]
            doc_map[d_id]["bm25_score"] = score
            doc_map[d_id]["bm25_rank"] = rank + 1
            doc_map[d_id]["matched_terms"] = matched

        # เติมคะแนน Dense
        # จัดเรียงเพื่อหาลำดับ Dense Rank
        sorted_dense = sorted(dense_results, key=lambda x: x[0], reverse=True)
        for rank, (score, doc) in enumerate(sorted_dense):
            d_id = doc["id"]
            doc_map[d_id]["dense_score"] = score
            doc_map[d_id]["dense_rank"] = rank + 1

        # 3. รวมคะแนนตาม Strategy
        final_list = []
        for d_id, item in doc_map.items():
            bm25_s = item["bm25_score"]
            dense_s = item["dense_score"]
            bm25_r = item["bm25_rank"]
            dense_r = item["dense_rank"]

            if strategy == "sparse_only":
                final_score = bm25_s
                match_type = "🔤 BM25 Lexical"
            elif strategy == "dense_only":
                final_score = dense_s
                match_type = "🧠 Semantic Vector"
            elif strategy == "rrf":
                # Reciprocal Rank Fusion: 1/(k + rank_bm25) + 1/(k + rank_dense)
                rrf_bm25 = 1.0 / (self.k_rrf + bm25_r)
                rrf_dense = 1.0 / (self.k_rrf + dense_r)
                final_score = round(rrf_bm25 + rrf_dense, 6)
                if bm25_s > 0.1 and dense_s > 0.1:
                    match_type = "🔀 Hybrid RRF (Both)"
                elif bm25_s > dense_s:
                    match_type = "🔤 RRF (Keyword Dominant)"
                else:
                    match_type = "🧠 RRF (Semantic Dominant)"
            else:
                # Weighted Linear Combination: alpha * Dense + (1 - alpha) * BM25
                final_score = round((alpha * dense_s) + ((1.0 - alpha) * bm25_s), 4)
                if bm25_s > 0.1 and dense_s > 0.1:
                    match_type = "🔀 Hybrid Fusion"
                elif dense_s >= bm25_s:
                    match_type = "🧠 Semantic Dominant"
                else:
                    match_type = "🔤 Keyword Dominant"

            # สร้าง Output Object
            enriched_doc = dict(item["doc"])
            enriched_doc["hybrid_score"] = final_score
            enriched_doc["bm25_score"] = bm25_s
            enriched_doc["dense_score"] = dense_s
            enriched_doc["bm25_rank"] = bm25_r
            enriched_doc["dense_rank"] = dense_r
            enriched_doc["match_type"] = match_type
            enriched_doc["matched_terms"] = item["matched_terms"]
            enriched_doc["retrieval_strategy"] = strategy

            final_list.append(enriched_doc)

        # 4. จัดเรียงตามคะแนนรวมจากมากไปน้อย
        final_list.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # กรองเฉพาะรายการที่คะแนนมากกว่า 0 (ถ้ามี)
        valid_results = [d for d in final_list if d["hybrid_score"] > 0]
        if not valid_results and final_list:
            # กรณี query สั้นมาก ให้ส่งผลลัพธ์แรกสุด
            valid_results = final_list[:top_k]

        return valid_results[:top_k]


class QuestionRouter:
    """
    Router Pattern (ตามหลักสูตร Session 12):
    จำแนกประเภทคำถามของผู้ใช้เพื่อ Route ไปยัง Retriever ที่เหมาะสมที่สุด:
    - 'graph': คำถามเชิงโครงสร้าง, ลำดับการเรียก (Call Hierarchy), และความสัมพันธ์ข้ามโมดูล
    - 'vector': คำถามเชิงเนื้อหา, การอธิบายตรรกะ, สูตรคำนวณ, และอัลกอริทึม
    - 'hybrid': ผสานข้อมูลทั้งความหมายและโครงสร้างเข้าด้วยกัน
    """
    GRAPH_KEYWORDS = [
        "calls", "called by", "call", "who calls", "what calls", "which calls",
        "imports", "import", "imported by",
        "depends on", "dependency", "dependencies",
        "inherits", "inherited by",
        "impact", "impacts", "affected by", "affects",
        "defined in", "where is", "list functions in",
        "connections", "connected to", "relationship", "relationships",
        "structure", "architecture", "caller", "callee", "hierarchy",
        "graph", "graphs", "mermaid", "diagram", "flowchart",
        "โครงสร้าง", "เรียก", "ถูกเรียก", "นำเข้า", "ขึ้นกับ", "กระทบ", "ความสัมพันธ์", "สถาปัตยกรรม",
        "กราฟ", "แผนภาพ", "แผนผัง", "ไดอะแกรม"
    ]

    def classify(self, question: str) -> str:
        """จำแนกคำถามเป็น 'graph' หรือ 'vector'"""
        q_lower = question.lower()
        for kw in self.GRAPH_KEYWORDS:
            if kw in q_lower:
                return "graph"
        return "vector"

    def classify_with_reason(self, question: str) -> Dict[str, Any]:
        """จำแนกคำถามพร้อมระบุเหตุผลและคีย์เวิร์ดที่จับคู่ได้"""
        q_lower = question.lower()
        matched = []
        for kw in self.GRAPH_KEYWORDS:
            if kw in q_lower:
                matched.append(kw)

        if matched:
            return {
                "route": "graph",
                "mode_badge": "🔗 Graph Architecture Mode",
                "reason": f"จับพบคีย์เวิร์ดเชิงโครงสร้าง: {', '.join(matched)}",
                "matched_keywords": matched,
                "question": question
            }
        return {
            "route": "vector",
            "mode_badge": "🔍 Semantic Vector Mode",
            "reason": "คำถามเชิงเนื้อหา/ตรรกะ/อัลกอริทึม (Conceptual Vector Search)",
            "matched_keywords": [],
            "question": question
        }

