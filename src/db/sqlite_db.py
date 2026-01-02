import sqlite3
import numpy as np
import json
from typing import List, Tuple


class SQLiteVectorDB:
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.embeddings: List[np.ndarray] = []
        self.ids: List[int] = []
        self.create_table()

    def create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            metadata TEXT DEFAULT '{}'
        );
        """
        self.conn.execute(query)
        self.conn.commit()

    def insert_documents(self, data: List[Tuple[str, dict, list]]):
        """
        data: [(content, metadata, embedding)]
        """
        cursor = self.conn.cursor()

        for content, metadata, embedding in data:
            cursor.execute(
                "INSERT INTO documents (content, metadata) VALUES (?, ?)",
                (content, json.dumps(metadata)),
            )
            
            if cursor.lastrowid is not None:
                self.ids.append(cursor.lastrowid)

            self.embeddings.append(np.array(embedding, dtype=np.float32))

        self.conn.commit()

    def search(self, query_vector: List[float], limit: int = 5) -> list[tuple]:
        """
        - Row[0]: id
        - Row[1]: content
        - Row[2]: metadata
        - Row[3]: similarity
        """
        if not self.embeddings:
            return []

        query = np.array(query_vector, dtype=np.float32)

        similarities = np.array(
            [
                np.dot(query, emb) / (np.linalg.norm(query) * np.linalg.norm(emb))
                for emb in self.embeddings
            ]
        )

        top_k_idx = similarities.argsort()[-limit:][::-1]

        results = []
        for idx in top_k_idx:
            doc_id = self.ids[idx]
            row = self.conn.execute(
                "SELECT id, content, metadata FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()

            if row:
                results.append(
                    (row[0], row[1], json.loads(row[2]), float(similarities[idx]))
                )

        return results

    def delete_all_documents(self):
        self.conn.execute("DELETE FROM documents;")
        self.conn.commit()
        self.embeddings.clear()
        self.ids.clear()

    def is_empty(self) -> bool:
        cursor = self.conn.execute("SELECT COUNT(*) FROM documents;")
        return cursor.fetchone()[0] == 0

    def close(self):
        self.conn.close()
