import os
from dotenv import load_dotenv

import psycopg2
from psycopg2.extras import execute_values

from db.vector_db import VectorDB


class PostgresVectorDB(VectorDB):
    def __init__(self):
        try:
            load_dotenv()
            db_name = os.getenv("DB_NAME", "semantic_search_db")
            db_user = os.getenv("DB_USER", "vector_admin")
            db_password = os.getenv("DB_PASSWORD", "vector_password")
            db_host = os.getenv("DB_HOST", "localhost")
            db_port = os.getenv("DB_PORT", "5433")

            self.conn = psycopg2.connect(dbname=db_name, user=db_user, password=db_password, host=db_host, port=db_port)
            self.conn.autocommit = False
            self.create_table()

        except Exception as e:
            raise Exception(f"Database connection failed: {e}")

    def create_table(self):
        create_table_query = """
        CREATE TABLE IF NOT EXISTS embeddings.documents (
            id BIGSERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            metadata JSONB DEFAULT '{}',
            embedding vector(384),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS document_embedding_idx
        ON embeddings.documents
        USING hnsw (embedding vector_cosine_ops);
        """

        with self.conn.cursor() as cursor:
            cursor.execute(create_table_query)
            self.conn.commit()

    def insert_documents(self, data: list[tuple]):
        insert_query = """
        INSERT INTO embeddings.documents (content, metadata, embedding)
        VALUES %s
        """

        with self.conn.cursor() as cursor:
            execute_values(cursor, insert_query, data)
            self.conn.commit()

    def search(self, query_vector: list[float], limit: int = 5) -> list[tuple]:
        """
        - Row[0]: id
        - Row[1]: content
        - Row[2]: metadata
        - Row[3]: similarity
        """
        search_query = """
        SELECT id, content, metadata, 1 - (embedding <=> %s::vector) AS similarity
        FROM embeddings.documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """

        with self.conn.cursor() as cursor:
            cursor.execute(search_query, (query_vector, query_vector, limit))
            rows = cursor.fetchall()
            return rows

    def delete_all_documents(self):
        delete_query = "TRUNCATE TABLE embeddings.documents RESTART IDENTITY;"

        with self.conn.cursor() as cursor:
            cursor.execute(delete_query)
            self.conn.commit()

    def is_empty(self) -> bool:
        count_query = "SELECT COUNT(*) FROM embeddings.documents;"

        with self.conn.cursor() as cursor:
            cursor.execute(count_query)
            result = cursor.fetchone()
            if not result:
                return True
            return result[0] == 0

    def close(self):
        if self.conn:
            self.conn.close()