import os
from dotenv import load_dotenv

from db.postgres_db import PostgresVectorDB
from db.sqlite_db import SQLiteVectorDB


def get_vector_db() -> PostgresVectorDB | SQLiteVectorDB:
    load_dotenv()
    db_type = os.getenv("DB_OPTION", "sqlite").lower()
    db_path = os.getenv("SQLITE_DB_PATH")

    if db_type == "postgres":
        return PostgresVectorDB()
    
    if db_type == "sqlite" and db_path:
        return SQLiteVectorDB(db_path=db_path)

    return SQLiteVectorDB()
