import os
from dotenv import load_dotenv

from db.postgres_db import PostgresVectorDB
from db.sqlite_db import SQLiteVectorDB


def get_vector_db() -> PostgresVectorDB | SQLiteVectorDB:
    load_dotenv()
    db_type = os.getenv("DB_OPTION", "sqlite").lower()

    if db_type == "postgres":
        return PostgresVectorDB()
    
    return SQLiteVectorDB()