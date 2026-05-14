from pathlib import Path
import psycopg2
from psycopg2 import pool as pg_pool
from config.settings import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

_pool: pg_pool.SimpleConnectionPool | None = None


def get_pool() -> pg_pool.SimpleConnectionPool:
    global _pool
    if _pool is None:
        _pool = pg_pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
    return _pool


def get_connection():
    return get_pool().getconn()


def release_connection(conn) -> None:
    get_pool().putconn(conn)


def init_db() -> None:
    sql = (Path(__file__).parent / "migrations" / "001_init.sql").read_text()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    finally:
        release_connection(conn)
