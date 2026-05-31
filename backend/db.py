import logging
from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlmodel import Session

from config import settings


_log = logging.getLogger(__name__)


@event.listens_for(Engine, "connect")
def _enable_sqlite_fk(dbapi_connection, _) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


engine = create_engine(settings.database_url, echo=False)


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


# Additive column migrations applied after schema.sql runs. SQLite supports
# ADD COLUMN but not full DDL, so for now we only handle nullable columns.
# Each entry: (table, column_name, type_ddl). PRAGMA table_info gates the
# ALTER so this is idempotent on every startup.
_COLUMN_MIGRATIONS: list[tuple[str, str, str]] = [
    ("spaces", "is_home", "INTEGER NOT NULL DEFAULT 0"),
    ("command_corrections", "corrected_params_json", "TEXT"),
]


# Tables that existed in earlier revisions but no longer model anything alive.
# Dropped on every startup so the SQLite file converges to the current schema.
_DROPPED_TABLES: list[str] = ["labels"]


# Columns retired after the schema.sql shipped them. SQLite ≥3.35 supports
# ALTER TABLE ... DROP COLUMN; older versions silently swallow the error
# via the IF EXISTS guard around PRAGMA table_info.
_DROPPED_COLUMNS: list[tuple[str, str]] = [
    ("spaces", "description"),
]


def _ensure_column(conn, table: str, column: str, ddl: str) -> None:
    rows = conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
    existing = {r[1] for r in rows}
    if column in existing:
        return
    _log.info("migrating: ALTER TABLE %s ADD COLUMN %s %s", table, column, ddl)
    conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def _drop_column_if_present(conn, table: str, column: str) -> None:
    rows = conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
    existing = {r[1] for r in rows}
    if column not in existing:
        return
    _log.info("migrating: ALTER TABLE %s DROP COLUMN %s", table, column)
    try:
        conn.exec_driver_sql(f"ALTER TABLE {table} DROP COLUMN {column}")
    except Exception as e:  # noqa: BLE001
        # SQLite <3.35 doesn't support DROP COLUMN. Leaving the orphan
        # column is harmless because the SQLModel mapping no longer
        # references it.
        _log.warning("could not drop %s.%s (%s) — leaving as orphan", table, column, e)


async def init_db() -> None:
    schema_path = Path(__file__).parent / "schema.sql"
    schema_sql = schema_path.read_text(encoding="utf-8")
    with engine.begin() as conn:
        for statement in schema_sql.split(";"):
            statement = statement.strip()
            if statement:
                conn.exec_driver_sql(statement)
        for table, column, ddl in _COLUMN_MIGRATIONS:
            _ensure_column(conn, table, column, ddl)
        for table in _DROPPED_TABLES:
            conn.exec_driver_sql(f"DROP TABLE IF EXISTS {table}")
        for table, column in _DROPPED_COLUMNS:
            _drop_column_if_present(conn, table, column)
