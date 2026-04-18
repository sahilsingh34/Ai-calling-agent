import csv
import sqlite3
from pathlib import Path
from typing import Any


DB_FILE = Path(__file__).resolve().with_name("leads.db")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS leads (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    phone       TEXT NOT NULL UNIQUE,
    package     TEXT DEFAULT 'General Enquiry',
    status      TEXT DEFAULT 'pending',
    outcome     TEXT DEFAULT '',
    duration    TEXT DEFAULT '',
    called_at   TEXT DEFAULT '',
    created_at  TEXT DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    lead_id     INTEGER,
    session_id  TEXT,
    stage       TEXT,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    score       INTEGER DEFAULT 0,
    is_gold     INTEGER DEFAULT 0,
    language    TEXT DEFAULT 'hinglish',
    created_at  TEXT DEFAULT (datetime('now','localtime')),
    FOREIGN KEY(lead_id) REFERENCES leads(id)
);
"""


VALID_STATUSES = {
    "pending",
    "dialing",
    "completed",
    "no-answer",
    "busy",
    "failed",
}


def normalize_phone(phone: str) -> str:
    digits = "".join(ch for ch in str(phone).strip() if ch.isdigit())
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    if len(digits) != 10:
        return str(phone).strip()
    return f"+91{digits}"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        for statement in CREATE_TABLE_SQL.split(";"):
            if statement.strip():
                conn.execute(statement)
        conn.commit()


def add_turn(lead_id: int | None, role: str, content: str, session_id: str = "", stage: str = "", language: str = "hinglish") -> None:
    init_db()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO turns (lead_id, role, content, session_id, stage, language) 
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (lead_id, role, content, session_id, stage, language),
        )
        conn.commit()


def get_gold_examples(language: str = "hinglish", stage: str = "") -> list[dict[str, Any]]:
    init_db()
    with get_conn() as conn:
        sql = "SELECT * FROM turns WHERE is_gold = 1 AND language = ?"
        params = [language]
        if stage:
            sql += " AND stage = ?"
            params.append(stage)
        sql += " LIMIT 3"
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def get_unscored_turns(limit: int = 50) -> list[dict[str, Any]]:
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM turns WHERE score = 0 AND is_gold = 0 LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def score_turn(turn_id: int, score: int) -> None:
    init_db()
    with get_conn() as conn:
        conn.execute("UPDATE turns SET score = ? WHERE id = ?", (score, turn_id))
        conn.commit()


def get_lead_by_phone(phone: str) -> dict[str, Any] | None:
    init_db()
    normalized = normalize_phone(phone)
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM leads WHERE phone = ? OR phone = ?", 
            (normalized, normalized.replace("+91", ""))
        ).fetchone()
    return dict(row) if row else None


def add_lead(name: str, phone: str, package: str = "General Enquiry") -> int:
    init_db()
    clean_name = name.strip()
    clean_phone = normalize_phone(phone)
    clean_package = package.strip() or "General Enquiry"
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO leads (name, phone, package)
            VALUES (?, ?, ?)
            """,
            (clean_name, clean_phone, clean_package),
        )
        conn.commit()
        return cur.lastrowid if cur.rowcount else 0


def update_status(phone: str, status: str, outcome: str = "", duration: str = "") -> None:
    init_db()
    normalized = normalize_phone(phone)
    status_value = status if status in VALID_STATUSES else "failed"
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE leads
            SET status = ?,
                outcome = ?,
                duration = ?,
                called_at = datetime('now','localtime')
            WHERE phone = ? OR phone = ?
            """,
            (status_value, outcome, duration, normalized, normalized.replace("+91", "")),
        )
        conn.commit()


def get_pending(limit: int | None = None) -> list[dict[str, Any]]:
    init_db()
    sql = "SELECT * FROM leads WHERE status = 'pending' ORDER BY id ASC"
    params: tuple[Any, ...] = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def get_all() -> list[dict[str, Any]]:
    init_db()
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM leads ORDER BY id DESC").fetchall()
    return [dict(row) for row in rows]


def get_stats() -> dict[str, int]:
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS count FROM leads GROUP BY status"
        ).fetchall()
    counts = {row["status"]: row["count"] for row in rows}
    return {
        "total": sum(counts.values()),
        "pending": counts.get("pending", 0),
        "dialing": counts.get("dialing", 0),
        "completed": counts.get("completed", 0),
        "no_answer": counts.get("no-answer", 0),
        "busy": counts.get("busy", 0),
        "failed": counts.get("failed", 0),
    }


def import_from_csv(filepath: str) -> int:
    init_db()
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    imported = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row = {(key or "").strip().lower(): (value or "").strip() for key, value in raw_row.items()}
            name = row.get("name", "") or row.get("full name", "")
            phone = row.get("phone", "") or row.get("number", "") or row.get("mobile", "") or row.get("mobile number", "")
            package = row.get("package", "") or row.get("package interest", "") or "General Enquiry"
            if name and phone:
                if add_lead(name, phone, package):
                    imported += 1
    return imported


def create_sample_data() -> None:
    samples = [
        ("Riya Sharma", "9876543210", "Goa Summer Escape"),
        ("Aman Verma", "9988776655", "Dubai City Break"),
        ("Neha Kapoor", "9123456780", "Kashmir Family Tour"),
        ("Suresh Nair", "9765432109", "Kerala Backwaters"),
        ("Pooja Jain", "9812345670", "Thailand Holiday"),
    ]
    for name, phone, package in samples:
        add_lead(name, phone, package)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Trip in Minutes lead database helper")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("init", help="Create the SQLite database")
    subparsers.add_parser("sample", help="Insert sample leads")

    import_parser = subparsers.add_parser("import", help="Import leads from CSV")
    import_parser.add_argument("filepath")

    subparsers.add_parser("stats", help="Show lead counts")
    args = parser.parse_args()

    if args.command == "init":
        init_db()
        print(f"Database ready at {DB_FILE}")
    elif args.command == "sample":
        init_db()
        create_sample_data()
        print("Sample leads added.")
    elif args.command == "import":
        count = import_from_csv(args.filepath)
        print(f"Imported {count} leads.")
    else:
        print(json.dumps(get_stats(), indent=2))
