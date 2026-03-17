"""
db_query.py

Utility for querying the GoodBooks ERP PostgreSQL database directly.
Used by menu_bot, report_bot, formula_bot, project_bot, schema_bot.
"""

import os
import time
import logging
import re
import pandas as pd
from sqlalchemy import create_engine, text as sa_text
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Module-level table list cache — avoids a DB round trip on every request
_table_cache: list = []
_table_cache_ts: float = 0.0
_TABLE_CACHE_TTL: int = 300  # seconds (5 minutes)

# PostgreSQL connection string from .env (falls back to hardcoded default)
PG_URL = os.getenv(
    "PG_URL",
    "postgresql://gbuser:aidev123@217.217.249.121:5432/IMPSYS_backup"
)
PG_DATABASE = os.getenv("PG_DATABASE", "IMPSYS_backup")

# Singleton engine — created once, reused across all calls (enables connection pooling)
_engine = None


def _get_engine():
    """Return the shared SQLAlchemy engine for PostgreSQL (singleton)."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            PG_URL,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True
        )
    return _engine


def _get_llm():
    """Lazy-load LLM to avoid circular imports at module level."""
    try:
        from shared_resources import ai_resources
        return ai_resources.response_llm
    except Exception:
        return None


def _get_columns(table_name: str) -> list:
    """Get column names for a table from PostgreSQL information_schema."""
    try:
        engine = _get_engine()
        query = sa_text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = :tname "
            "ORDER BY ordinal_position"
        )
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"tname": table_name.lower()})
        return df["column_name"].tolist()
    except Exception as e:
        logger.warning(f"Could not get columns for table '{table_name}': {e}")
        return []


def _get_all_tables() -> list:
    """Return all user table names from PostgreSQL information_schema (cached for TTL seconds)."""
    global _table_cache, _table_cache_ts
    now = time.time()
    if _table_cache and (now - _table_cache_ts) < _TABLE_CACHE_TTL:
        return _table_cache
    try:
        engine = _get_engine()
        query = sa_text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' "
            "ORDER BY table_name"
        )
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        _table_cache = df["table_name"].tolist()
        _table_cache_ts = now
        return _table_cache
    except Exception as e:
        logger.warning(f"Could not get table list: {e}")
        return _table_cache  # return stale cache on failure rather than empty list


def _detect_table_from_question(user_question: str) -> "str | None":
    """
    Find the most likely PostgreSQL table name mentioned in the user question.

    Strategy:
      1. Exact match — any word in the question exactly equals a table name (case-insensitive).
      2. Partial match — any word (≥5 chars, not a generic schema word) is contained
         inside a table name, OR its singular form is (e.g. "employees" → "MEMPLOYEE").
         Among partial matches prefer the shortest table name (most specific).

    Returns the real table name from the DB, or None if no match found.
    """
    all_tables = _get_all_tables()
    if not all_tables:
        return None

    tables_upper = {t.upper(): t for t in all_tables}   # UPPERCASE → original name
    words = re.findall(r'\b\w{3,}\b', user_question)
    words_sorted = sorted(set(words), key=len, reverse=True)  # longest word first

    # Pass 1 — exact match (any length)
    for word in words_sorted:
        if word.upper() in tables_upper:
            return tables_upper[word.upper()]

    # Pass 2 — partial match (meaningful words only)
    skip_words = {
        'table', 'tables', 'column', 'columns', 'record', 'records',
        'field', 'fields', 'value', 'values', 'database', 'schema',
        'describe', 'display', 'fetch', 'query', 'there', 'where',
        'which', 'data', 'from', 'show', 'list', 'give', 'find', 'what',
        'the', 'and', 'for', 'all', 'are', 'with', 'this', 'that'
    }
    partial_hits = []
    for word in words_sorted:
        w_up = word.upper()
        if len(w_up) < 5:
            continue
        if word.lower() in skip_words:
            continue
        for tbl_up, tbl_orig in tables_upper.items():
            if w_up in tbl_up:
                partial_hits.append((len(tbl_up), tbl_orig))
            elif w_up.endswith('S') and w_up[:-1] in tbl_up:
                partial_hits.append((len(tbl_up), tbl_orig))

    if partial_hits:
        partial_hits.sort(key=lambda x: (x[0], 0 if x[1].upper().startswith('M') else 1, x[1]))
        return partial_hits[0][1]

    return None


def _generate_sql(llm, table_name: str, col_names: list, user_question: str, max_rows: int) -> str:
    """Ask the LLM to generate a PostgreSQL SELECT query."""
    if col_names:
        schema_info = f"Table: {table_name}\nColumns: {', '.join(col_names)}"
    else:
        schema_info = f"Database: {PG_DATABASE} (PostgreSQL)"

    sql_prompt = (
        f"You are a SQL expert for PostgreSQL. Generate a single valid PostgreSQL SELECT query.\n\n"
        f"Schema info:\n{schema_info}\n\n"
        f"User question: {user_question}\n\n"
        f"Rules:\n"
        f"- Only use SELECT statements — no INSERT, UPDATE, DELETE, DROP\n"
        f"- Use LIMIT {max_rows} at the end of the query (PostgreSQL does NOT support TOP)\n"
        f"- IMPORTANT: All table and column names in this PostgreSQL database are LOWERCASE. Never use uppercase or quoted identifiers.\n"
        f"  * Correct:   SELECT employeename FROM memployee LIMIT N\n"
        f"  * Incorrect: SELECT \"EMPLOYEENAME\" FROM \"MEMPLOYEE\" LIMIT N\n"
        f"- Column selection: Read the user question carefully and select ONLY the columns needed to answer it.\n"
        f"  * RULE 1: If the user asks for 'all X names', 'all X codes', 'all X titles', 'all X values' — select ONLY that single name/code/title column. Do NOT use SELECT *.\n"
        f"    Examples: 'give me all employee names' → SELECT employeename FROM memployee LIMIT N\n"
        f"              'list all project names'     → SELECT projectname FROM mproject LIMIT N\n"
        f"              'show all report names'      → SELECT reportname FROM mreport LIMIT N\n"
        f"  * RULE 2: If the user asks for specific multiple fields — select only those columns.\n"
        f"    Example: 'employee names and codes' → SELECT employeename, employecode FROM memployee LIMIT N\n"
        f"  * RULE 3: If the user asks for all data, all records, all columns, or does not specify — use SELECT *.\n"
        f"    Example: 'get all data from memployee' → SELECT * FROM memployee LIMIT N\n"
        f"- Row filtering: If the user asks about a specific item, keyword, or entity, add a WHERE clause to filter rows.\n"
        f"  * Extract the key search term from the question and filter using ILIKE (case-insensitive).\n"
        f"  * Examples: 'employee named John' → WHERE CAST(employeename AS TEXT) ILIKE '%John%'\n"
        f"              'reports in finance module' → WHERE CAST(reportname AS TEXT) ILIKE '%finance%'\n"
        f"  * Only fetch ALL rows without a WHERE clause if the user asks for everything with no specific filter.\n"
        f"- Use ILIKE for case-insensitive text searches (PostgreSQL)\n"
        f"- ALWAYS wrap every column with CAST(col AS TEXT) when using ILIKE or = with a string value\n"
        f"  * Correct:   WHERE CAST(reportviewtype AS TEXT) ILIKE '%leave%'\n"
        f"  * Incorrect: WHERE reportviewtype = 'LEAVE BALANCE'\n"
        f"- Do NOT use double quotes around table or column names\n"
        f"- Use COALESCE instead of ISNULL\n"
        f"- Use NOW() instead of GETDATE()\n"
        f"- NEVER use parameterized queries or placeholder variables — always embed literal values directly\n"
        f"  * Correct:   WHERE CAST(\"NAME\" AS TEXT) ILIKE '%leave%'\n"
        f"  * Incorrect: WHERE CAST(\"NAME\" AS TEXT) ILIKE '%' || $1 || '%'\n"
        f"- Return ONLY the raw SQL query — no explanation, no markdown, no code fences\n\n"
        f"SQL:"
    )
    raw = llm.invoke(sql_prompt)
    sql = raw.content if hasattr(raw, "content") else str(raw)
    logger.info(f"LLM raw output for SQL generation: {repr(sql[:300])}")
    # Unwrap {'output': '...'} or {"output": "..."} format if LLM returned it wrapped
    if sql.strip().startswith("{") and ("'output'" in sql or '"output"' in sql):
        try:
            import ast
            parsed = ast.literal_eval(sql.strip())
            if isinstance(parsed, dict) and "output" in parsed:
                sql = str(parsed["output"])
        except Exception:
            pass
    # Strip markdown code fences if present
    sql = re.sub(r"```(?:sql)?", "", sql, flags=re.IGNORECASE).replace("```", "").strip()
    # Convert any T-SQL TOP N → LIMIT N for PostgreSQL compatibility
    sql = _fix_pg_syntax(sql)
    return sql


def _fix_pg_syntax(sql: str) -> str:
    """
    Convert T-SQL syntax to PostgreSQL syntax.
    Handles: SELECT TOP N → SELECT ... LIMIT N
    """
    # Convert TOP N to LIMIT N
    top_match = re.search(r'\bSELECT\s+(DISTINCT\s+)?TOP\s+(\d+)\s+', sql, re.IGNORECASE)
    if top_match:
        n = top_match.group(2)
        distinct = top_match.group(1) or ""
        # Remove TOP N from SELECT clause
        sql = re.sub(
            r'\bSELECT\s+(?:DISTINCT\s+)?TOP\s+\d+\s+',
            f'SELECT {distinct}',
            sql, count=1, flags=re.IGNORECASE
        ).strip()
        # Append LIMIT N (remove existing LIMIT first if any)
        sql = re.sub(r'\bLIMIT\s+\d+\s*;?\s*$', '', sql.strip(), flags=re.IGNORECASE).strip()
        sql = sql.rstrip(';') + f' LIMIT {n};'

    # Replace square bracket identifiers [col] → lowercase unquoted
    sql = re.sub(r'\[([^\]]+)\]', lambda m: m.group(1).lower(), sql)

    # Lowercase all double-quoted identifiers "TABLE" → table
    # PostgreSQL folds unquoted identifiers to lowercase; quoted uppercase names fail
    sql = re.sub(r'"([^"]+)"', lambda m: m.group(1).lower(), sql)

    # Replace ISNULL( with COALESCE(
    sql = re.sub(r'\bISNULL\s*\(', 'COALESCE(', sql, flags=re.IGNORECASE)

    # Replace GETDATE() with NOW()
    sql = re.sub(r'\bGETDATE\s*\(\s*\)', 'NOW()', sql, flags=re.IGNORECASE)

    # Replace NVARCHAR casts: CAST(x AS NVARCHAR(MAX)) → CAST(x AS TEXT)
    sql = re.sub(r'\bNVARCHAR\s*\(\s*MAX\s*\)', 'TEXT', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bNVARCHAR\s*\(\s*\d+\s*\)', 'TEXT', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bNVARCHAR\b', 'TEXT', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bVARCHAR\s*\(\s*MAX\s*\)', 'TEXT', sql, flags=re.IGNORECASE)

    return sql


def _format_df(df: pd.DataFrame) -> str:
    """
    Format a DataFrame for readable output regardless of column count.
    - ≤10 columns : standard table format (df.to_string)
    - >10 columns : one record per block with key: value pairs, skipping blank/null fields
    """
    if df.empty:
        return "(no rows)"
    num_cols = len(df.columns)
    if num_cols <= 10:
        return df.to_string(index=False)
    # Many-column path — vertical record blocks
    lines = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        lines.append(f"--- Record {i} ---")
        for col in df.columns:
            val = row[col]
            if val is not None and str(val).strip() not in ("", "None", "nan", "NaT"):
                lines.append(f"  {col}: {val}")
        lines.append("")
    return "\n".join(lines)


def _like_sql(table_name: str, col_names: list, search_term: str, max_rows: int) -> str:
    """Build a fallback ILIKE SQL query for PostgreSQL from keywords."""
    keywords = [w for w in search_term.split() if len(w) > 2] or [search_term]
    tbl = table_name.lower()
    word_blocks = []
    for kw in keywords[:6]:
        safe_kw = kw.replace("'", "")
        col_conds = " OR ".join(
            f"CAST({col.lower()} AS TEXT) ILIKE '%{safe_kw}%'"
            for col in col_names
        )
        word_blocks.append(f"({col_conds})")
    where_clause = " OR ".join(word_blocks)
    return f"SELECT * FROM {tbl} WHERE {where_clause} LIMIT {max_rows}"


def query_table(table_name: str, search_term: str, max_rows: int = 50) -> str:
    """
    Query *table_name* in PostgreSQL using Text-to-SQL (LLM-generated query).

    Strategy:
      1. LLM generates a PostgreSQL SELECT query from the user question + table schema.
      2. If LLM query fails or returns no rows, fall back to keyword ILIKE search.
      3. If still no rows, return a LIMIT 30 sample so the LLM has schema context.

    Returns a formatted string ready to be injected into an LLM prompt.
    """
    try:
        # ── Normalize table name to lowercase (PostgreSQL stores names in lowercase)
        table_name = table_name.lower()

        # ── Get column names ───────────────────────────────────────────────────
        col_names = _get_columns(table_name)

        df = None

        # ── Step 1: Try Text-to-SQL ────────────────────────────────────────────
        llm = _get_llm()
        if llm:
            try:
                generated_sql = _generate_sql(llm, table_name, col_names, search_term, max_rows)
                logger.info(f"Text-to-SQL generated: {generated_sql[:200]}")

                # Safety: block any non-SELECT statements
                if not generated_sql.strip().upper().startswith("SELECT"):
                    logger.warning(f"Non-SELECT query blocked. LLM output was: {repr(generated_sql[:300])}")
                    raise ValueError(f"Non-SELECT query blocked: {generated_sql[:100]}")

                # Safety: reject suspiciously long queries
                if len(generated_sql) > 5000:
                    raise ValueError(f"Generated SQL too long ({len(generated_sql)} chars) — likely hallucinated")

                engine = _get_engine()
                with engine.connect() as conn:
                    df = pd.read_sql(sa_text(generated_sql), conn)

                if df.empty:
                    logger.info("Text-to-SQL returned 0 rows — falling back to ILIKE")
                    df = None
                else:
                    logger.info(f"Text-to-SQL matched {len(df)} rows")
            except Exception as sql_exc:
                logger.warning(
                    f"Text-to-SQL failed [{type(sql_exc).__name__}]: {sql_exc} — falling back to ILIKE"
                )
                df = None

        # ── Step 2: Fallback — keyword ILIKE search ────────────────────────────
        if df is None and col_names:
            try:
                fallback_sql = _like_sql(table_name, col_names, search_term, max_rows)
                engine = _get_engine()
                with engine.connect() as conn:
                    df = pd.read_sql(sa_text(fallback_sql), conn)
            except Exception as like_exc:
                logger.warning(f"ILIKE fallback failed [{type(like_exc).__name__}]: {like_exc}")
                df = None

        # ── Step 3: If still empty, return a broad sample ─────────────────────
        if df is None or df.empty:
            if col_names:
                try:
                    engine = _get_engine()
                    with engine.connect() as conn:
                        df = pd.read_sql(sa_text(f'SELECT * FROM {table_name.lower()} LIMIT 30'), conn)
                    return (
                        f"No exact matches for '{search_term}' in table '{table_name}'.\n"
                        f"Available data overview ({len(df)} rows shown):\n\n"
                        + _format_df(df)
                    )
                except Exception:
                    pass
            return f"No data found for '{search_term}' in table '{table_name}'."

        return (
            f"Results for '{search_term}' from table '{table_name}' "
            f"({len(df)} matching rows):\n\n"
            + _format_df(df)
        )

    except Exception as exc:
        logger.error(f"PostgreSQL query error on table '{table_name}': {exc}")
        return f"Error querying table '{table_name}': {exc}"
