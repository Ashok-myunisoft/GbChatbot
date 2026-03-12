"""
db_query.py

Utility for querying the GoodBooks ERP MSSQL database directly.
Used by menu_bot, report_bot, formula_bot, project_bot, schema_bot.
"""

import os
import time
import logging
import re
import urllib.parse
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Module-level table list cache — avoids a DB round trip on every request
_table_cache: list = []
_table_cache_ts: float = 0.0
_TABLE_CACHE_TTL: int = 300  # seconds (5 minutes)

# MSSQL connection details from .env
MSSQL_HOST     = os.getenv("MSSQL_HOST", "192.168.0.112")
MSSQL_USER     = os.getenv("MSSQL_USER", "unisoft")
MSSQL_PASSWORD = os.getenv("MSSQL_PASSWORD", "unisoft@2012")
MSSQL_DATABASE = os.getenv("MSSQL_DATABASE", "BASICTEST")

# Singleton engine — created once, reused across all calls (enables connection pooling)
_engine = None


def _get_engine():
    """Return the shared SQLAlchemy engine for MSSQL via pymssql (singleton)."""
    global _engine
    if _engine is None:
        password = urllib.parse.quote_plus(MSSQL_PASSWORD)
        url = f"mssql+pymssql://{MSSQL_USER}:{password}@{MSSQL_HOST}/{MSSQL_DATABASE}"
        _engine = create_engine(
            url,
            connect_args={"timeout": 30},
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True  # test connection before using it from pool
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
    """Get column names for a table from MSSQL INFORMATION_SCHEMA."""
    try:
        engine = _get_engine()
        df = pd.read_sql(
            f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
            f"WHERE TABLE_NAME = '{table_name}' ORDER BY ORDINAL_POSITION",
            engine
        )
        return df["COLUMN_NAME"].tolist()
    except Exception as e:
        logger.warning(f"Could not get columns for table '{table_name}': {e}")
        return []


def _get_all_tables() -> list:
    """Return all user table names from MSSQL INFORMATION_SCHEMA (cached for TTL seconds)."""
    global _table_cache, _table_cache_ts
    now = time.time()
    if _table_cache and (now - _table_cache_ts) < _TABLE_CACHE_TTL:
        return _table_cache
    try:
        engine = _get_engine()
        df = pd.read_sql(
            "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
            "WHERE TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_NAME",
            engine
        )
        _table_cache = df["TABLE_NAME"].tolist()
        _table_cache_ts = now
        return _table_cache
    except Exception as e:
        logger.warning(f"Could not get table list: {e}")
        return _table_cache  # return stale cache on failure rather than empty list


def _detect_table_from_question(user_question: str) -> "str | None":
    """
    Find the most likely MSSQL table name mentioned in the user question.

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
    # Exclude words that are generic schema/query vocabulary or too short
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
        if len(w_up) < 5:                        # skip very short words
            continue
        if word.lower() in skip_words:           # skip generic schema words
            continue
        for tbl_up, tbl_orig in tables_upper.items():
            if w_up in tbl_up:
                partial_hits.append((len(tbl_up), tbl_orig))
            elif w_up.endswith('S') and w_up[:-1] in tbl_up:   # plural: employees→employee
                partial_hits.append((len(tbl_up), tbl_orig))

    if partial_hits:
        partial_hits.sort(key=lambda x: x[0])   # shortest table name = most specific
        return partial_hits[0][1]

    return None


def _generate_sql(llm, table_name: str, col_names: list, user_question: str, max_rows: int) -> str:
    """Ask the LLM to generate a T-SQL SELECT query for MSSQL."""
    if col_names:
        schema_info = f"Table: [{table_name}]\nColumns: {', '.join(col_names)}"
    else:
        # No specific table — LLM must infer the table from the user question
        schema_info = f"Database: {MSSQL_DATABASE} (Microsoft SQL Server)"

    sql_prompt = (
        f"You are a SQL expert for Microsoft SQL Server (T-SQL). Generate a single valid T-SQL SELECT query.\n\n"
        f"Schema info:\n{schema_info}\n\n"
        f"User question: {user_question}\n\n"
        f"Rules:\n"
        f"- Only use SELECT statements — no INSERT, UPDATE, DELETE, DROP, EXEC\n"
        f"- Use TOP {max_rows} instead of LIMIT\n"
        f"- Column selection: Read the user question carefully and select ONLY the columns needed to answer it.\n"
        f"  * If the user asks for specific fields (e.g. names, codes, dates, amounts) — select only those columns.\n"
        f"  * If the user asks for all data, all records, or does not specify columns — use SELECT *.\n"
        f"  * Examples: 'employee names and codes' → SELECT TOP N [EMPLOYEENAME],[EMPLOYECODE] FROM [...]\n"
        f"              'all employees' or 'all records' → SELECT TOP N * FROM [...]\n"
        f"              'show sales amount by date' → SELECT TOP N [SALEDATE],[AMOUNT] FROM [...]\n"
        f"- Row filtering: If the user asks about a specific item, keyword, or entity, add a WHERE clause to filter rows.\n"
        f"  * Extract the key search term from the question and filter using LIKE.\n"
        f"  * Examples: 'save endpoint for leave request' → WHERE CAST([WEBSERVICENAME] AS NVARCHAR(MAX)) LIKE '%leave%' OR CAST([URIITEMPLATE] AS NVARCHAR(MAX)) LIKE '%leave%'\n"
        f"              'employee named John' → WHERE CAST([EMPLOYEENAME] AS NVARCHAR(MAX)) LIKE '%John%'\n"
        f"              'reports in finance module' → WHERE CAST([REPORTNAME] AS NVARCHAR(MAX)) LIKE '%finance%'\n"
        f"  * Only fetch ALL rows without a WHERE clause if the user asks for everything with no specific filter.\n"
        f"- Use LIKE for text searches (MSSQL LIKE is case-insensitive by default)\n"
        f"- ALWAYS wrap every column with CAST([col] AS NVARCHAR(MAX)) when using LIKE or = with a string value — do this for ALL columns without exception, because any column could be tinyint, int, or bit even if its name looks like text (e.g. REPORTVIEWTYPE, STATUS, TYPE are often integers).\n"
        f"  * Correct:   WHERE CAST([REPORTVIEWTYPE] AS NVARCHAR(MAX)) LIKE '%leave%'\n"
        f"  * Incorrect: WHERE [REPORTVIEWTYPE] = 'LEAVE BALANCE'  ← never do this\n"
        f"- Use square brackets [ ] around table and column names\n"
        f"- NEVER use parameterized queries or placeholder variables (e.g. @SearchTerm, @param, ?, :param) — always embed literal values directly in the WHERE clause\n"
        f"  * Correct:   WHERE CAST([NAME] AS NVARCHAR(MAX)) LIKE '%leave%'\n"
        f"  * Incorrect: WHERE CAST([NAME] AS NVARCHAR(MAX)) LIKE '%' + @SearchTerm + '%'\n"
        f"- Return ONLY the raw SQL query — no explanation, no markdown, no code fences\n\n"
        f"SQL:"
    )
    raw = llm.invoke(sql_prompt)
    sql = raw.content if hasattr(raw, "content") else str(raw)
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
    # Convert LIMIT N → TOP N for MSSQL compatibility
    sql = _fix_mssql_syntax(sql)
    return sql


def _fix_mssql_syntax(sql: str) -> str:
    """
    Convert generic SQL to T-SQL syntax.
    Handles: LIMIT N  →  TOP N  (MSSQL does not support LIMIT).
    """
    limit_match = re.search(r'\bLIMIT\s+(\d+)\s*;?\s*$', sql.strip(), re.IGNORECASE)
    if limit_match:
        n = limit_match.group(1)
        # Remove LIMIT clause from end
        sql = sql[:limit_match.start()].strip().rstrip(';')
        # Insert TOP N after SELECT DISTINCT, or after plain SELECT
        if re.search(r'\bSELECT\s+DISTINCT\b', sql, re.IGNORECASE):
            sql = re.sub(
                r'\bSELECT\s+DISTINCT\s+',
                f'SELECT DISTINCT TOP {n} ',
                sql, count=1, flags=re.IGNORECASE
            )
        else:
            sql = re.sub(
                r'\bSELECT\s+',
                f'SELECT TOP {n} ',
                sql, count=1, flags=re.IGNORECASE
            )
        sql = sql.strip() + ';'
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
    """Build a fallback LIKE SQL query for MSSQL from keywords."""
    keywords = [w for w in search_term.split() if len(w) > 2] or [search_term]
    word_blocks = []
    for kw in keywords[:6]:
        safe_kw = kw.replace("'", "")
        col_conds = " OR ".join(
            f"CAST([{col}] AS NVARCHAR(MAX)) LIKE '%{safe_kw}%'"
            for col in col_names
        )
        word_blocks.append(f"({col_conds})")
    where_clause = " OR ".join(word_blocks)
    return f"SELECT TOP {max_rows} * FROM [{table_name}] WHERE {where_clause}"


def query_table(table_name: str, search_term: str, max_rows: int = 50) -> str:
    """
    Query *table_name* in MSSQL using Text-to-SQL (LLM-generated query).

    Strategy:
      1. LLM generates a T-SQL SELECT query from the user question + table schema.
      2. If LLM query fails or returns no rows, fall back to keyword LIKE search.
      3. If still no rows, return a TOP 30 sample so the LLM has schema context.

    Returns a formatted string ready to be injected into an LLM prompt.
    """
    try:
        # ── Get column names ───────────────────────────────────────────────────
        col_names = _get_columns(table_name)
        # col_names may be empty if table_name is a virtual name (e.g. "Unisoft")
        # In that case the LLM will infer the real table from the user question

        df = None

        # ── Step 1: Try Text-to-SQL ────────────────────────────────────────────
        llm = _get_llm()
        if llm:
            try:
                generated_sql = _generate_sql(llm, table_name, col_names, search_term, max_rows)
                logger.info(f"Text-to-SQL generated: {generated_sql[:200]}")

                # Safety: block any non-SELECT statements
                if not generated_sql.strip().upper().startswith("SELECT"):
                    raise ValueError("Non-SELECT query blocked")

                # Safety: reject suspiciously long queries (LLM hallucination with too many LIKE conditions)
                if len(generated_sql) > 5000:
                    raise ValueError(f"Generated SQL too long ({len(generated_sql)} chars) — likely hallucinated")

                engine = _get_engine()
                df = pd.read_sql(generated_sql, engine)

                if df.empty:
                    logger.info("Text-to-SQL returned 0 rows — falling back to LIKE")
                    df = None
                else:
                    logger.info(f"Text-to-SQL matched {len(df)} rows")
            except Exception as sql_exc:
                logger.warning(f"Text-to-SQL failed ({sql_exc}) — falling back to LIKE")
                df = None

        # ── Step 2: Fallback — keyword LIKE search ─────────────────────────────
        if df is None and col_names:
            try:
                fallback_sql = _like_sql(table_name, col_names, search_term, max_rows)
                engine = _get_engine()
                df = pd.read_sql(fallback_sql, engine)
            except Exception as like_exc:
                logger.warning(f"LIKE fallback failed ({like_exc})")
                df = None

        # ── Step 3: If still empty, return a broad sample ─────────────────────
        if df is None or df.empty:
            if col_names:
                try:
                    engine = _get_engine()
                    df = pd.read_sql(f"SELECT TOP 30 * FROM [{table_name}]", engine)
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
        logger.error(f"MSSQL query error on table '{table_name}': {exc}")
        return f"Error querying table '{table_name}': {exc}"
