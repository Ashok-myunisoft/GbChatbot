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
    "postgresql://gbuser:aidev123@host.docker.internal:5432/IMPSYS_backup"
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
            "SELECT DISTINCT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' "
            "ORDER BY table_name"
        )
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        _table_cache = df["table_name"].drop_duplicates().tolist()
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

    # Pass 1 — exact single-word match (case-insensitive)
    for word in words_sorted:
        if word.upper() in tables_upper:
            return tables_upper[word.upper()]

    # Pass 2 — compound bigram match for multi-word table names
    # e.g. "purchase order" → "PURCHASEORDER" finds MPURCHASEORDER, PLPURCHASEORDER, etc.
    word_list = re.findall(r'\b\w+\b', user_question)
    compound_hits = []
    for i in range(len(word_list) - 1):
        compound = (word_list[i] + word_list[i + 1]).upper()
        for tbl_up, tbl_orig in tables_upper.items():
            if tbl_up == compound:
                compound_hits.append((0, len(tbl_up), tbl_orig))   # exact
            else:
                # compound appears right after a short prefix (1–3 chars: M, PL, FW, HR, GL…)
                for offset in range(1, 4):
                    if tbl_up[offset:] == compound:
                        compound_hits.append((offset, len(tbl_up), tbl_orig))
                        break
                    elif tbl_up[offset:].startswith(compound):
                        compound_hits.append((offset + 1, len(tbl_up), tbl_orig))
                        break
    if compound_hits:
        compound_hits.sort(key=lambda x: (x[0], x[1], x[2]))
        return compound_hits[0][2]

    # Pass 3 — partial match (meaningful words only, any table prefix)
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
            match_word = None
            if w_up in tbl_up:
                match_word = w_up
            elif w_up.endswith('S') and w_up[:-1] in tbl_up:
                match_word = w_up[:-1]
            if match_word:
                # Quality score — lower is better, prefix-agnostic:
                # 0 = word matches from position 0 of table name  (SALE → SALESORDER)
                # 1 = word matches from position 1–3 (after any short prefix: M, PL, FW…)
                # 2 = word appears anywhere else inside the table name
                if tbl_up.startswith(match_word):
                    quality = 0
                elif any(tbl_up[p:].startswith(match_word) for p in range(1, 4)):
                    quality = 1
                else:
                    quality = 2
                partial_hits.append((quality, len(tbl_up), tbl_orig))

    if partial_hits:
        partial_hits.sort(key=lambda x: (x[0], x[1], x[2]))
        return partial_hits[0][2]

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
        f"    Examples: 'give me all employee names' → SELECT employeename FROM memployee LIMIT {max_rows}\n"
        f"              'list all project names'     → SELECT projectname FROM mproject LIMIT {max_rows}\n"
        f"              'show all report names'      → SELECT reportname FROM mreport LIMIT {max_rows}\n"
        f"  * RULE 2: If the user asks for specific multiple fields — select only those columns.\n"
        f"    Example: 'employee names and codes' → SELECT employeename, employecode FROM memployee LIMIT {max_rows}\n"
        f"  * RULE 3: If the user asks for all data, all records, all columns, or does not specify — use SELECT *.\n"
        f"    Example: 'get all data from memployee' → SELECT * FROM memployee LIMIT {max_rows}\n"
        f"- RULE 4: If the user asks 'show me', 'get all', 'list', 'give me', 'can I get', 'display' without a specific filter value — do NOT add a WHERE clause. Just SELECT the relevant column(s) with LIMIT.\n"
        f"  * Examples: 'show me the reportname'  → SELECT reportname FROM mreport LIMIT {max_rows}\n"
        f"              'get all reports'          → SELECT reportname FROM mreport LIMIT {max_rows}\n"
        f"              'can I get reportname'     → SELECT reportname FROM mreport LIMIT {max_rows}\n"
        f"              'show me all menus'        → SELECT * FROM mmenu LIMIT {max_rows}\n"
        f"              'list all formulas'        → SELECT * FROM mformulafield LIMIT {max_rows}\n"
        f"- Row filtering: ONLY add a WHERE clause when the user provides a specific filter value (a name, keyword, ID, or category to search for).\n"
        f"  * Words like 'show', 'get', 'list', 'give', 'all', 'me', 'the' are NOT filter values — ignore them.\n"
        f"- EXACT NAME LOOKUP vs KEYWORD SEARCH — choose carefully:\n"
        f"  * EXACT MATCH: When the user asks for a specific named item (e.g. 'menucode for Program', 'what is the id of Sales', 'code of Finance')\n"
        f"    → Use LOWER(CAST(col AS TEXT)) = LOWER('exact_name')  — do NOT use ILIKE '%name%'\n"
        f"    Examples: 'menucode for Program'      → WHERE LOWER(CAST(menuname AS TEXT)) = LOWER('program')\n"
        f"              'what is the id of Sales'   → WHERE LOWER(CAST(modulename AS TEXT)) = LOWER('sales')\n"
        f"              'employee named John Smith' → WHERE LOWER(CAST(employeename AS TEXT)) = LOWER('john smith')\n"
        f"  * KEYWORD SEARCH: When the user searches broadly by category, module, or partial description\n"
        f"    → Use CAST(col AS TEXT) ILIKE '%keyword%'\n"
        f"    Examples: 'reports in finance module'   → WHERE CAST(reportname AS TEXT) ILIKE '%finance%'\n"
        f"              'menus related to purchase'   → WHERE CAST(menuname AS TEXT) ILIKE '%purchase%'\n"
        f"- ALWAYS wrap every column with CAST(col AS TEXT) when using ILIKE or = with a string value\n"
        f"  * Correct:   LOWER(CAST(menuname AS TEXT)) = LOWER('program')\n"
        f"  * Incorrect: menuname = 'Program'\n"
        f"- Do NOT use double quotes around table or column names\n"
        f"- Use COALESCE instead of ISNULL\n"
        f"- Use NOW() instead of GETDATE()\n"
        f"- NEVER use parameterized queries or placeholder variables — always embed literal values directly\n"
        f"  * Correct:   WHERE CAST(name AS TEXT) ILIKE '%leave%'\n"
        f"  * Incorrect: WHERE CAST(name AS TEXT) ILIKE '%' || $1 || '%'\n"
        f"- NEVER use CASE WHEN COUNT(*) — this is an existence-check pattern, NOT a data retrieval query\n"
        f"  * Incorrect: SELECT CASE WHEN COUNT(*) > 0 THEN 'true' ELSE 'false' END ...\n"
        f"  * Correct:   SELECT col FROM table WHERE condition LIMIT N\n"
        f"- NEVER use nested subqueries or CTEs for simple data retrieval — keep the query flat\n"
        f"  * Incorrect: SELECT * FROM (SELECT * FROM (SELECT * FROM table))\n"
        f"  * Correct:   SELECT col FROM table WHERE condition LIMIT N\n"
        f"- ALWAYS produce a simple flat SELECT: SELECT [columns] FROM [table] [WHERE ...] LIMIT N\n"
        f"- Return ONLY the raw SQL query — no explanation, no markdown, no code fences\n\n"
        f"SQL:"
    )
    raw = llm.invoke(sql_prompt)
    sql = raw.content if hasattr(raw, "content") else str(raw)
    logger.info(f"LLM raw output for SQL generation: {repr(sql[:300])}")
    # Unwrap {'output': '...'} or {"output": "..."} format if LLM returned it wrapped
    if sql.strip().startswith("{") and ("'output'" in sql or '"output"' in sql):
        match = re.search(r"""['"]output['"]\s*:\s*['"](.+)""", sql.strip(), re.DOTALL)
        if match:
            raw_sql = match.group(1)
            raw_sql = re.sub(r"""['"]\s*\}?\s*$""", "", raw_sql)
            raw_sql = raw_sql.replace("\\n", "\n").replace("\\t", "\t").replace("\\'", "'").replace('\\"', '"')
            sql = raw_sql.strip()
        else:
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

    # ── Structural validation — reject broken/invalid patterns immediately ────
    # Check 1: CASE WHEN COUNT(*) — existence-check pattern, not a data query
    if re.search(r'\bCASE\s+WHEN\s+COUNT\s*\(', sql, re.IGNORECASE):
        raise ValueError(f"LLM generated CASE WHEN COUNT(*) existence-check pattern — rejecting")

    # Check 2: Deeply nested SELECT subqueries (more than 1 level deep)
    inner_selects = len(re.findall(r'\bSELECT\b', sql, re.IGNORECASE))
    if inner_selects > 2:
        raise ValueError(f"LLM generated deeply nested subqueries ({inner_selects} SELECTs) — rejecting")

    # Check 3: Unbalanced parentheses (truncated mid-query by token limit)
    if sql.count('(') != sql.count(')'):
        raise ValueError(
            f"Generated SQL has unbalanced parentheses "
            f"(open={sql.count('(')}, close={sql.count(')')}) — likely truncated by token limit"
        )

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


_STOP_WORDS = {
    "what", "which", "where", "when", "who", "how", "why",
    "are", "is", "was", "were", "has", "have", "had", "does", "did",
    "the", "for", "from", "with", "about", "give", "show", "list", "get",
    "all", "you", "your", "can", "tell", "find", "fetch", "display",
    "me", "my", "our", "its", "their", "this", "that", "these", "those",
    "any", "some", "more", "also", "please", "now", "do",
}

def _like_sql(table_name: str, col_names: list, search_term: str, max_rows: int) -> str:
    """Build a fallback ILIKE SQL query for PostgreSQL from keywords."""
    raw_words = search_term.split()
    keywords = [w for w in raw_words if len(w) > 2 and w.lower() not in _STOP_WORDS]
    if not keywords:
        keywords = [w for w in raw_words if len(w) > 2] or [search_term]
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

        # ── Pre-check: if question is a pure "list all / show all" with no filter
        # bypass LLM SQL generation entirely and directly SELECT * — 100% reliable
        _list_all_patterns = re.compile(
            r'^\s*(show|list|get|give|fetch|display|can i get|what are)(\s+me|\s+all|\s+the)?\s+'
            r'(all\s+)?[\w\s]{1,30}$',
            re.IGNORECASE
        )
        _filter_words = re.compile(
            r'\b(where|filter|named|called|with|by|for|in|of|from|whose|having|like|equal|between|greater|less)\b',
            re.IGNORECASE
        )
        is_list_all_request = (
            _list_all_patterns.match(search_term.strip()) is not None
            and not _filter_words.search(search_term)
        )
        if is_list_all_request and col_names:
            logger.info(f"⚡ List-all detected — skipping LLM, running SELECT * directly")
            try:
                engine = _get_engine()
                with engine.connect() as conn:
                    df = pd.read_sql(sa_text(f'SELECT * FROM {table_name} LIMIT {max_rows}'), conn)
                if not df.empty:
                    return (
                        f"Results from table '{table_name}' ({len(df)} rows):\n\n"
                        + _format_df(df)
                    )
            except Exception as direct_exc:
                logger.warning(f"Direct SELECT failed: {direct_exc} — falling through to LLM")
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
                        f"[NOTE: No matching rows found for the query. Showing a general sample of {len(df)} rows from '{table_name}' for context only — this may NOT directly answer the user's question.]\n\n"
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
