"""
db_query.py

READ-ONLY SmolAgent-style PostgreSQL query layer.
✔ Text → SQL (RunPod SQL endpoint)
✔ Relationship-aware (JOIN support)
✔ Safe (SELECT only)
✔ Self-correcting SQL
✔ Single-file
"""

import os
import re
import time
import logging
import pandas as pd
from sqlalchemy import create_engine, text as sa_text
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# =========================================================
# ⚙️ DB CONFIG
# =========================================================
PG_URL = os.getenv(
    "PG_URL",
    "postgresql://gbuser:aidev123@host.docker.internal:5432/IMPSYS_backup"
)
PG_DATABASE = os.getenv("PG_DATABASE", "IMPSYS_backup")

_engine = None
_table_cache: list = []
_table_cache_ts: float = 0.0
_TABLE_CACHE_TTL: int = 300

# Column type + PK caches (1 hour TTL — schema rarely changes)
_col_type_cache: dict = {}
_col_type_cache_ts: dict = {}
_pk_cache: dict = {}
_pk_cache_ts: dict = {}
_SCHEMA_CACHE_TTL: int = 3600

# Query result cache — avoids sql_llm RunPod call for repeated questions
_query_cache: dict = {}
_QUERY_CACHE_TTL: int = 300  # 5 minutes


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(PG_URL, pool_size=5, max_overflow=10, pool_pre_ping=True)
    return _engine


# =========================================================
# 🗂️ TOOL 1 — SCHEMA
# =========================================================

def get_tables() -> list:
    global _table_cache, _table_cache_ts
    now = time.time()
    if _table_cache and (now - _table_cache_ts) < _TABLE_CACHE_TTL:
        return _table_cache
    try:
        q = sa_text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='public' AND table_type='BASE TABLE' "
            "ORDER BY table_name"
        )
        with _get_engine().connect() as conn:
            df = pd.read_sql(q, conn)
        _raw = df["table_name"].drop_duplicates().tolist()
        # Deduplicate partition tables: macloperation_2024, _2025 → keep first seen base
        _seen_bases: set = set()
        _deduped = []
        for t in _raw:
            base = re.sub(r'(_\d+)+$', '', t.lower())
            if base not in _seen_bases:
                _seen_bases.add(base)
                _deduped.append(t)
        _table_cache = _deduped
        _table_cache_ts = now  # noqa: F841
        return _table_cache
    except Exception as e:
        logger.warning(f"Could not get table list: {e}")
        return _table_cache


_col_cache: dict = {}
_col_cache_ts: dict = {}

def get_columns(table: str) -> list:
    key = table.lower()
    now = time.time()
    if key in _col_cache and (now - _col_cache_ts.get(key, 0)) < _SCHEMA_CACHE_TTL:
        return _col_cache[key]
    try:
        q = sa_text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name=:tname "
            "ORDER BY ordinal_position"
        )
        with _get_engine().connect() as conn:
            df = pd.read_sql(q, conn, params={"tname": key})
        result = df["column_name"].tolist()
        _col_cache[key] = result
        _col_cache_ts[key] = now
        return result
    except Exception as e:
        logger.warning(f"Could not get columns for '{table}': {e}")
        return []


def get_column_types(table: str) -> list:
    """Returns list of (column_name, pg_type) for a table. Cached 1 hour."""
    key = table.lower()
    now = time.time()
    if key in _col_type_cache and (now - _col_type_cache_ts.get(key, 0)) < _SCHEMA_CACHE_TTL:
        return _col_type_cache[key]
    try:
        q = sa_text("""
            SELECT column_name,
                   CASE
                     WHEN data_type = 'character varying'
                          THEN 'VARCHAR(' || COALESCE(character_maximum_length::text, '255') || ')'
                     WHEN data_type = 'character'
                          THEN 'CHAR(' || COALESCE(character_maximum_length::text, '1') || ')'
                     WHEN data_type IN ('integer','bigint','smallint','numeric',
                                        'boolean','text','date','uuid','json','jsonb')
                          THEN UPPER(data_type)
                     WHEN data_type = 'timestamp without time zone' THEN 'TIMESTAMP'
                     WHEN data_type = 'timestamp with time zone'    THEN 'TIMESTAMPTZ'
                     WHEN data_type = 'double precision'            THEN 'FLOAT'
                     ELSE data_type
                   END AS pg_type
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = :tbl
            ORDER BY ordinal_position
        """)
        with _get_engine().connect() as conn:
            df = pd.read_sql(q, conn, params={"tbl": key})
        result = list(zip(df["column_name"], df["pg_type"]))
        _col_type_cache[key] = result
        _col_type_cache_ts[key] = now
        return result
    except Exception as e:
        logger.warning(f"Could not get column types for '{table}': {e}")
        return [(c, "TEXT") for c in get_columns(table)]


def get_primary_key(table: str) -> list:
    """Returns primary key column name(s) for a table. Cached 1 hour."""
    key = table.lower()
    now = time.time()
    if key in _pk_cache and (now - _pk_cache_ts.get(key, 0)) < _SCHEMA_CACHE_TTL:
        return _pk_cache[key]
    try:
        q = sa_text("""
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
               AND tc.table_schema    = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
              AND tc.table_schema    = 'public'
              AND tc.table_name      = :tbl
            ORDER BY kcu.ordinal_position
        """)
        with _get_engine().connect() as conn:
            df = pd.read_sql(q, conn, params={"tbl": key})
        result = df["column_name"].tolist()
        _pk_cache[key] = result
        _pk_cache_ts[key] = now
        return result
    except Exception as e:
        logger.warning(f"Could not get PK for '{table}': {e}")
        return []


def _build_create_table(table: str, rels: list) -> str:
    """Build a compact CREATE TABLE DDL string for sqlcoder-7b-2.
    Limits columns to key ones (PK + FK + name/status/code) to stay within token limit.
    Tables with 30+ columns caused 'Schema too large' errors — this fixes that.
    """
    tbl = table.lower()
    col_types = get_column_types(tbl)
    pk_cols   = set(get_primary_key(tbl))

    # FK source columns for this table
    fk_cols = {r['source_column'] for r in rels if r['source_table'].lower() == tbl}

    # Important column name patterns — always keep these
    _KEY_SUFFIXES = ('id', 'name', 'code', 'status', 'type', 'date', 'description', 'expression', 'location')
    # Audit/system columns that add size but are rarely needed in queries
    # NOTE: formulaexpression, filelocation, remarks, purpose removed — they are queryable business data
    _DROP_COLS = {
        'createdon', 'modifiedon', 'createdbyid', 'modifiedbyid',
        'version', 'sourcetype', 'tenantid', 'entityid',
        'webserviceid', 'sortorder', 'addedtype',
        'reporturi',
    }

    essential = []   # PK + FK cols — always included
    important = []   # name/code/status cols — include if space
    rest      = []   # everything else — include only up to cap

    for col, typ in col_types:
        col_l = col.lower()
        if col_l in _DROP_COLS:
            continue
        entry = f"    {col} {typ}"
        if col_l in pk_cols or col_l in fk_cols:
            essential.append(entry)
        elif any(col_l.endswith(s) for s in _KEY_SUFFIXES):
            important.append(entry)
        else:
            rest.append(entry)

    # Cap total columns at 12 to prevent "Schema too large" errors
    _MAX_COLS = 12
    selected_cols = essential[:]
    remaining = _MAX_COLS - len(selected_cols)
    if remaining > 0:
        selected_cols += important[:remaining]
    remaining = _MAX_COLS - len(selected_cols)
    if remaining > 0:
        selected_cols += rest[:remaining]

    if pk_cols:
        selected_cols.append(f"    PRIMARY KEY ({', '.join(sorted(pk_cols))})")

    for r in rels:
        if r['source_table'].lower() == tbl and r['source_column'] in fk_cols:
            selected_cols.append(
                f"    FOREIGN KEY ({r['source_column']}) "
                f"REFERENCES {r['target_table'].lower()}({r['target_column']})"
            )

    return f"CREATE TABLE {tbl} (\n" + ",\n".join(selected_cols) + "\n);"


def get_relationships() -> list:
    try:
        q = sa_text("""
            SELECT
                tc.table_name  AS source_table,
                kcu.column_name AS source_column,
                ccu.table_name  AS target_table,
                ccu.column_name AS target_column
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
               AND tc.table_schema    = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
               AND ccu.table_schema    = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_schema     = 'public'
        """)
        with _get_engine().connect() as conn:
            df = pd.read_sql(q, conn)
        return df.to_dict(orient="records")
    except Exception as e:
        logger.warning(f"Could not get relationships: {e}")
        return []


def get_schema_tool(question: str = "", table_hint: str = "", max_tables: int = 2) -> str:
    """TOOL 1 — Returns tables + columns + FK relationships.
    question:   used for Schema RAG / keyword detection to find relevant tables.
    table_hint: table name forced by the calling bot (e.g. MREPORT, MFORMULA).
                Always included first, before RAG/keyword results.
    """
    tables = get_tables()
    rels = get_relationships()

    if question:
        # Pass 0 — Schema RAG (semantic search, faster + more accurate than keyword)
        # Warm guarantee: if index not ready yet, try loading from disk once (fast if exists)
        _rag_tables: list = []
        try:
            from schema_rag import is_index_ready, search_schema, build_or_load_index
            if not is_index_ready():
                try:
                    build_or_load_index()   # instant if index already on disk from prev run
                except Exception:
                    pass
            if is_index_ready():
                _rag_tables = search_schema(question, top_k=5)
        except Exception:
            pass

        # table_hint (from calling bot) always takes priority as primary table
        forced = table_hint.lower() if table_hint else ""

        # System table prefixes — never include these in generated SQL schema
        _SYS_PREFIXES = ('act_', 'qrtz_', 'conversation_', 'dump', 'bulk_',
                         'z_', 'tr_', 'aparna', 'temp', 'billtemp', 'chargedetail')

        if _rag_tables:
            # RAG found relevant tables — only outgoing FKs (source = rag table)
            # Avoids pulling in unrelated tables that merely reference the target
            primary_table = forced or _rag_tables[0].lower()
            related: set = set(t.lower() for t in _rag_tables)
            if forced:
                related.add(forced)
            for t in _rag_tables:
                for r in rels:
                    if r['source_table'].lower() == t.lower():
                        related.add(r['target_table'].lower())
        else:
            # Existing keyword logic (Pass 1/2/3) — completely unchanged
            target = _detect_table_from_question(question)
            primary_table = forced or (target.lower() if target else "")
            related: set = set()
            if forced:
                related.add(forced)
            if target:
                related.add(target.lower())
                for r in rels:
                    if r['source_table'].lower() == target.lower():
                        related.add(r['target_table'].lower())

        # Second-hop FK expansion — adds FK targets of FK targets (enables 3-table JOINs)
        # e.g. MMENU → MMODULE → MCATEGORY: MCATEGORY was previously never in DDL
        for hop2_src in list(related):
            for r in rels:
                if r['source_table'].lower() == hop2_src:
                    related.add(r['target_table'].lower())

        # Strip system/framework tables from related set — prevents bad JOINs + schema bloat
        related = {t for t in related if not t.startswith(_SYS_PREFIXES)}

        # Ensure primary table is always first before the :5 cap
        primary_list = [t for t in tables if t.lower() == primary_table]
        rest_related  = [t for t in tables if t.lower() in related and t.lower() != primary_table]
        others        = [t for t in tables if t.lower() not in related]
        priority      = primary_list + rest_related
        selected = (priority if _rag_tables else priority + others[:max(0, max_tables - len(priority))])[:max_tables]

        # BUILD CREATE TABLE DDL — the format sqlcoder-7b-2 was trained on
        # Includes real data types, PKs and FK constraints → accurate SQL generation
        ddl_blocks = [_build_create_table(t, rels) for t in selected]
        return "\n\n".join(ddl_blocks)
    else:
        ddl_blocks = [_build_create_table(t, rels) for t in tables[:5]]
        return "\n\n".join(ddl_blocks)


# ── System-table JOIN sanitizer ────────────────────────────────────────────────
_SQL_SYS_PREFIXES = ('act_', 'qrtz_', 'conversation_', 'dump', 'bulk_',
                     'z_', 'tr_', 'aparna', 'temp', 'billtemp', 'chargedetail')

def _strip_system_joins(sql: str) -> str:
    """Remove JOIN clauses where the joined table is a system/framework table.
    sqlcoder-7b-2 sometimes hallucinates Activiti/Quartz joins from training data
    even when those tables are not in the schema DDL.
    """
    lines = sql.split('\n')
    clean = []
    for line in lines:
        m = re.search(r'\bJOIN\s+(\w+)', line, re.IGNORECASE)
        if m and m.group(1).lower().startswith(_SQL_SYS_PREFIXES):
            logger.debug(f"Stripped hallucinated system JOIN: {line.strip()}")
            continue
        clean.append(line)
    result = '\n'.join(clean).strip()
    # Remove trailing ON clause orphaned by stripped JOIN (ends with ON or AND)
    result = re.sub(r'\s+(ON|AND)\s*$', '', result, flags=re.IGNORECASE).strip()
    return result


# =========================================================
# 🧠 TOOL 2 — GENERATE SQL
# =========================================================

def generate_sql_tool(question: str, schema: str, max_rows: int = 50) -> str:
    """TOOL 2 — Generate PostgreSQL SELECT from question + schema via RunPod."""
    from shared_resources import call_sql_endpoint

    # Send clean question — sqlcoder-7b-2 was trained on plain NL questions only.
    # Rules embedded in the question field are ignored by the model.
    # Schema is in CREATE TABLE format with real types — the model infers correct
    # types, PKs and JOIN conditions from the DDL directly.
    clean_question = f"{question} Limit results to {max_rows} rows."

    sql = call_sql_endpoint(query=clean_question, schema=schema)

    # Unwrap {'sql': '...'} or {'output': '...'} if endpoint returned wrapped format
    if sql.strip().startswith("{"):
        match = re.search(r"""['"](?:sql|output)['"]\s*:\s*['"](.+)""", sql.strip(), re.DOTALL)
        if match:
            raw_sql = match.group(1)
            raw_sql = re.sub(r"""['"]\s*\}?\s*$""", "", raw_sql)
            raw_sql = raw_sql.replace("\\n", "\n").replace("\\t", "\t").replace("\\'", "'").replace('\\"', '"')
            sql = raw_sql.strip()

    sql = re.sub(r"```(?:sql)?", "", sql, flags=re.IGNORECASE).replace("```", "").strip()
    sql = _fix_pg_syntax(sql)

    # Structural validation — reject broken patterns before execution
    if re.search(r'\bCASE\s+WHEN\s+COUNT\s*\(', sql, re.IGNORECASE):
        raise ValueError("LLM generated CASE WHEN COUNT(*) existence-check pattern — rejecting")
    if len(re.findall(r'\bSELECT\b', sql, re.IGNORECASE)) > 2:
        raise ValueError(f"LLM generated deeply nested subqueries — rejecting")
    if sql.count('(') != sql.count(')'):
        raise ValueError(
            f"Generated SQL has unbalanced parentheses "
            f"(open={sql.count('(')}, close={sql.count(')')}) — likely truncated by token limit"
        )
    return sql


# =========================================================
# 🔒 TOOL 3 — EXECUTE SQL (READ ONLY)
# =========================================================

def execute_sql_tool(sql: str) -> dict:
    """TOOL 3 — Execute SQL safely (SELECT only). Returns {'df': DataFrame} or {'error': str}."""
    sql_lower = sql.lower().strip()

    if not sql_lower.startswith("select"):
        return {"error": "Only SELECT queries are allowed"}

    if re.search(r'\b(insert|update|delete|drop|alter|truncate|create)\b', sql_lower):
        return {"error": "Write operations are not allowed"}

    try:
        with _get_engine().connect() as conn:
            # 10-second statement timeout — cancels runaway queries
            conn.execute(sa_text("SET LOCAL statement_timeout = '10000'"))
            df = pd.read_sql(sa_text(sql), conn)
        return {"df": df, "count": len(df)}
    except Exception as e:
        return {"error": str(e)}


# =========================================================
# 🔧 TOOL 4 — FIX SQL (SELF HEALING)
# =========================================================

def fix_sql_tool(sql: str, error: str, schema: str = "") -> str:
    """TOOL 4 — Fix broken SQL using error message + original schema via RunPod.
    Schema is pre-compacted to only tables referenced in the broken SQL.
    """
    from shared_resources import call_sql_endpoint

    # Use compact schema (only tables in broken SQL) to keep payload small
    compact = _compact_schema_for_fix(sql, schema) if schema else ""

    fix_query = (
        f"Fix this broken PostgreSQL SELECT query and return only the corrected SQL.\n"
        f"Error: {error}\n"
        f"Rules: Only SELECT, fix the syntax error, keep same meaning, all names lowercase.\n"
        f"Only use columns and tables that exist in the schema provided."
    )
    fix_schema = f"Available schema:\n{compact}\n\nBroken SQL to fix:\n{sql}"

    fixed = call_sql_endpoint(query=fix_query, schema=fix_schema)
    fixed = re.sub(r"```(?:sql)?", "", fixed, flags=re.IGNORECASE).replace("```", "").strip()
    return _fix_pg_syntax(fixed)


def _compact_schema_for_fix(sql: str, full_schema: str) -> str:
    """Return only the CREATE TABLE blocks for tables referenced in the broken SQL.
    Keeps the self-heal payload small so sqlcoder focuses on the right tables.
    """
    # Extract table names after FROM / JOIN keywords
    referenced = re.findall(r'\b(?:FROM|JOIN)\s+(\w+)', sql, re.IGNORECASE)
    tables_in_sql = {t.lower() for t in referenced if t}
    if not tables_in_sql:
        return full_schema

    blocks = full_schema.split('\n\n')
    relevant = []
    for block in blocks:
        m = re.match(r'CREATE\s+TABLE\s+(\w+)\s*\(', block.strip(), re.IGNORECASE)
        if m and m.group(1).lower() in tables_in_sql:
            relevant.append(block)
    return '\n\n'.join(relevant) if relevant else full_schema


def _local_fix_sql(sql: str, error: str) -> "str | None":
    """Instantly fix the most common SQL errors without a RunPod call.
    Returns corrected SQL, or None if this function cannot fix the error.
    Called BEFORE RunPod self-heal to save 5-28 seconds per fix.
    """
    err = error.lower()

    # ── Type mismatch: integer = text / invalid input syntax ───────────────
    # LLM compared a numeric column to a string literal.
    # Fix: strip the entire WHERE clause → return all rows.
    if "operator does not exist" in err or "invalid input syntax for type" in err:
        fixed = re.sub(
            r'\bWHERE\b.+?(?=\b(?:GROUP BY|ORDER BY|LIMIT|HAVING)\b|;?\s*$)',
            '', sql, flags=re.IGNORECASE | re.DOTALL
        ).strip().rstrip(';') + ';'
        if fixed.strip() != sql.strip():
            logger.info("[local-fix] stripped WHERE clause (type mismatch)")
            return _fix_pg_syntax(fixed)

    # ── Column does not exist ───────────────────────────────────────────────
    col_match = re.search(
        r'column "([^"]+)" (?:of relation "[^"]+"\s*)?does not exist',
        error, re.IGNORECASE
    )
    if col_match:
        bad_col = col_match.group(1).lower()
        # Try removing a JOIN clause that references this column
        join_removed = re.sub(
            rf'(?:LEFT\s+|RIGHT\s+|INNER\s+|OUTER\s+|FULL\s+)?'
            rf'JOIN\s+\S+\s*(?:\w+\s+)?ON\s+[^\n]*?{re.escape(bad_col)}[^\n]*',
            '', sql, flags=re.IGNORECASE
        ).strip()
        if join_removed.strip() != sql.strip():
            logger.info(f"[local-fix] removed JOIN with bad column '{bad_col}'")
            return _fix_pg_syntax(join_removed)
        # Try removing WHERE condition that contains this column
        where_removed = re.sub(
            rf'(?:\s+AND\s+|\s+OR\s+)?(?:\w+\.)?{re.escape(bad_col)}\s*[=<>!]+\s*\S+',
            '', sql, flags=re.IGNORECASE
        ).strip()
        if where_removed.strip() != sql.strip():
            logger.info(f"[local-fix] removed WHERE condition with bad column '{bad_col}'")
            return _fix_pg_syntax(where_removed)

    # ── Relation does not exist → can't fix locally, let fallback handle it ─
    # ── Syntax error → usually truncated → can't fix locally ────────────────
    return None


# =========================================================
# 🔧 SYNTAX FIX (T-SQL → PostgreSQL)
# =========================================================

def _fix_pg_syntax(sql: str) -> str:
    top_match = re.search(r'\bSELECT\s+(DISTINCT\s+)?TOP\s+(\d+)\s+', sql, re.IGNORECASE)
    if top_match:
        n = top_match.group(2)
        distinct = top_match.group(1) or ""
        sql = re.sub(r'\bSELECT\s+(?:DISTINCT\s+)?TOP\s+\d+\s+', f'SELECT {distinct}', sql, count=1, flags=re.IGNORECASE).strip()
        sql = re.sub(r'\bLIMIT\s+\d+\s*;?\s*$', '', sql.strip(), flags=re.IGNORECASE).strip()
        sql = sql.rstrip(';') + f' LIMIT {n};'
    sql = re.sub(r'\[([^\]]+)\]', lambda m: m.group(1).lower(), sql)
    sql = re.sub(r'"([^"]+)"', lambda m: m.group(1).lower(), sql)
    sql = re.sub(r'\bISNULL\s*\(', 'COALESCE(', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bGETDATE\s*\(\s*\)', 'NOW()', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bNVARCHAR\s*\(\s*MAX\s*\)', 'TEXT', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bNVARCHAR\s*\(\s*\d+\s*\)', 'TEXT', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bNVARCHAR\b', 'TEXT', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bVARCHAR\s*\(\s*MAX\s*\)', 'TEXT', sql, flags=re.IGNORECASE)
    return sql


# =========================================================
# 🔤 KEYWORD FALLBACK SQL (last resort)
# =========================================================

_STOP_WORDS = {
    "what", "which", "where", "when", "who", "how", "why",
    "are", "is", "was", "were", "has", "have", "had", "does", "did",
    "the", "for", "from", "with", "about", "give", "show", "list", "get",
    "all", "you", "your", "can", "tell", "find", "fetch", "display",
    "me", "my", "our", "its", "their", "this", "that", "these", "those",
    "any", "some", "more", "also", "please", "now", "do",
}

_LIST_INTENTS = {"list", "all", "give", "show", "get", "fetch", "display", "what", "which"}

def _build_fallback_sql(query: str, max_rows: int, table_hint: str = "") -> str:
    """Build a simple query when LLM + self-heal both fail.
    For 'list all / give / show' queries → SELECT * with no WHERE (user wants all records).
    For specific filter queries → ILIKE keyword search.
    table_hint: when provided by the calling bot, always use this table (avoids wrong detection).
    """
    table = table_hint or _detect_table_from_question(query)
    if not table:
        return ""
    tbl = table.lower()
    # Use cached get_column_types — avoids extra DB call, already fetched for schema
    col_type_pairs = get_column_types(table)
    col_names = [c for c, _ in col_type_pairs] if col_type_pairs else get_columns(table)
    if not col_names:
        return f"SELECT * FROM {tbl} LIMIT {max_rows}"

    # Detect listing intent — user wants all records, no filter needed
    q_words = set(query.lower().split())
    if q_words & _LIST_INTENTS and not any(
        w for w in query.split()
        if len(w) > 2 and w.lower() not in _STOP_WORDS and w.lower() not in _LIST_INTENTS
    ):
        return f"SELECT * FROM {tbl} LIMIT {max_rows}"

    # Specific filter — extract meaningful keywords
    keywords = [w for w in query.split() if len(w) > 2 and w.lower() not in _STOP_WORDS and w.lower() not in _LIST_INTENTS]
    if not keywords:
        return f"SELECT * FROM {tbl} LIMIT {max_rows}"

    word_blocks = []
    for kw in keywords[:6]:
        safe_kw = kw.replace("'", "")
        col_conds = " OR ".join(
            f"CAST({col.lower()} AS TEXT) ILIKE '%{safe_kw}%'"
            for col in col_names
        )
        if col_conds:
            word_blocks.append(f"({col_conds})")
    if not word_blocks:
        return f"SELECT * FROM {tbl} LIMIT {max_rows}"
    return f"SELECT * FROM {tbl} WHERE {' OR '.join(word_blocks)} LIMIT {max_rows}"


# =========================================================
# 📊 FORMATTING
# =========================================================

def _format_df(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    # Single column — return as clean numbered list (clearest for LLM)
    if len(df.columns) == 1:
        col = df.columns[0]
        values = [str(v) for v in df[col] if v is not None and str(v).strip() not in ("", "None", "nan", "NaT")]
        return "\n".join(f"{i + 1}. {v}" for i, v in enumerate(values))
    lines = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        lines.append(f"--- Record {i} ---")
        for col in df.columns:
            val = row[col]
            if val is not None and str(val).strip() not in ("", "None", "nan", "NaT"):
                lines.append(f"  {col}: {val}")
        lines.append("")
    return "\n".join(lines)


# =========================================================
# 🤖 AGENT ORCHESTRATION (RunPod — no local model needed)
# =========================================================

def run_db_agent(query: str, max_rows: int = 50, table_hint: str = "") -> str:
    """
    Orchestrates the 4 tools using RunPod SQL endpoint:
      Tool 1: get_schema_tool   → fetch tables + FK relationships
      Tool 2: generate_sql_tool → LLM generates JOIN-aware SELECT (retries once on cold-start timeout)
      Tool 3: execute_sql_tool  → run safely (SELECT only)
      Tool 4: fix_sql_tool      → self-heal on SQL error with schema context
      Tool 5: keyword fallback  → last resort ILIKE query if all above fail
    table_hint: optional table name passed by bots via query_table() to force correct table selection.
    """
    # Check query cache first — skip sql_llm entirely if same question asked recently
    global _query_cache
    _cache_key = f"{query.lower().strip()}:{max_rows}"
    _now = time.time()
    if _cache_key in _query_cache:
        _cached_result, _cached_ts = _query_cache[_cache_key]
        if _now - _cached_ts < _QUERY_CACHE_TTL:
            logger.info(f"Query cache hit — skipping RunPod sql_llm call")
            return _cached_result

    # Cache eviction — trim expired entries to prevent unbounded memory growth
    if len(_query_cache) > 50:
        expired_keys = [k for k, (_, ts) in _query_cache.items() if _now - ts >= _QUERY_CACHE_TTL]
        for k in expired_keys:
            del _query_cache[k]
        if expired_keys:
            logger.info(f"Cache eviction: removed {len(expired_keys)} expired entries")

    # Fast path — skip SQL LLM for simple listing queries (saves 5-28s per request)
    # Triggered when: "list all X", "give me X", "show X", "what are X" with no specific filter value
    _q_words = set(query.lower().split())
    _has_list_intent = bool(_q_words & _LIST_INTENTS)
    _filter_words = [
        w for w in query.split()
        if len(w) > 2
        and w.lower() not in _STOP_WORDS
        and w.lower() not in _LIST_INTENTS
        and not any(c.isdigit() for c in w)
    ]
    _target_table = table_hint or _detect_table_from_question(query)
    if _has_list_intent and not _filter_words and _target_table:
        _fast_sql = f"SELECT * FROM {_target_table.lower()} LIMIT {max_rows}"
        logger.info(f"Fast path (no LLM): {_fast_sql}")
        _fast_result = execute_sql_tool(_fast_sql)
        if "df" in _fast_result:
            _final = _format_df(_fast_result["df"]) if not _fast_result["df"].empty else "(no rows)"
            _query_cache[_cache_key] = (_final, _now)
            return _final

    try:
        # Tool 1 — Schema (query-aware: relevant tables first, table_hint forces correct table)
        # max_tables=3: safe now that each DDL is capped at 12 cols (~35 chars each = ~1260 chars total)
        # Retry ladder (schema-too-large path) still falls back to 2→1 tables as safety net
        schema = get_schema_tool(question=query, table_hint=table_hint, max_tables=3)

        # Tool 2 — Generate SQL
        # TimeoutError = cold start → retry once (worker will be warm)
        # RuntimeError = job FAILED (endpoint error) → skip straight to keyword fallback
        sql = None
        try:
            sql = generate_sql_tool(query, schema, max_rows)
            if sql:
                sql = _strip_system_joins(sql)
        except TimeoutError:
            logger.warning("SQL LLM timed out (cold start) — retrying once")
            try:
                sql = generate_sql_tool(query, schema, max_rows)
            except (TimeoutError, RuntimeError) as e:
                logger.warning(f"SQL generation failed after retry: {e} — using keyword fallback")
        except ValueError as e:
            # Truncated / too-complex SQL — retry with simplified 2-table schema
            if any(k in str(e).lower() for k in ("truncated", "unbalanced", "nested", "existence-check")):
                logger.warning(f"SQL complexity error: {e} — retrying with 2-table schema")
                try:
                    simple_schema = get_schema_tool(question=query, table_hint=table_hint, max_tables=2)
                    sql = generate_sql_tool(query, simple_schema, max_rows)
                    if sql:
                        sql = _strip_system_joins(sql)
                    logger.info("Retry with 2-table schema succeeded")
                except Exception as _e2:
                    logger.warning(f"2-table retry also failed: {_e2}")
            else:
                logger.warning(f"SQL validation error: {e} — using keyword fallback")
        except RuntimeError as e:
            if "schema too large" in str(e).lower():
                logger.warning(f"Schema too large — retrying with 2-table schema")
                try:
                    simple_schema = get_schema_tool(question=query, table_hint=table_hint, max_tables=2)
                    sql = generate_sql_tool(query, simple_schema, max_rows)
                    if sql:
                        sql = _strip_system_joins(sql)
                    logger.info("2-table retry succeeded after schema-too-large")
                except Exception as _e2:
                    if "schema too large" in str(_e2).lower():
                        logger.warning(f"2-table still too large — retrying with 1-table schema")
                        try:
                            single_schema = get_schema_tool(question=query, table_hint=table_hint, max_tables=1)
                            sql = generate_sql_tool(query, single_schema, max_rows)
                            logger.info("1-table retry succeeded after schema-too-large")
                        except Exception as _e3:
                            logger.warning(f"1-table retry also failed: {_e3} — using keyword fallback")
                    else:
                        logger.warning(f"2-table retry also failed: {_e2} — using keyword fallback")
            else:
                logger.warning(f"SQL endpoint job failed: {e} — using keyword fallback")

        if sql is None:
            # sql endpoint unavailable — go straight to keyword fallback
            fallback_sql = _build_fallback_sql(query, max_rows, table_hint)
            if fallback_sql:
                logger.info(f"Keyword fallback SQL (endpoint unavailable): {fallback_sql[:200]}")
                result = execute_sql_tool(fallback_sql)
            else:
                result = {"error": "SQL endpoint unavailable and no fallback table detected"}
        else:
            logger.info(f"Generated SQL: {sql[:200]}")

            if not sql.strip().upper().startswith("SELECT"):
                raise ValueError(f"Non-SELECT query blocked: {sql[:100]}")

            # Tool 3 — Execute
            result = execute_sql_tool(sql)

            # Tool 4 — Self-heal if error
            if "error" in result:
                logger.warning(f"SQL error: {result['error']} — attempting self-heal")
                fixed_sql = None

                # Pass 4a — local instant fix (no RunPod call, saves 5-28s)
                fixed_sql = _local_fix_sql(sql, result["error"])
                if fixed_sql:
                    logger.info(f"[local-fix] Fixed SQL: {fixed_sql[:200]}")
                    local_result = execute_sql_tool(fixed_sql)
                    if "df" in local_result:
                        result = local_result
                        fixed_sql = None  # mark as resolved — skip RunPod fix
                    else:
                        logger.warning(f"Local fix failed too: {local_result.get('error')} — escalating to RunPod")
                        fixed_sql = None  # let RunPod try below

                # Pass 4b — RunPod self-heal (schema compacted to only referenced tables)
                if "error" in result:
                    try:
                        fixed_sql = fix_sql_tool(sql, result["error"], schema)
                    except TimeoutError:
                        logger.warning("Self-heal LLM timed out — retrying once")
                        try:
                            fixed_sql = fix_sql_tool(sql, result["error"], schema)
                        except (TimeoutError, RuntimeError) as e:
                            logger.warning(f"Self-heal failed after retry: {e}")
                    except RuntimeError as e:
                        logger.warning(f"Self-heal endpoint job failed: {e}")

                    if fixed_sql:
                        logger.info(f"Fixed SQL: {fixed_sql[:200]}")
                        result = execute_sql_tool(fixed_sql)

            # Keyword fallback — last resort if LLM + self-heal both failed
            if "error" in result:
                logger.warning(f"SQL failed after self-heal: {result['error']} — trying keyword fallback")
                fallback_sql = _build_fallback_sql(query, max_rows, table_hint)
                if fallback_sql:
                    logger.info(f"Fallback SQL: {fallback_sql[:200]}")
                    result = execute_sql_tool(fallback_sql)

        if "df" in result and not result["df"].empty:
            df = result["df"]
            logger.info(f"Agent matched {len(df)} rows")
            final = (
                f"Results for '{query}' ({len(df)} rows):\n\n"
                + _format_df(df)
            )
            _query_cache[_cache_key] = (final, time.time())
            return final

        if "error" in result:
            logger.warning(f"All attempts failed: {result['error']}")
            # Detect relational failure from the SQL itself — no hardcoded keyword list
            # If sqlcoder generated a JOIN query but it still failed → multi-table question
            _was_join_attempt = sql is not None and bool(re.search(r'\bJOIN\b', sql, re.IGNORECASE))
            if _was_join_attempt:
                return (
                    f"No data found for: {query}\n"
                    f"Note: This question requires data from multiple tables. "
                    f"The system attempted a JOIN query but could not retrieve results. "
                    f"Try asking about a single table directly, e.g. 'list all menus' or 'show all reports'."
                )

        return f"No data found for: {query}"

    except Exception as e:
        logger.error(f"run_db_agent error: {e}")
        return f"Error: {e}"


# =========================================================
# 🚀 PUBLIC API
# =========================================================

def query_table(table_name: str, search_term: str, max_rows: int = 50) -> str:
    """Compatibility wrapper — called by all bots. Delegates to run_db_agent.
    Passes table_name as a hint so Schema RAG always prioritises the correct table.
    """
    return run_db_agent(search_term, max_rows, table_hint=table_name)


# =========================================================
# 🔁 COMPATIBILITY ALIASES (used by schema_bot)
# =========================================================

def _get_columns(table_name: str) -> list:
    return get_columns(table_name)


def _get_all_tables() -> list:
    return get_tables()


def _detect_table_from_question(user_question: str) -> "str | None":
    """Find the most likely table name from the user question (3-pass)."""
    all_tables = get_tables()
    if not all_tables:
        return None

    tables_upper = {t.upper(): t for t in all_tables}
    words = re.findall(r'\b\w{3,}\b', user_question)
    words_sorted = sorted(set(words), key=len, reverse=True)

    # Pass 1 — exact single-word match (case-insensitive)
    for word in words_sorted:
        if word.upper() in tables_upper:
            return tables_upper[word.upper()]

    # Pass 2 — compound bigram match for multi-word table names
    # e.g. "purchase order" → "PURCHASEORDER" finds MPURCHASEORDER
    word_list = re.findall(r'\b\w+\b', user_question)
    compound_hits = []
    for i in range(len(word_list) - 1):
        compound = (word_list[i] + word_list[i + 1]).upper()
        for tbl_up, tbl_orig in tables_upper.items():
            if tbl_up == compound:
                compound_hits.append((0, len(tbl_up), tbl_orig))
            else:
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

    skip_words = {
        'table', 'tables', 'column', 'columns', 'record', 'records',
        'field', 'fields', 'value', 'values', 'database', 'schema',
        'describe', 'display', 'fetch', 'query', 'there', 'where',
        'which', 'data', 'from', 'show', 'list', 'give', 'find', 'what',
        'the', 'and', 'for', 'all', 'are', 'with', 'this', 'that',
        'have', 'will', 'been', 'they', 'them', 'then', 'your', 'does',
        'also', 'just', 'like', 'only', 'same', 'such', 'know', 'make',
        'take', 'used', 'very', 'many', 'much', 'most', 'more', 'each',
        'both', 'once', 'back', 'want', 'need', 'tell', 'look', 'done',
        'come', 'sure', 'true', 'else', 'when', 'here', 'than', 'well',
        'name', 'type', 'code', 'date', 'time', 'into', 'some', 'call',
        'said', 'says', 'work', 'were', 'year', 'gets', 'give', 'sort',
    }
    # Pass 3 — partial match (meaningful words only, any table prefix)
    partial_hits = []
    for word in words_sorted:
        w_up = word.upper()
        if len(w_up) < 4 or word.lower() in skip_words:
            continue
        for tbl_up, tbl_orig in tables_upper.items():
            match_word = None
            if w_up in tbl_up:
                match_word = w_up
            elif w_up.endswith('S') and w_up[:-1] in tbl_up:
                match_word = w_up[:-1]
            if match_word:
                if tbl_up.startswith(match_word):
                    quality = 0
                elif any(tbl_up[p:].startswith(match_word) for p in range(1, 4)):
                    quality = 1
                else:
                    quality = 2
                partial_hits.append((quality, len(tbl_up), tbl_orig))
                continue
            # Reverse check: table base name is a prefix of the question word
            # Handles compound column words like "employeename" → MEMPLOYEE,
            # "vendorname" → MVENDOR, "reportcode" → MREPORT
            # Strip up to 3 leading prefix chars (e.g. M, ML, FW) then check
            for p in range(min(3, len(tbl_up))):
                base = tbl_up[p:]
                if len(base) >= 5 and w_up.startswith(base) and len(w_up) > len(base):
                    partial_hits.append((1, len(tbl_up), tbl_orig))
                    break

    if partial_hits:
        # Core GoodBooks bot tables take priority over generic/system tables with same match quality
        # e.g. "menu" → MMENU wins over MENUM, "report" → MREPORT wins over MREPORTTYPE
        _CORE_TABLES = {'mmenu', 'mreport', 'mformulafield', 'mfile', 'mproject', 'mformula'}
        partial_hits.sort(key=lambda x: (
            x[0],                                          # quality: 0=prefix, 1=suffix, 2=other
            0 if x[2].lower() in _CORE_TABLES else 1,     # core tables first
            0 if x[2].upper().startswith('M') else 1,     # M-prefix tables before others
            x[1],                                          # shorter table name first
        ))
        return partial_hits[0][2]

    return None


# =========================================================
# 🧪 TEST
# =========================================================

if __name__ == "__main__":
    print(run_db_agent("get employee names with department names"))

