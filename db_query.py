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

# Learning store — optional; all hooks wrapped in try/except
# Removing or missing learning_store.py disables learning silently
try:
    import learning_store as _ls
    _LEARNING_ENABLED = True
    logger.info("[Learning] Learning store enabled ✅")
except ImportError:
    _LEARNING_ENABLED = False

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
_QUERY_CACHE_TTL: int = 1800  # 30 minutes


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
            base = re.sub(r'_\d{4,}[a-z0-9]*$', '', base)
            base = re.sub(r'_[a-z]{2,5}\d{2,4}$', '', base)
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

    # Only declare FKs for columns that actually made it into selected_cols
    # Cap at 5 declarations — tables like MEMPLOYEE have 10+ FKs which bloat the DDL
    # and cause "Schema too large" errors on the RunPod sqlcoder endpoint
    _selected_col_names = {e.strip().split()[0].lower() for e in selected_cols}
    _fk_decl_count = 0
    for r in rels:
        if _fk_decl_count >= 5:
            break
        if r['source_table'].lower() == tbl and r['source_column'].lower() in _selected_col_names:
            selected_cols.append(
                f"    FOREIGN KEY ({r['source_column']}) "
                f"REFERENCES {r['target_table'].lower()}({r['target_column']})"
            )
            _fk_decl_count += 1

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
    _join_hint = " Use JOINs if the question asks for related data from multiple tables." if any(
        w in question.lower() for w in ["with their", "and their", "along with", "including", "together with"]
    ) else ""
    clean_question = f"{question} Limit results to {max_rows} rows.{_join_hint}"

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

# Audit/system columns stripped from fast-path SELECT and listings
_AUDIT_COLS = {
    'createdon', 'modifiedon', 'createdbyid', 'modifiedbyid',
    'version', 'sourcetype', 'tenantid', 'entityid',
    'webserviceid', 'sortorder', 'addedtype', 'reporturi',
}

# =========================================================
# SESSION CONTEXT STORE (follow-up question support)
# =========================================================

_session_df_store: dict = {}   # session_id -> (DataFrame, query_str, timestamp)
_SESSION_DF_TTL: int = 600     # 10 minutes

def _store_session_df(session_id: str, df, query: str):
    if session_id:
        _session_df_store[session_id] = (df, query, time.time())

def _get_session_df(session_id: str):
    entry = _session_df_store.get(session_id)
    if entry and time.time() - entry[2] < _SESSION_DF_TTL:
        return entry[0], entry[1]
    return None, None

_FOLLOWUP_INDICATORS = {
    "those", "that", "them", "these", "above", "previous",
    "first", "second", "third", "last", "listed",
}

def _is_followup_question(query: str) -> bool:
    """Returns True if the question likely refers to a previous result set."""
    words = set(re.sub(r"[^\w\s]", " ", query).lower().split())
    return bool(words & _FOLLOWUP_INDICATORS) and len(query.split()) <= 20

def _filter_df_by_question(query: str, df) -> "object | None":
    """
    Apply a simple filter/select from a follow-up question to a stored DataFrame.
    Returns filtered DataFrame or None if no applicable filter found.
    """
    import pandas as pd
    if df is None or df.empty:
        return None
    q_lower = re.sub(r"[^\w\s]", " ", query).lower()
    q_words = q_lower.split()
    cols_lower = {c.lower(): c for c in df.columns}

    # Ordinal access — "first", "second" ... "tenth", "last"
    ordinal_map = {
        "first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4,
        "sixth": 5, "seventh": 6, "eighth": 7, "ninth": 8, "tenth": 9,
        "last": -1, "latest": -1, "previous": -2,
    }
    for ord_word, idx in ordinal_map.items():
        if ord_word in q_words:
            try:
                row = df.iloc[[idx]].reset_index(drop=True)
                if not row.empty:
                    return row
            except IndexError:
                pass

    # Column+value filter: "which has status Active", "status 1"
    # Also handles fuzzy column name match: "module" → "modulename"
    filter_col = None
    filter_val = None
    for i, word in enumerate(q_words):
        # Exact column match
        if word in cols_lower:
            filter_col = cols_lower[word]
        else:
            # Fuzzy: question word is substring of a column name or vice-versa
            if len(word) >= 4:
                for cl, orig in cols_lower.items():
                    if word in cl or cl.startswith(word[:4]):
                        filter_col = orig
                        break
        if filter_col:
            for j in range(i + 1, min(i + 5, len(q_words))):
                if q_words[j] not in {"is", "are", "the", "be", "=", "a", "an", "of", "in"}:
                    filter_val = q_words[j]
                    break
            if filter_val:
                break
            filter_col = None  # reset if no value found after this col word

    if filter_col and filter_val:
        col_data = df[filter_col].astype(str).str.lower()
        filtered = df[col_data.str.contains(filter_val, na=False)]
        if not filtered.empty:
            return filtered.reset_index(drop=True)

    # Value-only scan: "active", "inactive", "enabled", "1", "0"
    scan_words = {
        "active", "inactive", "enabled", "disabled", "yes", "no",
        "true", "false", "open", "closed", "pending", "approved", "rejected",
    }
    for word in q_words:
        if word in scan_words:
            for col in df.columns:
                col_data = df[col].astype(str).str.lower()
                filtered = df[col_data.str.contains(word, na=False)]
                if not filtered.empty:
                    return filtered.reset_index(drop=True)

    return None


_STOP_WORDS = {
    "what", "which", "where", "when", "who", "how", "why",
    "are", "is", "was", "were", "has", "have", "had", "does", "did",
    "the", "for", "from", "with", "about", "give", "show", "list", "get",
    "all", "you", "your", "can", "tell", "find", "fetch", "display",
    "me", "my", "our", "its", "their", "this", "that", "these", "those",
    "any", "some", "more", "also", "please", "now", "do",
    # ERP table-noun plurals — treat as intent words, not filter values
    # e.g. "list all menus" → "menus" should NOT be a filter keyword
    "menus", "formulas", "reports", "modules", "employees", "records",
    "transactions", "invoices", "orders", "items", "projects", "files",
    "users", "roles", "groups", "departments", "designations", "categories",
    "programs", "screens", "components", "parameters", "templates",
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

    # Specific filter — extract meaningful keywords (strip punctuation so "Order?" → "Order")
    _clean_q2 = re.sub(r"[^\w\s]", " ", query)
    keywords = [w for w in _clean_q2.split() if len(w) > 2 and w.lower() not in _STOP_WORDS and w.lower() not in _LIST_INTENTS]
    if not keywords:
        return f"SELECT * FROM {tbl} LIMIT {max_rows}"

    # col_names_lower defined here — used by both "for" pattern and general keyword logic below
    col_names_lower = {c.lower() for c in col_names}

    # ── "what is X for Y" pattern ───────────────────────────────────────────
    # Split at " for " to cleanly separate column reference from filter value.
    # Uses len >= 2 for the column side — catches short abbreviations like URI,
    # SQL, ID that the general substring check (len > 4) would miss.
    #   "What is the report URI for Ledger Report?" →
    #       before: ["report","URI"] → SELECT reportname, reporturi
    #       after:  ["Ledger","Report"] → WHERE ILIKE '%Ledger%' AND ILIKE '%Report%'
    _for_pat = re.search(r'\bfor\b', _clean_q2, re.IGNORECASE)
    if _for_pat:
        _before_for = [
            w for w in _clean_q2[:_for_pat.start()].split()
            if len(w) >= 2 and w.lower() not in _STOP_WORDS and w.lower() not in _LIST_INTENTS
        ]
        _after_for = [
            w for w in _clean_q2[_for_pat.end():].split()
            if len(w) > 2 and w.lower() not in _STOP_WORDS and w.lower() not in _LIST_INTENTS
        ]
        if _before_for and _after_for:
            _pat_select: list = []
            for _pw in _before_for:
                _pwl = _pw.lower()
                if _pwl in col_names_lower:
                    _pat_select.append(_pwl)
                else:
                    _sub = [c for c in col_names if _pwl in c.lower()]
                    if _sub:
                        _pat_select.extend(_sub[:2])
            _pat_select = list(dict.fromkeys(_pat_select))
            if _pat_select:
                _pat_clause = ", ".join(_pat_select)
                _pat_blocks = []
                for _av in _after_for[:5]:
                    _safe = _av.replace("'", "")
                    _conds = " OR ".join(
                        f"CAST({c.lower()} AS TEXT) ILIKE '%{_safe}%'" for c in col_names
                    )
                    if _conds:
                        _pat_blocks.append(f"({_conds})")
                if _pat_blocks:
                    return (
                        f"SELECT {_pat_clause} FROM {tbl} "
                        f"WHERE {' AND '.join(_pat_blocks)} LIMIT {max_rows}"
                    )

    # Column-aware lookup: detect if any keyword IS a column name in this table.
    # Supports EXACT and SUBSTRING match:
    #   exact:     "menucode"   → "menucode"            ✓
    #   substring: "expression" → "formulaexpression"   ✓  (len > 4 avoids noise)
    # col_names_lower already defined above (shared with "for" pattern block)
    select_cols: list = []
    _col_matched_words: set = set()
    for _kw in keywords:
        _kwl = _kw.lower()
        if _kwl in col_names_lower:
            select_cols.append(_kwl)
            _col_matched_words.add(_kw)
        elif len(_kwl) > 4:
            _sub = [c for c in col_names if _kwl in c.lower()]
            if _sub:
                select_cols.extend(_sub[:2])
                _col_matched_words.add(_kw)
    select_cols = list(dict.fromkeys(select_cols))          # deduplicate, preserve order
    filter_kws  = [w for w in keywords if w not in _col_matched_words]

    if select_cols and filter_kws:
        select_clause = ", ".join(c.lower() for c in select_cols)
        word_blocks = []
        for kw in filter_kws[:5]:
            safe_kw = kw.replace("'", "")
            col_conds = " OR ".join(
                f"CAST({col.lower()} AS TEXT) ILIKE '%{safe_kw}%'"
                for col in col_names
            )
            if col_conds:
                word_blocks.append(f"({col_conds})")
        if word_blocks:
            # AND between filter keywords — "Purchase" AND "Order" won't match "Purchase Requisition"
            return f"SELECT {select_clause} FROM {tbl} WHERE {' AND '.join(word_blocks)} LIMIT {max_rows}"

    # Column-only listing — "list all menuname" → SELECT menuname FROM mmenu LIMIT n
    # No filter keywords needed — user just wants all values of that column
    if select_cols and not filter_kws:
        select_clause = ", ".join(c.lower() for c in select_cols)
        return f"SELECT {select_clause} FROM {tbl} LIMIT {max_rows}"

    # Generic filter — AND between keywords for precision (was OR — caused false positives)
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
    # AND: every keyword must match — prevents partial matches like "Purchase Requisition" for "Purchase Order"
    return f"SELECT * FROM {tbl} WHERE {' AND '.join(word_blocks)} LIMIT {max_rows}"


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
# 🔤 NAME VARIANT MATCHING (compound names like "Basic Salary")
# =========================================================

def _try_name_variants(phrase: str, target_table: str, select_cols: list,
                       all_cols: list, max_rows: int) -> "pd.DataFrame | None":
    """Try 5 phrase variants for compound names: 'Basic Salary' → 'BasicSalary', etc."""
    tbl = target_table.lower()
    clean = re.sub(r"[^\w\s]", "", phrase).strip()
    if not clean:
        return None
    no_space  = re.sub(r'\s+', '', clean)
    underscore = re.sub(r'\s+', '_', clean)
    sel_clause = ", ".join(c.lower() for c in select_cols) if select_cols else "*"

    for variant in [
        f"'%{clean}%'",
        f"'%{no_space}%'",
        f"'%{underscore}%'",
    ]:
        col_conds = " OR ".join(
            f"CAST({c.lower()} AS TEXT) ILIKE {variant}" for c in all_cols
        )
        if not col_conds:
            continue
        sql = f"SELECT {sel_clause} FROM {tbl} WHERE {col_conds} LIMIT {max_rows}"
        res = execute_sql_tool(sql)
        if "df" in res and not res["df"].empty:
            logger.info(f"Name variant matched with pattern {variant}")
            return res["df"]
    return None


# =========================================================
# 🔗 FK FILTER (e.g. "menus where module is Payroll")
# =========================================================

def _try_fk_filter(query: str, filter_kws: list, target_table: str,
                   rels: list, max_rows: int) -> "str | None":
    """
    When filter keywords don't match any column in target_table directly,
    look them up in FK-referenced tables and do a two-step query.
    Example: 'module=Payroll' → find moduleid in mmodule → filter mmenu by moduleid.
    """
    tbl = target_table.lower()
    target_cols = get_columns(target_table)
    if not target_cols:
        return None

    outgoing_fks = [
        (r['source_column'].lower(), r['target_table'].lower(), r['target_column'].lower())
        for r in rels if r['source_table'].lower() == tbl
    ]
    if not outgoing_fks:
        return None

    for kw in filter_kws:
        safe_kw = re.sub(r"[^\w\s]", "", kw).strip()
        if len(safe_kw) < 2:
            continue
        for fk_src_col, fk_tgt_table, fk_tgt_col in outgoing_fks:
            tgt_cols = get_columns(fk_tgt_table)
            if not tgt_cols:
                continue
            col_conds = " OR ".join(
                f"CAST({c.lower()} AS TEXT) ILIKE '%{safe_kw}%'" for c in tgt_cols
            )
            lookup_sql = (
                f"SELECT DISTINCT {fk_tgt_col} FROM {fk_tgt_table} "
                f"WHERE {col_conds} LIMIT 100"
            )
            lookup_res = execute_sql_tool(lookup_sql)
            if "error" in lookup_res or lookup_res["df"].empty:
                continue
            ids = [
                str(v) for v in lookup_res["df"][fk_tgt_col].tolist()
                if v is not None and str(v).strip() not in ("", "None", "nan")
            ]
            if not ids:
                continue
            ids_str = ", ".join(ids)
            sel_cols = [c for c in target_cols if c.lower() not in _AUDIT_COLS]
            sel_clause = ", ".join(sel_cols) if sel_cols else "*"
            main_sql = (
                f"SELECT {sel_clause} FROM {tbl} "
                f"WHERE {fk_src_col} IN ({ids_str}) LIMIT {max_rows}"
            )
            main_res = execute_sql_tool(main_sql)
            if "df" in main_res and not main_res["df"].empty:
                logger.info(
                    f"FK filter: '{kw}' found in {fk_tgt_table} → "
                    f"matched {len(main_res['df'])} rows in {tbl}"
                )
                return (
                    f"Results for '{query}' ({len(main_res['df'])} rows):\n\n"
                    + _format_df(main_res["df"])
                )
    return None


# =========================================================
# 🔀 MULTI-TABLE PANDAS MERGE (replaces GPT JOINs)
# =========================================================

def _try_multi_table_merge(query: str, primary_table: str,
                           rels: list, max_rows: int) -> "str | None":
    """
    Execute separate SELECT queries on primary + FK-related tables and merge
    in Python (pandas). Avoids unreliable GPT-generated multi-table JOIN SQL.
    Supports self-joins (e.g. MMENU.parentid → MMENU.menuid for parent menu name).
    """
    primary_lower = primary_table.lower()
    primary_cols = [c for c in get_columns(primary_table) if c.lower() not in _AUDIT_COLS]
    if not primary_cols:
        return None

    # Query the primary table
    p_sql = f"SELECT {', '.join(primary_cols)} FROM {primary_lower} LIMIT {max_rows}"
    p_res = execute_sql_tool(p_sql)
    if "error" in p_res or p_res["df"].empty:
        return None

    merged_df = p_res["df"].copy()

    # Find outgoing FKs from the primary table
    outgoing_fks = [
        (r['source_column'].lower(), r['target_table'].lower(), r['target_column'].lower())
        for r in rels if r['source_table'].lower() == primary_lower
    ]
    if not outgoing_fks:
        return None

    # Score FK relevance — fix: strip exactly ONE leading 'm' prefix (^m not ^m+)
    clean_q = re.sub(r"[^\w\s]", " ", query).lower()
    q_words = set(clean_q.split())

    def _fk_score(fk_tuple):
        src_col, tgt_table, tgt_col = fk_tuple
        base = re.sub(r'^m', '', tgt_table)            # mmodule→module, mmenu→menu (ONE m only)
        score = sum(2 for w in q_words if len(w) >= 4 and (w in base or base.startswith(w[:4])))
        score += sum(1 for w in q_words if len(w) >= 4 and w in tgt_col)
        score += sum(1 for w in q_words if len(w) >= 4 and w in src_col)
        return score

    ranked_fks = sorted(outgoing_fks, key=_fk_score, reverse=True)

    # Track which FK source ID cols were resolved — drop them from final output
    _resolved_id_cols: set = set()

    # Allow up to 3 joins when 3+ FKs are relevant to the question (score > 0)
    # e.g. "menus with module name and parent menu name" needs 2 FK joins (module + self-join)
    _relevant_count = sum(1 for fk in ranked_fks[:3] if _fk_score(fk) > 0)
    _max_joins = 3 if _relevant_count >= 3 else 2

    joins_done = 0
    for src_col, tgt_table, tgt_col in ranked_fks:
        if joins_done >= _max_joins:
            break
        if src_col not in merged_df.columns:
            continue

        # Self-join support: e.g. MMENU.parentid → MMENU.menuid (parent menu name)
        if tgt_table == primary_lower:
            self_cols = [c for c in get_columns(primary_table) if c.lower() not in _AUDIT_COLS]
            s_sql = f"SELECT {', '.join(self_cols)} FROM {primary_lower} LIMIT 5000"
            s_res = execute_sql_tool(s_sql)
            if "error" in s_res or s_res["df"].empty or tgt_col not in s_res["df"].columns:
                continue
            self_df = s_res["df"].copy()
            # Rename all columns with _parent suffix to avoid clash
            self_df = self_df.rename(columns={
                c: f"{c}_parent" for c in self_df.columns if c != tgt_col
            })
            try:
                merged_df = merged_df.merge(
                    self_df, left_on=src_col, right_on=tgt_col, how='left'
                )
                drop_cols = [c for c in merged_df.columns if c.endswith('_x') or c.endswith('_y')]
                if drop_cols:
                    merged_df = merged_df.drop(columns=drop_cols)
                _resolved_id_cols.add(src_col)
                joins_done += 1
                logger.info(f"Self-join: {primary_lower}.{src_col} → {primary_lower}.{tgt_col}")
            except Exception as _se:
                logger.warning(f"Self-join failed: {_se}")
            continue

        tgt_all_cols = [c for c in get_columns(tgt_table) if c.lower() not in _AUDIT_COLS]
        if not tgt_all_cols:
            continue

        t_sql = f"SELECT {', '.join(tgt_all_cols)} FROM {tgt_table} LIMIT 5000"
        t_res = execute_sql_tool(t_sql)
        if "error" in t_res or t_res["df"].empty:
            continue
        if tgt_col not in t_res["df"].columns:
            continue

        tgt_df = t_res["df"].copy()

        # Rename columns that clash with primary table (keep join key unchanged)
        rename_map = {
            c: f"{c}_{tgt_table}"
            for c in tgt_df.columns
            if c in merged_df.columns and c != tgt_col
        }
        if rename_map:
            tgt_df = tgt_df.rename(columns=rename_map)

        try:
            merged_df = merged_df.merge(tgt_df, left_on=src_col,
                                        right_on=tgt_col, how='left')
            drop_cols = [c for c in merged_df.columns if c.endswith('_x') or c.endswith('_y')]
            if drop_cols:
                merged_df = merged_df.drop(columns=drop_cols)
            _resolved_id_cols.add(src_col)
            joins_done += 1
            logger.info(f"Merged {primary_lower} ← {tgt_table} on {src_col}={tgt_col}")
        except Exception as _me:
            logger.warning(f"Pandas merge failed ({tgt_table}): {_me}")
            continue

    if joins_done == 0:
        return None  # No merge happened — let GPT try

    # Drop resolved FK ID columns — show names not raw IDs
    # e.g. after joining MMODULE, drop moduleid (keep modulename)
    # Only drop if a corresponding name/description col is now present
    _merged_cols_lower = {c.lower() for c in merged_df.columns}
    _drop_id_cols = []
    for id_col in _resolved_id_cols:
        base = re.sub(r'id$', '', id_col)  # moduleid → module, departmentid → department
        has_name = any(
            c for c in _merged_cols_lower
            if c.startswith(base) and not c.endswith('id') and c != id_col
        )
        if has_name and id_col in merged_df.columns:
            _drop_id_cols.append(id_col)
    if _drop_id_cols:
        merged_df = merged_df.drop(columns=_drop_id_cols)
        logger.info(f"Dropped resolved FK ID cols: {_drop_id_cols}")

    # Drop remaining raw negative-number ID columns (system IDs like -1899999989)
    for col in list(merged_df.columns):
        if col.lower().endswith('id') and col in merged_df.columns:
            try:
                sample = merged_df[col].dropna().head(3)
                if len(sample) > 0 and all(
                    isinstance(v, (int, float)) and float(v) < -1000000
                    for v in sample
                ):
                    merged_df = merged_df.drop(columns=[col])
            except Exception:
                pass

    logger.info(f"Multi-table merge: {joins_done} join(s), {len(merged_df)} rows, "
                f"{len(merged_df.columns)} cols")
    return (
        f"Results for '{query}' ({len(merged_df)} rows):\n\n"
        + _format_df(merged_df)
    )


# =========================================================
# 🔢 AGGREGATION ENGINE (SUM / COUNT / AVG / MAX / MIN)
# =========================================================

_AGG_KEYWORDS = {
    "SUM":   {"total", "sum", "gross total", "altogether"},
    "COUNT": {"count", "how many", "number of", "total number", "how much"},
    "AVG":   {"average", "avg", "mean"},
    "MAX":   {"maximum", "max", "highest", "largest", "most"},
    "MIN":   {"minimum", "min", "lowest", "smallest", "least"},
}
_AGG_COL_HINTS = {"amount", "total", "salary", "pay", "wage", "price", "cost",
                  "value", "balance", "gross", "net", "tax", "fee", "rate"}
# Columns that are numeric but should NEVER be aggregated (IDs, flags, counters)
_NON_METRIC_COLS = {
    "status", "version", "sortorder", "sort_order", "level", "levels",
    "sourcetype", "addedtype", "isparentmenu", "gcmtypeid", "tenantid",
    "entityid", "webserviceid",
}


def _try_aggregation_engine(query: str, target_table: str, max_rows: int,
                            rels: "list | None" = None) -> "str | None":
    """
    Detect aggregation intent (SUM/COUNT/AVG/MAX/MIN) from question keywords
    and build SQL dynamically from schema — no GPT or hardcoding.
    e.g. "total gross pay" → SELECT SUM(grosspay) FROM memployee
         "count of employees" → SELECT COUNT(*) FROM memployee
         "total salary for Finance" → SELECT SUM(salary) FROM ... WHERE dept IN (Finance IDs)
    """
    if not target_table:
        return None
    q_lower = query.lower()

    agg_func = None
    for func, triggers in _AGG_KEYWORDS.items():
        if any(t in q_lower for t in triggers):
            agg_func = func
            break
    if not agg_func:
        return None

    tbl = target_table.lower()

    # Build optional FK-based WHERE clause for filtered aggregation
    # e.g. "total salary for Finance department" → WHERE departmentid IN (<Finance IDs>)
    _where_clause = ""
    if rels:
        outgoing_fks = [
            (r['source_column'].lower(), r['target_table'].lower(), r['target_column'].lower())
            for r in rels if r['source_table'].lower() == tbl
        ]
        # Extract filter candidates: words not in stop words, not in agg triggers, len >= 3
        _agg_trigger_words = {w for triggers in _AGG_KEYWORDS.values() for w in triggers}
        _filter_candidates = [
            w for w in re.sub(r"[^\w\s]", " ", query).split()
            if len(w) >= 3
            and w.lower() not in _STOP_WORDS
            and w.lower() not in _LIST_INTENTS
            and w.lower() not in _agg_trigger_words
        ]
        _tbl_col_set = {c.lower() for c in get_columns(target_table)}
        # Only consider words that are NOT column names in the primary table (entity names)
        _entity_candidates = [w for w in _filter_candidates if w.lower() not in _tbl_col_set]
        for kw in _entity_candidates[:3]:
            safe_kw = re.sub(r"[^\w\s]", "", kw).strip()
            if len(safe_kw) < 3:
                continue
            for fk_src, fk_tgt_tbl, fk_tgt_col in outgoing_fks:
                tgt_cols = get_columns(fk_tgt_tbl)
                if not tgt_cols:
                    continue
                col_conds = " OR ".join(
                    f"CAST({c.lower()} AS TEXT) ILIKE '%{safe_kw}%'" for c in tgt_cols
                )
                lookup_sql = (
                    f"SELECT DISTINCT {fk_tgt_col} FROM {fk_tgt_tbl} "
                    f"WHERE {col_conds} LIMIT 50"
                )
                lk_res = execute_sql_tool(lookup_sql)
                if "error" in lk_res or lk_res["df"].empty:
                    continue
                ids = [
                    str(v) for v in lk_res["df"][fk_tgt_col].tolist()
                    if v is not None and str(v).strip() not in ("", "None", "nan")
                ]
                if ids:
                    _where_clause = f"WHERE {fk_src} IN ({', '.join(ids)})"
                    logger.info(f"Aggregation FK filter: '{kw}' → {fk_tgt_tbl}.{fk_tgt_col} IN {ids[:3]}")
                    break
            if _where_clause:
                break

    if agg_func == "COUNT":
        sql = f"SELECT COUNT(*) AS total_count FROM {tbl} {_where_clause}".strip()
        res = execute_sql_tool(sql)
        if "df" in res and not res["df"].empty:
            logger.info(f"Aggregation engine (COUNT): {sql}")
            return f"Results for '{query}':\n\n" + _format_df(res["df"])
        return None

    # SUM / AVG / MAX / MIN — pick best numeric column by name similarity
    col_types = get_column_types(tbl)
    _NUM_TYPES = {"integer", "bigint", "smallint", "numeric", "decimal",
                  "float", "double", "real", "money"}
    numeric_cols = [
        c for c, t in col_types
        if any(nt in t.lower() for nt in _NUM_TYPES)
        and c.lower() not in _AUDIT_COLS
        and c.lower() not in _NON_METRIC_COLS
        and not c.lower().endswith("id")    # FK/PK id cols are never metrics
        and not c.lower().endswith("type")  # type flag cols are never metrics
    ]
    if not numeric_cols:
        return None

    q_words = set(re.sub(r"[^\w\s]", " ", query).lower().split())

    def _col_score(col: str) -> int:
        cl = col.lower()
        score = sum(2 for w in q_words if len(w) >= 3 and (w in cl or cl in w))
        score += sum(1 for h in _AGG_COL_HINTS if h in cl)
        return score

    # Sort candidates by score — try top 3 in case best col has all NULLs
    scored = sorted(numeric_cols, key=_col_score, reverse=True)
    if not scored or _col_score(scored[0]) == 0:
        return None  # No column matches the question at all

    for best_col in scored[:3]:
        sql = f"SELECT {agg_func}({best_col}) AS result FROM {tbl} {_where_clause}".strip()
        res = execute_sql_tool(sql)
        if "df" in res and not res["df"].empty:
            val = res["df"].iloc[0, 0]
            if val is not None and str(val).strip() not in ("", "None", "nan", "0"):
                logger.info(f"Aggregation engine ({agg_func} on {best_col}): {sql}")
                return f"Results for '{query}':\n\n" + _format_df(res["df"])
    return None


# =========================================================
# 📊 REPORT ENGINE (GROUP BY aggregation)
# =========================================================

_GROUP_TRIGGERS = {
    " by ", " per ", "grouped by", "group by", "per each", "broken down",
    " each ", "breakdown", "distribution", "count per ", "total per ",
    "summary by", "count by", "categorize", "group wise", "wise ",
}


def _try_report_engine(query: str, target_table: str, rels: list, max_rows: int) -> "str | None":
    """
    Detect GROUP BY intent and build aggregation SQL dynamically.
    e.g. "reports by category"   → SELECT category, COUNT(*) GROUP BY category
         "sales per department"  → SELECT dept, SUM(amount) GROUP BY dept
    No hardcoding of column names — all derived from schema.
    """
    if not target_table:
        return None
    q_lower = query.lower()
    if not any(t in q_lower for t in _GROUP_TRIGGERS):
        return None

    tbl = target_table.lower()
    col_types = get_column_types(tbl)
    q_words = re.sub(r"[^\w\s]", " ", query).lower().split()

    def _name_score(col: str) -> int:
        cl = col.lower()
        score = 0
        for w in q_words:
            if len(w) >= 4:
                if w == cl or w in cl:
                    score += 2   # exact substring match — stronger signal
                elif cl.startswith(w[:4]):
                    score += 1   # prefix match — weaker signal
        return score

    # Detect agg function
    agg_func = "COUNT"
    agg_col = "*"
    for func, triggers in _AGG_KEYWORDS.items():
        if any(t in q_lower for t in triggers) and func != "COUNT":
            agg_func = func
            break

    _NUM_TYPES = {"integer", "bigint", "smallint", "numeric", "decimal",
                  "float", "double", "real", "money"}
    _TXT_TYPES = {"character", "varchar", "text", "char"}

    # Group cols: text columns only, excluding ID/audit/flag columns
    text_cols = [
        c for c, t in col_types
        if any(tt in t.lower() for tt in _TXT_TYPES)
        and c.lower() not in _AUDIT_COLS
        and not c.lower().endswith("id")    # IDs are never group labels
        and not c.lower().endswith("type")  # type flags are not meaningful groups
    ]
    num_cols = [
        c for c, t in col_types
        if any(nt in t.lower() for nt in _NUM_TYPES)
        and c.lower() not in _AUDIT_COLS
        and c.lower() not in _NON_METRIC_COLS
        and not c.lower().endswith("id")
        and not c.lower().endswith("type")
    ]

    if not text_cols:
        return None

    # Pick group column — text col with best name match to query
    group_col = max(text_cols, key=_name_score)
    if _name_score(group_col) == 0:
        return None  # No column matches question words — give up

    # If grouping column might be in a related table (FK), resolve via join
    # e.g. "reports by category" — categoryname is in MREPORTCATEGORY
    _fk_group_col = None
    _fk_group_table = None
    _fk_src_col = None
    _fk_tgt_col = None  # explicit save to avoid loop-scope ambiguity
    outgoing_fks = [
        (r['source_column'].lower(), r['target_table'].lower(), r['target_column'].lower())
        for r in rels if r['source_table'].lower() == tbl
    ]
    for src_col, tgt_tbl, tgt_col in outgoing_fks:
        tgt_col_types = get_column_types(tgt_tbl)
        tgt_text = [c for c, t in tgt_col_types
                    if any(tt in t.lower() for tt in _TXT_TYPES)
                    and c.lower() not in _AUDIT_COLS
                    and not c.lower().endswith("id")]
        if tgt_text:
            best_fk_col = max(tgt_text, key=_name_score)
            if _name_score(best_fk_col) > _name_score(group_col):
                _fk_group_col = best_fk_col
                _fk_group_table = tgt_tbl
                _fk_src_col = src_col
                _fk_tgt_col = tgt_col   # ← explicit save (fixes scope bug)
                break

    if agg_func != "COUNT" and num_cols:
        agg_col = max(num_cols, key=_name_score)
    else:
        agg_func = "COUNT"
        agg_col = "*"

    if _fk_group_table and _fk_tgt_col:
        sql = (
            f"SELECT {_fk_group_table}.{_fk_group_col}, "
            f"{agg_func}({agg_col}) AS result "
            f"FROM {tbl} "
            f"JOIN {_fk_group_table} "
            f"  ON {tbl}.{_fk_src_col} = {_fk_group_table}.{_fk_tgt_col} "
            f"GROUP BY {_fk_group_table}.{_fk_group_col} "
            f"ORDER BY result DESC LIMIT {max_rows}"
        )
    else:
        sql = (
            f"SELECT {group_col}, {agg_func}({agg_col}) AS result "
            f"FROM {tbl} GROUP BY {group_col} "
            f"ORDER BY result DESC LIMIT {max_rows}"
        )

    res = execute_sql_tool(sql)
    if "df" in res and not res["df"].empty:
        logger.info(f"Report engine ({agg_func} GROUP BY {group_col}): {sql}")
        return f"Results for '{query}' ({len(res['df'])} rows):\n\n" + _format_df(res["df"])

    # FK JOIN failed — retry with simple GROUP BY on primary table only
    if _fk_group_table:
        fallback_sql = (
            f"SELECT {group_col}, {agg_func}({agg_col}) AS result "
            f"FROM {tbl} GROUP BY {group_col} "
            f"ORDER BY result DESC LIMIT {max_rows}"
        )
        fb_res = execute_sql_tool(fallback_sql)
        if "df" in fb_res and not fb_res["df"].empty:
            logger.info(f"Report engine fallback (no FK JOIN): {fallback_sql}")
            return f"Results for '{query}' ({len(fb_res['df'])} rows):\n\n" + _format_df(fb_res["df"])
    return None


# =========================================================
# 🔍 RELATIONAL QUESTION DETECTOR
# =========================================================

def _is_relational_question(query: str, target_table: str, rels: list) -> bool:
    """
    Returns True if the question likely spans multiple tables.
    Detects entity words in the question that match FK-related table names.
    Works for ANY phrasing — not limited to "with their" / "and their".
    """
    q_words = set(re.sub(r"[^\w\s]", " ", query).lower().split())
    tbl = target_table.lower()

    # Classic trigger phrases (kept for fast-path)
    _PHRASE_TRIGGERS = {
        "with their", "and their", "along with", "including their",
        "together with", "with the", "and the", "show with", "list with",
    }
    if any(t in query.lower() for t in _PHRASE_TRIGGERS):
        return True

    # Entity detection: question words match FK-target table base names
    outgoing_fks = [r for r in rels if r['source_table'].lower() == tbl]
    for r in outgoing_fks:
        base = re.sub(r'^m', '', r['target_table'].lower())   # mmodule→module, mmenu→menu
        if len(base) >= 4 and any(
            w == base or w.startswith(base[:4]) or base.startswith(w)
            for w in q_words if len(w) >= 4
        ):
            return True
    return False


# =========================================================
# 🗺️ MENU PATH BUILDER (navigation: HR > Payroll > Setup)
# =========================================================

def build_menu_path(menu_name: str, max_depth: int = 8) -> str:
    """
    Traverse MMENU parentid chain to build full navigation path.
    Returns "Module > SubModule > Screen" style strings (one per match).
    """
    cols = {c.lower() for c in get_columns("mmenu")}
    if "parentid" not in cols or "menuname" not in cols:
        return ""

    clean = re.sub(r"[^\w\s]", "", menu_name).strip()
    if not clean:
        return ""

    find_sql = (
        f"SELECT menuid, menuname, parentid FROM mmenu "
        f"WHERE CAST(menuname AS TEXT) ILIKE '%{clean}%' LIMIT 10"
    )
    find_res = execute_sql_tool(find_sql)
    if "error" in find_res or find_res["df"].empty:
        return ""

    paths = []
    for _, row in find_res["df"].iterrows():
        chain = [str(row["menuname"])]
        current_parent = row.get("parentid")
        visited: set = set()
        depth = 0
        while (
            current_parent is not None
            and str(current_parent).strip() not in ("-1", "None", "nan", "")
            and depth < max_depth
        ):
            key = str(current_parent)
            if key in visited:
                break
            visited.add(key)
            depth += 1
            p_sql = (
                f"SELECT menuid, menuname, parentid FROM mmenu "
                f"WHERE menuid = {current_parent} LIMIT 1"
            )
            p_res = execute_sql_tool(p_sql)
            if "error" in p_res or p_res["df"].empty:
                break
            p_row = p_res["df"].iloc[0]
            chain.insert(0, str(p_row["menuname"]))
            current_parent = p_row.get("parentid")
        if len(chain) > 1:
            paths.append(" > ".join(chain))

    return "\n".join(paths) if paths else ""


# =========================================================
# 🧠 MICRO-PLANNER (fast internal thinking — no LLM, < 1ms)
# =========================================================

def should_think(query: str) -> bool:
    """
    Return True if the query is complex enough to benefit from micro-planning.
    Simple listing / greeting queries don't need a plan.
    """
    q = query.lower()
    if len(q.split()) > 6:
        return True
    return any(k in q for k in [
        "with", "by", "per", "total", "sum", "average", "count",
        "highest", "lowest", "top", "report", "analysis", "group",
        "join", "and their", "along with",
    ])


def think_fast(query: str, target_table: str, is_relational: bool) -> dict:
    """
    Build a lightweight query plan using ALREADY-COMPUTED values from the agent.
    NO re-detection, NO DB calls, NO LLM calls — pure keyword check on query.

    Uses the same keyword sets as existing engines (_AGG_KEYWORDS, _GROUP_TRIGGERS)
    so there is zero conflict with deterministic engine routing.

    Only used to enrich the GPT prompt when all deterministic engines have missed.
    """
    q     = query.lower()
    plan  = {
        "intent":            "general",
        "primary_table":     target_table or "",
        "needs_join":        is_relational,           # from _is_relational_question() — already computed
        "needs_aggregation": False,
        "needs_group_by":    False,
    }

    # Aggregation intent — same keywords as _AGG_KEYWORDS (no duplication of logic)
    if any(t in q for triggers in _AGG_KEYWORDS.values() for t in triggers):
        plan["intent"]            = "aggregation"
        plan["needs_aggregation"] = True

    # GROUP BY intent — same triggers as _GROUP_TRIGGERS (no duplication of logic)
    if any(t in q for t in _GROUP_TRIGGERS):
        plan["intent"]         = "report"
        plan["needs_group_by"] = True

    # JOIN overrides general (relational flag already validated by _is_relational_question)
    if is_relational:
        plan["intent"] = "join"

    # Most complex: JOIN + aggregation (e.g. "total salary per department")
    if is_relational and plan["needs_aggregation"]:
        plan["intent"] = "join_aggregation"

    return plan


# =========================================================
# 🤖 AGENT ORCHESTRATION (RunPod — no local model needed)
# =========================================================

def run_db_agent(query: str, max_rows: int = 50, table_hint: str = "", session_id: str = "") -> str:
    """
    Orchestrates the 4 tools using RunPod SQL endpoint:
      Tool 1: get_schema_tool   → fetch tables + FK relationships
      Tool 2: generate_sql_tool → LLM generates JOIN-aware SELECT (retries once on cold-start timeout)
      Tool 3: execute_sql_tool  → run safely (SELECT only)
      Tool 4: fix_sql_tool      → self-heal on SQL error with schema context
      Tool 5: keyword fallback  → last resort ILIKE query if all above fail
    table_hint: optional table name passed by bots via query_table() to force correct table selection.
    session_id: optional per-user key for follow-up DataFrame context.
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

    # Follow-up filter — if question references "those/that/first/last", try to answer
    # directly from the last result DataFrame without any SQL or LLM call
    if session_id and _is_followup_question(query):
        _stored_df, _stored_query = _get_session_df(session_id)
        if _stored_df is not None:
            _fu_result = _filter_df_by_question(query, _stored_df)
            if _fu_result is not None and not _fu_result.empty:
                logger.info(f"Follow-up filter hit — answered from stored DataFrame ({len(_fu_result)} rows)")
                return (
                    f"Results for '{query}' ({len(_fu_result)} rows):\n\n"
                    + _format_df(_fu_result)
                )

    # Fast path — skip SQL LLM for simple listing queries (saves 5-28s per request)
    # Triggered when: "list all X", "give me X", "show X", "what are X" with no specific filter value
    _q_words = set(query.lower().split())
    _has_list_intent = bool(_q_words & _LIST_INTENTS)
    _clean_q = re.sub(r"[^\w\s]", " ", query)
    _filter_words = [
        w for w in _clean_q.split()
        if len(w) > 2
        and w.lower() not in _STOP_WORDS
        and w.lower() not in _LIST_INTENTS
        and not any(c.isdigit() for c in w)
    ]
    _target_table = table_hint or _detect_table_from_question(query)
    if _has_list_intent and not _filter_words and _target_table:
        _all_fast_cols = get_columns(_target_table)
        _fast_select_cols = [c for c in _all_fast_cols if c.lower() not in _AUDIT_COLS]
        _fast_col_clause = ", ".join(_fast_select_cols) if _fast_select_cols else "*"
        _list_rows = max(max_rows * 4, 200)   # listing gets 4x rows (min 200)
        _fast_sql = f"SELECT {_fast_col_clause} FROM {_target_table.lower()} LIMIT {_list_rows}"
        logger.info(f"Fast path (no LLM): {_fast_sql}")
        _fast_result = execute_sql_tool(_fast_sql)
        if "df" in _fast_result:
            _final = _format_df(_fast_result["df"]) if not _fast_result["df"].empty else "(no rows)"
            _query_cache[_cache_key] = (_final, _now)
            return _final

    # Pre-flight: specific column lookup — "what is the menucode for Purchase Order?"
    # If a filter word IS a column name in the target table, the fallback SQL is already correct.
    # Try it immediately (0 RunPod calls) — saves 70-170s of wasted retry attempts.
    # Only skips sqlcoder when direct SQL actually returns data; otherwise falls through normally.
    _col_set: set = set()
    if _target_table and _filter_words:
        _col_set = {c.lower() for c in get_columns(_target_table)}
        # Exact OR substring match — "expression" matches "formulaexpression" (len > 4 guard)
        _has_col_keyword = any(
            w.lower() in _col_set or (len(w) > 4 and any(w.lower() in c for c in _col_set))
            for w in _filter_words
        )
        if _has_col_keyword:
            _preflight_sql = _build_fallback_sql(query, max_rows, _target_table)
            if _preflight_sql and "SELECT *" not in _preflight_sql:
                _preflight_res = execute_sql_tool(_preflight_sql)
                if "df" in _preflight_res and not _preflight_res["df"].empty:
                    logger.info(f"Pre-flight lookup succeeded — skipped GPT")
                    _final = (
                        f"Results for '{query}' ({len(_preflight_res['df'])} rows):\n\n"
                        + _format_df(_preflight_res["df"])
                    )
                    _query_cache[_cache_key] = (_final, _now)
                    if session_id:
                        _store_session_df(session_id, _preflight_res["df"], query)
                    return _final
                else:
                    # Pre-flight empty — try compound name variants (Basic Salary, Gross Pay, etc.)
                    _non_col_kws = [w for w in _filter_words if w.lower() not in _col_set]
                    _sel_col_kws = [w for w in _filter_words if w.lower() in _col_set]
                    if len(_non_col_kws) >= 2:
                        _phrase = " ".join(_non_col_kws)
                        _tbl_cols = get_columns(_target_table)
                        _variant_df = _try_name_variants(_phrase, _target_table, _sel_col_kws, _tbl_cols, max_rows)
                        if _variant_df is not None:
                            logger.info(f"Name variant pre-flight succeeded for '{_phrase}'")
                            _final = (
                                f"Results for '{query}' ({len(_variant_df)} rows):\n\n"
                                + _format_df(_variant_df)
                            )
                            _query_cache[_cache_key] = (_final, _now)
                            return _final

    # FK filter — triggered when filter keywords don't match any column in target table
    # e.g. "menus where module is Payroll" — "Payroll" is in MMODULE, not MMENU
    _fk_result = None
    if _target_table and _filter_words:
        _fk_unmatched = [w for w in _filter_words if w.lower() not in _col_set]
        if _fk_unmatched:
            _rels_for_fk = get_relationships()
            _fk_result = _try_fk_filter(query, _fk_unmatched, _target_table, _rels_for_fk, max_rows)
            if _fk_result:
                logger.info(f"FK filter path succeeded — skipped GPT")
                _query_cache[_cache_key] = (_fk_result, _now)
                return _fk_result

    # Status-filter pre-flight — after FK filter fails, try ILIKE fallback SQL before GPT
    # Handles: "list all active menus", "find pending reports", "show enabled modules"
    # These have filter words that are values (not columns or FK entities) → ILIKE search is correct
    if _target_table and _filter_words and not _fk_result:
        _sf_sql = _build_fallback_sql(query, max_rows, _target_table)
        if _sf_sql and "SELECT *" not in _sf_sql:
            _sf_res = execute_sql_tool(_sf_sql)
            if "df" in _sf_res and not _sf_res["df"].empty:
                logger.info(f"Status-filter pre-flight succeeded — skipped GPT")
                _final = (
                    f"Results for '{query}' ({len(_sf_res['df'])} rows):\n\n"
                    + _format_df(_sf_res["df"])
                )
                _query_cache[_cache_key] = (_final, _now)
                if session_id:
                    _store_session_df(session_id, _sf_res["df"], query)
                return _final

    # Report engine — GROUP BY aggregation without GPT (runs BEFORE plain aggregation)
    # "how many reports by category" → GROUP BY, not a plain COUNT(*)
    # e.g. "reports by category", "employees per department", "count by module"
    if _target_table:
        _rels_for_report = get_relationships()
        _report_result = _try_report_engine(query, _target_table, _rels_for_report, max_rows)
        if _report_result:
            logger.info(f"Report engine succeeded — skipped GPT")
            _query_cache[_cache_key] = (_report_result, _now)
            return _report_result

    # Aggregation engine — SUM/COUNT/AVG/MAX/MIN without GPT (plain, no GROUP BY)
    # e.g. "total gross pay", "count of employees", "average salary"
    # Pass rels for FK-based WHERE filter (e.g. "total salary for Finance dept")
    if _target_table:
        _agg_result = _try_aggregation_engine(query, _target_table, max_rows, _rels_for_report)
        if _agg_result:
            logger.info(f"Aggregation engine succeeded — skipped GPT")
            _query_cache[_cache_key] = (_agg_result, _now)
            if session_id:
                _store_session_df(session_id, pd.DataFrame(), query)
            return _agg_result

    # Multi-table merge — Python pandas JOIN for relational questions
    # Triggered by: classic phrases OR entity detection (FK-target table names in question)
    # More reliable than GPT-generated SQL JOINs for 2-3 table queries
    _was_relational = False   # captured for micro-planner — no logic change
    if _target_table:
        _rels_for_merge = get_relationships()
        _was_relational = _is_relational_question(query, _target_table, _rels_for_merge)
        if _was_relational:
            _merge_result = _try_multi_table_merge(query, _target_table, _rels_for_merge, max_rows)
            if _merge_result:
                logger.info(f"Multi-table merge succeeded — skipped GPT JOIN")
                _query_cache[_cache_key] = (_merge_result, _now)
                return _merge_result

    # Learning: Pattern cache — re-execute previously successful GPT SQL (skips GPT entirely)
    # Only fires after all deterministic engines have already been tried and missed.
    # Cached SQL always runs against live DB — data is never stale.
    # Auto-invalidates if re-execution fails (schema changed, table renamed, etc.)
    if _LEARNING_ENABLED and _target_table:
        try:
            _cached_sql = _ls.get_pattern(query)
            if _cached_sql:
                _pc_res = execute_sql_tool(_cached_sql)
                if "error" in _pc_res:
                    # SQL execution error — schema likely changed, auto-invalidate
                    logger.info(f"[Learning] Cached SQL broken — invalidating pattern")
                    _ls.invalidate_pattern(query)
                elif "df" in _pc_res and not _pc_res["df"].empty:
                    # Cache hit with data — return immediately, skip GPT
                    logger.info(f"[Learning] Pattern cache hit — skipped GPT")
                    _pc_final = (
                        f"Results for '{query}' ({len(_pc_res['df'])} rows):\n\n"
                        + _format_df(_pc_res["df"])
                    )
                    _query_cache[_cache_key] = (_pc_final, _now)
                    if session_id:
                        _store_session_df(session_id, _pc_res["df"], query)
                    return _pc_final
                # else: valid SQL but empty result — keep pattern, fall through normally
        except Exception:
            pass

    # Micro-planner: build plan from already-computed values (_target_table, _was_relational)
    # No re-detection, no DB calls, < 1ms. Only enriches GPT prompt — no engine logic touched.
    _gpt_query = query   # default: unchanged — if plan adds nothing, GPT gets original question
    if should_think(query) and _target_table:
        try:
            _plan = think_fast(query, _target_table, _was_relational)
            _hints = []
            if _plan["needs_join"]:         _hints.append("Use JOIN between tables.")
            if _plan["needs_aggregation"]:  _hints.append("Use SUM/COUNT/AVG.")
            if _plan["needs_group_by"]:     _hints.append("Use GROUP BY.")
            if _hints:
                _gpt_query = f"{query} [{' '.join(_hints)}]"
                logger.info(f"[MicroPlanner] intent={_plan['intent']} hints={_hints}")
        except Exception:
            _gpt_query = query   # safe fallback — never break the GPT path

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
            sql = generate_sql_tool(_gpt_query, schema, max_rows)
            if sql:
                sql = _strip_system_joins(sql)
        except TimeoutError:
            logger.warning("SQL LLM timed out (cold start) — retrying once")
            try:
                sql = generate_sql_tool(_gpt_query, schema, max_rows)
            except (TimeoutError, RuntimeError) as e:
                logger.warning(f"SQL generation failed after retry: {e} — using keyword fallback")
        except ValueError as e:
            # Truncated / too-complex SQL — retry with simplified 2-table schema
            if any(k in str(e).lower() for k in ("truncated", "unbalanced", "nested", "existence-check")):
                logger.warning(f"SQL complexity error: {e} — retrying with 2-table schema")
                try:
                    simple_schema = get_schema_tool(question=query, table_hint=table_hint, max_tables=2)
                    sql = generate_sql_tool(_gpt_query, simple_schema, max_rows)
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
                    sql = generate_sql_tool(_gpt_query, simple_schema, max_rows)
                    if sql:
                        sql = _strip_system_joins(sql)
                    logger.info("2-table retry succeeded after schema-too-large")
                except Exception as _e2:
                    if "schema too large" in str(_e2).lower():
                        logger.warning(f"2-table still too large — retrying with 1-table schema")
                        try:
                            single_schema = get_schema_tool(question=query, table_hint=table_hint, max_tables=1)
                            sql = generate_sql_tool(_gpt_query, single_schema, max_rows)
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

            # Learning: record SQL error for this table (monitoring + future skip logic)
            if "error" in result and _LEARNING_ENABLED and _target_table:
                try: _ls.save_failure(_target_table, result["error"])
                except Exception: pass

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
            if session_id:
                _store_session_df(session_id, df, query)
            # Learning: save successful GPT SQL as pattern + save content words as synonyms
            if _LEARNING_ENABLED and _target_table:
                try:
                    if sql and sql.strip().upper().startswith("SELECT"):
                        _ls.save_pattern(query, sql, _target_table)
                    # Save content words from this query as synonyms for the detected table
                    for _qw in re.findall(r'\b\w{4,}\b', query):
                        if _qw.lower() not in _STOP_WORDS and _qw.lower() not in _LIST_INTENTS:
                            _ls.save_synonym(_qw, _target_table)
                            break  # save only the first meaningful word (most specific)
                except Exception:
                    pass
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

def query_table(table_name: str, search_term: str, max_rows: int = 50, session_id: str = "") -> str:
    """Compatibility wrapper — called by all bots. Delegates to run_db_agent.
    Passes table_name as a hint so Schema RAG always prioritises the correct table.
    session_id: optional per-user key for follow-up DataFrame context.
    """
    return run_db_agent(search_term, max_rows, table_hint=table_name, session_id=session_id)


# =========================================================
# 🔁 COMPATIBILITY ALIASES (used by schema_bot)
# =========================================================

def _get_columns(table_name: str) -> list:
    return get_columns(table_name)


def _get_all_tables() -> list:
    return get_tables()


def _detect_table_from_question(user_question: str) -> "str | None":
    """Find the most likely table name from the user question (3-pass)."""

    # Pass 0 — learned synonyms (zero DB calls — fastest path)
    # e.g. "payslip" → MSALARYPROCESSING after first confirmed query
    if _LEARNING_ENABLED:
        try:
            _syn_words = sorted(re.findall(r'\b\w{4,}\b', user_question), key=len, reverse=True)
            for _sw in _syn_words:
                if _sw.lower() not in _STOP_WORDS:
                    _syn_tbl = _ls.get_synonym(_sw)
                    if _syn_tbl:
                        logger.info(f"[Learning] Synonym: '{_sw}' → {_syn_tbl}")
                        return _syn_tbl
        except Exception:
            pass

    all_tables = get_tables()
    if not all_tables:
        return None

    tables_upper = {t.upper(): t for t in all_tables}
    words = re.findall(r'\b\w{3,}\b', user_question)
    words_sorted = sorted(set(words), key=len, reverse=True)

    # Pass 1 — exact single-word match (case-insensitive)
    for word in words_sorted:
        if word.upper() in tables_upper:
            _p1_result = tables_upper[word.upper()]
            if _LEARNING_ENABLED:
                try: _ls.save_synonym(word, _p1_result)
                except Exception: pass
            return _p1_result

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
            0 if x[2].lower() in _CORE_TABLES else 1,     # core tables FIRST (above quality)
            x[0],                                          # quality: 0=prefix, 1=suffix, 2=other
            0 if x[2].upper().startswith('M') else 1,     # M-prefix tables before others
            x[1],                                          # shorter table name first
        ))
        _p3_result = partial_hits[0][2]
        if _LEARNING_ENABLED:
            try:
                # Save the first meaningful content word as synonym for this table
                for _p3w in words_sorted:
                    if len(_p3w) >= 4 and _p3w.lower() not in _STOP_WORDS:
                        _ls.save_synonym(_p3w, _p3_result)
                        break
            except Exception:
                pass
        return _p3_result

    return None


# =========================================================
# 🧪 TEST
# =========================================================

if __name__ == "__main__":
    print(run_db_agent("get employee names with department names"))

