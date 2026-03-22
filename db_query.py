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
        _table_cache_ts = now
        return _table_cache
    except Exception as e:
        logger.warning(f"Could not get table list: {e}")
        return _table_cache


def get_columns(table: str) -> list:
    try:
        q = sa_text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name=:tname "
            "ORDER BY ordinal_position"
        )
        with _get_engine().connect() as conn:
            df = pd.read_sql(q, conn, params={"tname": table.lower()})
        return df["column_name"].tolist()
    except Exception as e:
        logger.warning(f"Could not get columns for '{table}': {e}")
        return []


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


def get_schema_tool(tables_limit: int = 20, question: str = "") -> str:
    """TOOL 1 — Returns tables + columns + FK relationships.
    When question is provided, prioritises the detected target table and its
    FK neighbours so the LLM always sees the relevant schema regardless of
    how many tables exist in the database.
    """
    tables = get_tables()
    rels = get_relationships()

    if question:
        # Pass 0 — Schema RAG (semantic search, faster + more accurate than keyword)
        # Fully wrapped: any failure silently falls through to existing keyword logic below
        _rag_tables: list = []
        try:
            from schema_rag import is_index_ready, search_schema
            if is_index_ready():
                _rag_tables = search_schema(question, top_k=5)
        except Exception:
            pass

        if _rag_tables:
            # RAG found relevant tables — build related set from RAG results + their FK neighbours
            related: set = set(t.lower() for t in _rag_tables)
            for t in _rag_tables:
                for r in rels:
                    if r['source_table'].lower() == t.lower():
                        related.add(r['target_table'].lower())
                    elif r['target_table'].lower() == t.lower():
                        related.add(r['source_table'].lower())
        else:
            # Existing keyword logic (Pass 1/2/3) — completely unchanged
            target = _detect_table_from_question(question)
            related: set = set()
            if target:
                related.add(target.lower())
                for r in rels:
                    if r['source_table'].lower() == target.lower():
                        related.add(r['target_table'].lower())
                    elif r['target_table'].lower() == target.lower():
                        related.add(r['source_table'].lower())

        # Priority tables first (related), then fill remaining slots with others
        # When RAG found tables, use only those + FK neighbours (no padding with unrelated tables)
        priority = [t for t in tables if t.lower() in related]
        others   = [t for t in tables if t.lower() not in related]
        selected = priority if _rag_tables else priority + others[:max(0, 5 - len(priority))]

        schema_lines = []
        for t in selected:
            cols = get_columns(t)
            schema_lines.append(f"{t}({', '.join(cols[:15])})")  # cap at 15 cols per table

        # Only include FK relationships between selected tables
        selected_lower = {t.lower() for t in selected}
        rel_lines = [
            f"{r['source_table']}.{r['source_column']} -> {r['target_table']}.{r['target_column']}"
            for r in rels
            if r['source_table'].lower() in selected_lower
            and r['target_table'].lower() in selected_lower
        ]

        result = "SCHEMA:\n" + "\n".join(schema_lines)
        if rel_lines:
            result += "\nFKs:\n" + "\n".join(rel_lines)
        return result
    else:
        schema_lines = []
        for t in tables[:5]:
            cols = get_columns(t)
            schema_lines.append(f"{t}({', '.join(cols[:15])})")
        result = "SCHEMA:\n" + "\n".join(schema_lines)
        return result


# =========================================================
# 🧠 TOOL 2 — GENERATE SQL
# =========================================================

def generate_sql_tool(question: str, schema: str, max_rows: int = 50) -> str:
    """TOOL 2 — Generate PostgreSQL SELECT from question + schema via RunPod."""
    from shared_resources import call_sql_endpoint

    enriched_question = f"""{question}

STRICT RULES:
- ONLY SELECT queries — never INSERT, UPDATE, DELETE, DROP, ALTER
- Use LIMIT {max_rows}
- All table/column names are LOWERCASE — never use uppercase or quoted identifiers
- Use proper JOIN when question involves multiple tables
- Use aliases (e, d, o, c) for readability
- FILTER RULE: Only add WHERE if question has a specific filter value (name, ID, keyword)
  * 'show/list/get/give all X' with no filter → NO WHERE clause
  * 'employee named John' → WHERE CAST(employeename AS TEXT) ILIKE '%John%'
- Use ILIKE for text search: CAST(col AS TEXT) ILIKE '%value%'
- Use NOW() not GETDATE(), COALESCE not ISNULL
- Return ONLY the raw SQL — no explanation, no markdown, no code fences"""

    sql = call_sql_endpoint(query=enriched_question, schema=schema)

    # Unwrap {'output': '...'} if LLM returned wrapped format
    if sql.strip().startswith("{") and ("'output'" in sql or '"output"' in sql):
        match = re.search(r"""['"]output['"]\s*:\s*['"](.+)""", sql.strip(), re.DOTALL)
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

    if any(word in sql_lower for word in ["insert", "update", "delete", "drop", "alter"]):
        return {"error": "Write operations are not allowed"}

    try:
        with _get_engine().connect() as conn:
            df = pd.read_sql(sa_text(sql), conn)
        return {"df": df, "count": len(df)}
    except Exception as e:
        return {"error": str(e)}


# =========================================================
# 🔧 TOOL 4 — FIX SQL (SELF HEALING)
# =========================================================

def fix_sql_tool(sql: str, error: str) -> str:
    """TOOL 4 — Fix broken SQL using error message via RunPod."""
    from shared_resources import call_sql_endpoint

    fix_query = (
        f"Fix this broken PostgreSQL SELECT query and return only the corrected SQL.\n"
        f"Error: {error}\n"
        f"Rules: Only SELECT, fix the syntax error, keep same meaning, all names lowercase."
    )
    fix_schema = f"Broken SQL to fix:\n{sql}"

    fixed = call_sql_endpoint(query=fix_query, schema=fix_schema)
    fixed = re.sub(r"```(?:sql)?", "", fixed, flags=re.IGNORECASE).replace("```", "").strip()
    return _fix_pg_syntax(fixed)


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

def _build_fallback_sql(query: str, max_rows: int) -> str:
    """Build a simple ILIKE keyword query when LLM + self-heal both fail."""
    table = _detect_table_from_question(query)
    if not table:
        return ""
    col_names = get_columns(table)
    if not col_names:
        return f"SELECT * FROM {table.lower()} LIMIT {max_rows}"
    raw_words = query.split()
    keywords = [w for w in raw_words if len(w) > 2 and w.lower() not in _STOP_WORDS]
    if not keywords:
        keywords = [w for w in raw_words if len(w) > 2] or [query]
    tbl = table.lower()
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

def run_db_agent(query: str, max_rows: int = 50) -> str:
    """
    Orchestrates the 4 tools using RunPod SQL endpoint:
      Tool 1: get_schema_tool   → fetch tables + FK relationships
      Tool 2: generate_sql_tool → LLM generates JOIN-aware SELECT (retries once on cold-start timeout)
      Tool 3: execute_sql_tool  → run safely (SELECT only)
      Tool 4: fix_sql_tool      → self-heal on SQL error, retry once
      Tool 5: keyword fallback  → last resort ILIKE query if all above fail
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

    try:
        # Tool 1 — Schema (query-aware: relevant tables first + full table name list)
        schema = get_schema_tool(question=query)

        # Tool 2 — Generate SQL (with cold-start retry on TimeoutError)
        try:
            sql = generate_sql_tool(query, schema, max_rows)
        except TimeoutError:
            logger.warning("SQL LLM timed out (cold start) — retrying once, worker should be warm now")
            sql = generate_sql_tool(query, schema, max_rows)
        logger.info(f"Generated SQL: {sql[:200]}")

        if not sql.strip().upper().startswith("SELECT"):
            raise ValueError(f"Non-SELECT query blocked: {sql[:100]}")

        # Tool 3 — Execute
        result = execute_sql_tool(sql)

        # Tool 4 — Self-heal if error
        if "error" in result:
            logger.warning(f"SQL error: {result['error']} — attempting self-heal")
            try:
                fixed_sql = fix_sql_tool(sql, result["error"])
            except TimeoutError:
                logger.warning("Self-heal LLM timed out (cold start) — retrying once")
                fixed_sql = fix_sql_tool(sql, result["error"])
            logger.info(f"Fixed SQL: {fixed_sql[:200]}")
            result = execute_sql_tool(fixed_sql)

        # Keyword fallback — last resort if LLM + self-heal both failed
        if "error" in result:
            logger.warning(f"SQL failed after self-heal: {result['error']} — trying keyword fallback")
            fallback_sql = _build_fallback_sql(query, max_rows)
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

        return f"No data found for: {query}"

    except Exception as e:
        logger.error(f"run_db_agent error: {e}")
        return f"Error: {e}"


# =========================================================
# 🚀 PUBLIC API
# =========================================================

def query_table(table_name: str, search_term: str, max_rows: int = 50) -> str:
    """Compatibility wrapper — called by all bots. Delegates to run_db_agent."""
    return run_db_agent(f"{search_term} from {table_name}", max_rows)


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

    if partial_hits:
        partial_hits.sort(key=lambda x: (x[0], 0 if x[1].upper().startswith('M') else 1, x[1]))
        return partial_hits[0][1]

    return None


# =========================================================
# 🧪 TEST
# =========================================================

if __name__ == "__main__":
    print(run_db_agent("get employee names with department names"))
