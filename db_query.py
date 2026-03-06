"""
db_query.py

Utility for querying the DuckDB knowledge database.
Used by menu_bot, report_bot, formula_bot, project_bot, schema_bot.
"""

import os
import logging
import re
import duckdb

logger = logging.getLogger(__name__)

DATA_DIR    = "/app/data"
DUCKDB_PATH = os.path.join(DATA_DIR, "knowledge.duckdb")


def _get_llm():
    """Lazy-load LLM to avoid circular imports at module level."""
    try:
        from shared_resources import ai_resources
        return ai_resources.response_llm
    except Exception:
        return None


def _generate_sql(llm, table_name: str, col_names: list, user_question: str, max_rows: int) -> str:
    """Ask the LLM to generate a SQL SELECT query for the given question."""
    schema_info = f"Table: \"{table_name}\"\nColumns: {', '.join(col_names)}"
    sql_prompt = (
        f"You are a SQL expert for DuckDB. Generate a single valid DuckDB SQL SELECT query.\n\n"
        f"Table schema:\n{schema_info}\n\n"
        f"User question: {user_question}\n\n"
        f"Rules:\n"
        f"- Only use SELECT statements\n"
        f"- Use ILIKE for case-insensitive text searches\n"
        f"- Use CAST(col AS VARCHAR) when comparing non-text columns to strings\n"
        f"- Add LIMIT {max_rows} at the end\n"
        f"- Return ONLY the raw SQL query — no explanation, no markdown, no code fences\n\n"
        f"SQL:"
    )
    raw = llm.invoke(sql_prompt)
    sql = raw.content if hasattr(raw, "content") else str(raw)
    # Strip markdown code fences if present
    sql = re.sub(r"```(?:sql)?", "", sql, flags=re.IGNORECASE).replace("```", "").strip()
    return sql


def _ilike_sql(table_name: str, col_names: list, search_term: str, max_rows: int) -> str:
    """Build a fallback ILIKE SQL query from keywords."""
    keywords = [w for w in search_term.split() if len(w) > 2] or [search_term]
    word_blocks = []
    for kw in keywords[:6]:
        safe_kw = kw.replace("'", "")
        col_conds = " OR ".join(
            f'CAST("{col}" AS VARCHAR) ILIKE \'%{safe_kw}%\''
            for col in col_names
        )
        word_blocks.append(f"({col_conds})")
    where_clause = " OR ".join(word_blocks)
    return f'SELECT * FROM "{table_name}" WHERE {where_clause} LIMIT {max_rows}'


def query_table(table_name: str, search_term: str, max_rows: int = 50) -> str:
    """
    Query *table_name* in DuckDB using Text-to-SQL (LLM-generated query).

    Strategy:
      1. LLM generates a SQL SELECT query from the user question + table schema.
      2. If LLM query fails or returns no rows, fall back to keyword ILIKE search.
      3. If still no rows, return a 30-row sample so the LLM has schema context.

    Returns a formatted string ready to be injected into an LLM prompt.
    """
    if not os.path.exists(DUCKDB_PATH):
        return (
            "Structured database not initialised. "
            "knowledge_loader.load_all() must run at startup before queries can be served."
        )

    try:
        conn = duckdb.connect(DUCKDB_PATH, read_only=True)

        # Get column names for this table
        col_names = [
            row[0]
            for row in conn.execute(f'DESCRIBE "{table_name}"').fetchall()
        ]

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

                df = conn.execute(generated_sql).fetchdf()
                if df.empty:
                    logger.info("Text-to-SQL returned 0 rows — falling back to ILIKE")
                    df = None
                else:
                    logger.info(f"Text-to-SQL matched {len(df)} rows")
            except Exception as sql_exc:
                logger.warning(f"Text-to-SQL failed ({sql_exc}) — falling back to ILIKE")
                df = None

        # ── Step 2: Fallback — keyword ILIKE search ────────────────────────────
        if df is None:
            fallback_sql = _ilike_sql(table_name, col_names, search_term, max_rows)
            df = conn.execute(fallback_sql).fetchdf()

        conn.close()

        # ── Step 3: If still empty, return a broad sample ─────────────────────
        if df.empty:
            conn2 = duckdb.connect(DUCKDB_PATH, read_only=True)
            df = conn2.execute(f'SELECT * FROM "{table_name}" LIMIT 30').fetchdf()
            conn2.close()
            return (
                f"No exact matches for '{search_term}' in table '{table_name}'.\n"
                f"Available data overview ({len(df)} rows shown):\n\n"
                + df.to_string(index=False)
            )

        return (
            f"Results for '{search_term}' from table '{table_name}' "
            f"({len(df)} matching rows):\n\n"
            + df.to_string(index=False)
        )

    except Exception as exc:
        logger.error(f"DuckDB query error on table '{table_name}': {exc}")
        return f"Error querying table '{table_name}': {exc}"
