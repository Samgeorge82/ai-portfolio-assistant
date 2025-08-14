# streamlit_app.py
import os
import re
import json
import hashlib
from typing import List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
import numpy as np

# ===================== Robust OpenAI / LangChain imports =====================
# Works with both modern and older LangChain installs.
try:
    from langchain_openai import ChatOpenAI  # modern
    _LC_MODE = "modern"
except ModuleNotFoundError:
    from langchain.chat_models import ChatOpenAI  # legacy
    _LC_MODE = "legacy"

# ===================== App config =====================
st.set_page_config(page_title="Ask DGX — AI Portfolio Assistant", layout="wide")
st.title("⚡ Ask DGX — AI Portfolio Assistant")
st.caption("Structured filters first (numbers/dates/text), then optional LLM synthesis. No more ‘imaginary’ capacities.")

# ===================== Secrets / keys =====================
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
if OPENAI_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY

def make_llm(model_pref: str = "gpt-4o-mini", temperature: float = 0.0):
    kwargs = {"temperature": temperature}
    if _LC_MODE == "modern":
        kwargs["model"] = model_pref
    else:
        kwargs["model_name"] = "gpt-4o"
    return ChatOpenAI(**kwargs)

judge_llm = make_llm("gpt-4o-mini", 0.0)
writer_llm = make_llm("gpt-4o-mini", 0.1)

# ===================== UI =====================
uploaded_file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
question = st.text_input("Ask a portfolio-related question (e.g., 'projects in construction with total installed capacity > 6000 MW')")

with st.expander("Optional: map key columns (auto-detect tries its best)"):
    user_name_col = st.text_input("Project Name column (optional)", value="")
    user_status_col = st.text_input("Project Status/Phase column (optional)", value="")
    user_country_col = st.text_input("Country/Market column (optional)", value="")

# ===================== Helpers =====================
@st.cache_data(show_spinner=False)
def read_excel_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_excel(b)

def file_sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def normalize_col(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def guess_column(df: pd.DataFrame, candidates: List[str]) -> str:
    cols = list(df.columns)
    norm_map = {normalize_col(c): c for c in cols}
    for cand in candidates:
        if cand in norm_map:  # exact normalized hit
            return norm_map[cand]
    # fuzzy contains
    for c in cols:
        cn = normalize_col(c)
        if any(cand in cn for cand in candidates):
            return c
    return ""

def coerce_numeric(series: pd.Series) -> pd.Series:
    # Strip commas, MW suffixes, whitespace; convert to float
    s = series.astype(str).str.replace(",", "", regex=False)
    s = s.str.replace(r"\s*mw\s*$", "", regex=True, flags=re.I)
    s = s.str.strip()
    return pd.to_numeric(s, errors="coerce")

def coerce_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False, dayfirst=False)

def list_columns_with_types(df: pd.DataFrame, sample_rows: int = 200) -> Dict[str, str]:
    """
    Infer coarse types: 'number', 'date', 'text' by sampling.
    """
    info = {}
    sample = df.head(sample_rows)
    for c in df.columns:
        # try number
        num_try = pd.to_numeric(sample[c], errors="coerce")
        num_ratio = num_try.notna().mean()
        # try date
        date_try = pd.to_datetime(sample[c], errors="coerce")
        date_ratio = date_try.notna().mean()
        if num_ratio >= 0.8:
            info[c] = "number"
        elif date_ratio >= 0.8:
            info[c] = "date"
        else:
            info[c] = "text"
    return info

# ===================== Structured filter extraction =====================
SUPPORTED_OPS = {
    "equals": "==",
    "==": "==",
    "=": "==",
    "!=": "!=",
    "not equals": "!=",
    ">": ">",
    ">=": ">=",
    "<": "<",
    "<=": "<=",
    "contains": "contains",        # text only
    "icontains": "icontains",      # text only (case-insensitive)
    "in": "in",                    # text/enum
    "between": "between",          # numeric/date ranges
}

def build_schema_prompt(df: pd.DataFrame) -> str:
    types = list_columns_with_types(df)
    lines = [f"- {c} ({t})" for c, t in types.items()]
    return "\n".join(lines)

def parse_filters_with_llm(q: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Ask the model to emit structured filters in a strict JSON schema.
    """
    schema = build_schema_prompt(df)
    prompt = f"""
You convert natural-language portfolio questions into structured filters for a pandas DataFrame.

Columns (with inferred types):
{schema}

Supported operators (choose correctly by type):
- number/date: ==, !=, >, >=, <, <=, between
- text: equals (==), !=, contains, icontains, in

Return STRICT JSON with this shape (no prose, no markdown):
{{
  "filters": [
    {{"column": "<exact column name>", "operator": "<one of {list(SUPPORTED_OPS.keys())}>", "value": <value or [min,max] for between>}},
    ...
  ],
  "select": ["<optional columns to display>"]
}}

Rules:
- Normalize synonyms: "construction", "under construction" → match exactly what's in the data if possible.
- If units like "MW" appear, strip units and use numbers only.
- If a column name is ambiguous, pick the most likely from Columns list.
- If no explicit filters in the question, return {{"filters": [], "select": []}}.

Question: {q}
JSON:
"""
    try:
        out = judge_llm.invoke(prompt).content.strip()
        obj = json.loads(out)
        if "filters" not in obj:
            obj["filters"] = []
        if "select" not in obj:
            obj["select"] = []
        return obj
    except Exception:
        return {"filters": [], "select": []}

def apply_single_filter(df: pd.DataFrame, f: Dict[str, Any]) -> pd.Series:
    col = f.get("column", "")
    op = (f.get("operator", "") or "").lower()
    val = f.get("value", None)
    if col not in df.columns or op not in SUPPORTED_OPS:
        return pd.Series([True]*len(df))  # ignore bad filter

    # Infer column type again for safety
    types = list_columns_with_types(df)
    ctype = types.get(col, "text")

    if ctype == "number":
        series = coerce_numeric(df[col])
        if op in {"==", "!=","<", "<=", ">", ">="}:
            try:
                v = float(val)
            except Exception:
                return pd.Series([True]*len(df))
            expr = {
                "==": series == v,
                "!=": series != v,
                "<": series < v,
                "<=": series <= v,
                ">": series > v,
                ">=": series >= v,
            }[op]
            return expr.fillna(False)
        elif op == "between":
            try:
                lo, hi = float(val[0]), float(val[1])
            except Exception:
                return pd.Series([True]*len(df))
            return series.between(lo, hi, inclusive="both").fillna(False)
        elif op in {"contains", "icontains", "in"}:
            # not sensible for numeric; ignore
            return pd.Series([True]*len(df))
        else:
            return pd.Series([True]*len(df))

    if ctype == "date":
        series = coerce_date(df[col])
        if isinstance(val, list) and op == "between":
            try:
                lo = pd.to_datetime(val[0], errors="coerce")
                hi = pd.to_datetime(val[1], errors="coerce")
            except Exception:
                return pd.Series([True]*len(df))
            return series.between(lo, hi, inclusive="both").fillna(False)
        else:
            # normalize val to datetime for comparisons
            v = pd.to_datetime(val, errors="coerce")
            if pd.isna(v):
                return pd.Series([True]*len(df))
            if op == "==":
                return (series.dt.date == v.date()).fillna(False)
            if op == "!=":
                return (series.dt.date != v.date()).fillna(False)
            if op == ">":
                return (series > v).fillna(False)
            if op == ">=":
                return (series >= v).fillna(False)
            if op == "<":
                return (series < v).fillna(False)
            if op == "<=":
                return (series <= v).fillna(False)
            if op == "between" and isinstance(val, list):
                # covered above
                return pd.Series([True]*len(df))
            return pd.Series([True]*len(df))

    # text
    series = df[col].astype(str)
    if op in {"==", "equals"}:
        return (series.str.lower().str.strip() == str(val).lower().strip())
    if op in {"!=", "not equals"}:
        return (series.str.lower().str.strip() != str(val).lower().strip())
    if op == "contains":
        return series.str.contains(str(val), case=True, na=False)
    if op == "icontains":
        return series.str.contains(str(val), case=False, na=False)
    if op == "in":
        try:
            vals = [str(x).lower().strip() for x in (val if isinstance(val, list) else [val])]
        except Exception:
            vals = [str(val).lower().strip()]
        return series.str.lower().str.strip().isin(vals)
    # between for text not supported
    return pd.Series([True]*len(df))

def apply_filters(df: pd.DataFrame, filters: List[Dict[str, Any]]) -> pd.DataFrame:
    if not filters:
        return df
    mask = pd.Series([True]*len(df))
    for f in filters:
        mask = mask & apply_single_filter(df, f)
    return df[mask]

def pretty_number(x) -> str:
    try:
        f = float(x)
        if np.isnan(f):
            return ""
        return f"{f:,.0f}"
    except Exception:
        return str(x)

# ===================== Main logic =====================
if uploaded_file and question:
    # Read and memoize DF (bytes are hashable)
    file_bytes = uploaded_file.getvalue()
    df = read_excel_bytes(file_bytes)

    # Try to auto-map key columns unless user provided
    status_col_guess = guess_column(df, [
        "status", "project status", "phase", "project phase", "construction status"
    ])
    name_col_guess = guess_column(df, ["name", "project name", "project", "asset name"])
    country_col_guess = guess_column(df, ["country", "market", "region"])

    name_col = user_name_col or name_col_guess or (df.columns[0] if len(df.columns) else "")
    status_col = user_status_col or status_col_guess
    country_col = user_country_col or country_col_guess

    with st.expander("Detected schema & types"):
        types = list_columns_with_types(df)
        st.write(pd.DataFrame({"column": list(types.keys()), "type": list(types.values())}))

    # 1) Extract structured filters from the question
    parsed = parse_filters_with_llm(question, df)
    st.markdown("#### 🧠 Parsed filters")
    st.code(json.dumps(parsed, indent=2))

    # 2) Apply filters in Pandas (numbers/dates/text handled explicitly)
    df_filtered = apply_filters(df, parsed.get("filters", []))

    # If the user asked something like "in construction with total installed capacity > 6000",
    # this will now be deterministic and correct, no LLM guessing.

    # 3) Display results table first (truth source)
    if df_filtered.empty:
        st.warning("No rows match the filters. Try relaxing constraints or check column naming.")
    else:
        # If they filtered on capacity, format any column that looks like capacity
        cap_like = [c for c in df_filtered.columns if re.search(r"capacity|mw|kw|gw", c, re.I)]
        df_show = df_filtered.copy()
        for c in cap_like:
            # attempt numeric pretty print without losing original
            nums = coerce_numeric(df_show[c])
            mask = nums.notna()
            df_show.loc[mask, c] = nums[mask].apply(pretty_number)

        # Display only selected columns if provided
        sel = parsed.get("select", [])
        if sel:
            sel = [c for c in sel if c in df_show.columns]
            if sel:
                df_show = df_show[sel]

        st.markdown("### 📊 Matching rows (ground truth)")
        st.dataframe(df_show, use_container_width=True)

    # 4) Optional LLM synthesis over the filtered subset (small sample to avoid token bloat)
    want_summary = st.toggle("Generate LLM summary (optional)", value=True)
    if want_summary and not df_filtered.empty:
        # build a compact, typed summary of the filtered rows
        MAX_ROWS = 40  # cap to keep prompt manageable
        sample = df_filtered.head(MAX_ROWS)

        # Identify likely key fields for summary
        likely_name = name_col or guess_column(df, ["name", "project name"])
        likely_status = status_col or guess_column(df, ["status", "phase"])
        # pick a likely capacity column
        cap_col_guess = guess_column(df, [
            "total installed capacity (mw)",
            "total capacity (mw)",
            "capacity (mw)",
            "installed capacity",
            "capacity"
        ])

        # Build a safe, typed JSON payload for the LLM
        def safe_val(v):
            if pd.isna(v):
                return None
            if isinstance(v, (np.floating, float, int, np.integer)):
                return float(v)
            return str(v)

        rows_payload = []
        for _, r in sample.iterrows():
            item = {}
            for c in df.columns:
                item[c] = safe_val(r[c])
            rows_payload.append(item)

        prompt = f"""
You are summarizing filtered offshore wind portfolio rows. Be precise and concise.
- Do NOT invent numbers; only reference values present in the JSON rows.
- Prefer these fields if present: Name: "{likely_name}", Status: "{likely_status}", Capacity: "{cap_col_guess}".
- If capacity appears, treat it as MW (number).
- If totals are asked, compute from the rows provided only.

User question:
{question}

Rows (JSON, up to {MAX_ROWS} rows):
{json.dumps(rows_payload)[:18000]}

Write:
1) A clear 5–8 line answer grounded in the data.
2) A short bullet list of the matching projects with Name, Status, and Capacity (if available).
3) A 'Suggested actions 🔔' 3–5 bullets (validate any ambiguous fields, next steps).
"""
        llm_answer = writer_llm.invoke(prompt).content
        st.markdown("### 💡 Assistant’s summary (grounded in the table above)")
        st.write(llm_answer)

    # 5) Quick query templates (quality-of-life)
    with st.expander("🔎 Quick templates"):
        st.caption("Click to append:")
        cols = st.columns(3)
        if cols[0].button("Construction & capacity > 6000 MW"):
            st.session_state["q"] = "Which projects are in construction with total installed capacity > 6000 MW?"
        if cols[1].button("COD between 2028 and 2031"):
            st.session_state["q"] = "List projects with COD between 2028-01-01 and 2031-12-31."
        if cols[2].button("Country contains Germany & risk icontains high"):
            st.session_state["q"] = "Show projects where Country contains Germany and Risk icontains high."

else:
    st.info("Upload an Excel and ask a question to begin.")

# ===================== Notes =====================
# - This app FIRST applies Pandas filters inferred from natural language:
#   • numeric (>, >=, <, <=, ==, !=, between)
#   • date (same operators + between)
#   • text (equals/!=/contains/icontains/in)
# - That stops the LLM from making up capacities or IDs.
# - Then, optionally, the LLM writes a readable summary based only on the filtered subset.
# - If a filter column/operator is ambiguous, the parser picks the most likely from your sheet’s schema.
