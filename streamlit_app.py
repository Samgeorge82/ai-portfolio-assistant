# streamlit_app.py
# Unified app: Excel Q&A with LLM + Web Search  ➜  PLUS Scenario "Steel price ↑ → Capex impact"
# - Robust column mapping UI (works with any sheet naming)
# - Safe numeric coercion (€, commas, 'MEUR' suffixes, etc.)
# - FAISS vectorstore memoized per-file (no Streamlit cache errors)
# - No dependency on 'tabulate' (uses CSV/JSON for LLM context)

import os
import re
import json
import hashlib
from typing import List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
import numpy as np

# ===================== Defensive LangChain/OpenAI imports =====================
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # modern packages
    _LC_MODE = "modern"
except ModuleNotFoundError:
    # fallbacks for older envs
    from langchain.chat_models import ChatOpenAI
    from langchain_community.embeddings import OpenAIEmbeddings
    _LC_MODE = "legacy"

from langchain_community.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    _HAS_TAVILY = True
except Exception:
    _HAS_TAVILY = False

# ===================== App config & keys =====================
st.set_page_config(page_title="⚡ Ask DGX — AI Portfolio Assistant", layout="wide")
st.title("⚡ Ask DGX — AI Portfolio Assistant")
st.caption("Grounded Q&A over your Excel + optional web context. Now with a scenario engine (e.g., “Steel +10% → extra capex by year”).")

OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
TAVILY_KEY = st.secrets.get("TAVILY_API_KEY", os.getenv("TAVILY_API_KEY", ""))

if OPENAI_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY
if TAVILY_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_KEY

def make_llm(model_pref="gpt-4o-mini", temperature=0.1):
    kwargs = {"temperature": temperature}
    if _LC_MODE == "modern":
        kwargs["model"] = model_pref
    else:
        kwargs["model_name"] = "gpt-4o"
    return ChatOpenAI(**kwargs)

llm = make_llm("gpt-4o-mini", 0.1)
judge_llm = make_llm("gpt-4o-mini", 0.0)
emb = OpenAIEmbeddings()

# ===================== Inputs =====================
uploaded_file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
question = st.text_input("Ask a portfolio-related question (e.g., 'projects in construction with COD > 2030' or 'If steel +10%, extra capex 2028–2031?')")

# ===================== Helpers =====================
@st.cache_data(show_spinner=False)
def read_excel_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_excel(b)

def file_sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def money_to_float(series: pd.Series) -> pd.Series:
    x = series.astype(str)
    x = x.str.replace(r"[€,\s]", "", regex=True)
    x = x.str.replace(r"(?i)(meur|eur|m|mn)$", "", regex=True)
    return pd.to_numeric(x, errors="coerce")

def guess_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    cols = list(df.columns)
    norm = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        if cand.lower().strip() in norm:
            return norm[cand.lower().strip()]
    for c in cols:
        lc = c.lower()
        if any(cand.lower() in lc for cand in candidates):
            return c
    return None

def detect_steel_price_scenario(q: str) -> bool:
    ql = q.lower()
    return ("steel" in ql or "stahl" in ql) and ("price" in ql or "preis" in ql or "%" in ql) and ("capex" in ql or "cost" in ql or "kosten" in ql)

def extract_percentage(q: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", q)
    return float(m.group(1)) if m else None

def extract_year_range(q: str) -> Tuple[int | None, int | None]:
    yrs = [int(y) for y in re.findall(r"(20\d{2})", q)]
    if len(yrs) >= 2:
        return min(yrs), max(yrs)
    elif len(yrs) == 1:
        return yrs[0], yrs[0]
    return None, None

def llm_should_search(q: str) -> str:
    plan = f"""You are a planner. Decide if this portfolio question needs recent external web search.
Return exactly 'NO' or 'YES: <query>'. Only say YES for offshore-wind news/policy/supply updates from 2024–2025.
Question: {q}"""
    try:
        return judge_llm.invoke(plan).content.strip()
    except Exception:
        return "NO"

# ===================== Main =====================
if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    df = read_excel_bytes(file_bytes)

    # ---------- Column mapping UI (works with any sheet) ----------
    st.markdown("### 🧭 Map key columns")
    c1, c2, c3, c4 = st.columns(4)
    name_guess = guess_col(df, ["project name","name","asset"])
    cod_guess  = guess_col(df, ["cod target","cod","commercial operation date","cod date"])
    country_guess = guess_col(df, ["country","market","region"])

    name_col = c1.selectbox("Project Name", options=df.columns, index=(df.columns.get_loc(name_guess) if name_guess in df.columns else 0))
    cod_col  = c2.selectbox("COD Target (date)", options=df.columns, index=(df.columns.get_loc(cod_guess) if cod_guess in df.columns else 0))
    country_col = c3.selectbox("Country/Market (optional)", options=["<none>"]+list(df.columns),
                               index=(0 if not country_guess else 1+df.columns.get_loc(country_guess)))
    id_col = c4.selectbox("Project ID (optional)", options=["<none>"]+list(df.columns),
                          index=(0 if not guess_col(df,["project id","id"]) else 1+df.columns.get_loc(guess_col(df,["project id","id"]))))

    st.markdown("#### Capex buckets (MEUR) — map if available")
    b1, b2, b3, b4, b5 = st.columns(5)
    wtg_col = b1.selectbox("WTG", options=["<none>"]+list(df.columns), index=(0 if not guess_col(df,["wtg capex","turbine capex"]) else 1+df.columns.get_loc(guess_col(df,["wtg capex","turbine capex"]))))
    fnd_col = b2.selectbox("Foundations", options=["<none>"]+list(df.columns), index=(0 if not guess_col(df,["foundations capex","foundation capex","monopile","jacket"]) else 1+df.columns.get_loc(guess_col(df,["foundations capex","foundation capex","monopile","jacket"])))) 
    oss_col = b3.selectbox("OSS", options=["<none>"]+list(df.columns), index=(0 if not guess_col(df,["oss capex","substation capex","offshore substation"]) else 1+df.columns.get_loc(guess_col(df,["oss capex","substation capex","offshore substation"]))))
    arr_col = b4.selectbox("Array", options=["<none>"]+list(df.columns), index=(0 if not guess_col(df,["array cable capex","inter-array capex"]) else 1+df.columns.get_loc(guess_col(df,["array cable capex","inter-array capex"]))))
    exp_col = b5.selectbox("Export", options=["<none>"]+list(df.columns), index=(0 if not guess_col(df,["export system capex","export cable capex","grid connection capex"]) else 1+df.columns.get_loc(guess_col(df,["export system capex","export cable capex","grid connection capex"]))))

    # Normalize types
    work = df.copy()
    work[cod_col] = pd.to_datetime(work[cod_col], errors="coerce")
    def col_or_none(sel): return None if sel == "<none>" else sel
    for lbl, sel in [("WTG", wtg_col), ("Foundations", fnd_col), ("OSS", oss_col), ("Array", arr_col), ("Export", exp_col)]:
        dst = f"{lbl} Capex (MEUR)"
        if col_or_none(sel):
            work[dst] = money_to_float(work[sel]).fillna(0.0)
        else:
            work[dst] = 0.0

    with st.expander("🔎 Capex mapping debug"):
        dbg = []
        for lbl in ["WTG","Foundations","OSS","Array","Export"]:
            s = work[f"{lbl} Capex (MEUR)"]
            dbg.append({"Bucket": lbl, "Non-zero rows": int((s > 0).sum()), "Sum (MEUR)": round(float(s.sum()),1)})
        st.dataframe(pd.DataFrame(dbg))

    # ===================== Scenario branch (Steel price → extra capex) =====================
    handled_scenario = False
    if question:
        if detect_steel_price_scenario(question):
            handled_scenario = True
            pct = extract_percentage(question) or 10.0
            y0, y1 = extract_year_range(question)
            work["COD Year"] = work[cod_col].dt.year

            # Default exposure weights (tweak live if you want)
            st.markdown("### ⚙️ Steel exposure weights (tunable)")
            s1,s2,s3,s4,s5 = st.columns(5)
            w_wtg = s1.slider("WTG", 0.0, 1.0, 0.25, 0.05)
            w_fnd = s2.slider("Foundations", 0.0, 1.0, 1.00, 0.05)
            w_oss = s3.slider("OSS", 0.0, 1.0, 0.50, 0.05)
            w_arr = s4.slider("Array", 0.0, 1.0, 0.15, 0.05)
            w_exp = s5.slider("Export", 0.0, 1.0, 0.15, 0.05)

            exposure = (
                work["Foundations Capex (MEUR)"] * w_fnd +
                work["WTG Capex (MEUR)"]         * w_wtg +
                work["OSS Capex (MEUR)"]         * w_oss +
                work["Array Capex (MEUR)"]       * w_arr +
                work["Export Capex (MEUR)"]      * w_exp
            )
            work["Extra Capex (MEUR)"] = (exposure * (pct / 100.0)).round(1)

            out_cols = [name_col, "COD Year", "Extra Capex (MEUR)"]
            if id_col != "<none>": out_cols = [id_col] + out_cols
            scenario_df = work[out_cols].copy()

            if y0:
                scenario_df = scenario_df[(scenario_df["COD Year"] >= y0) & (scenario_df["COD Year"] <= y1)]

            st.markdown("### 📊 Scenario Result — Steel price shock")
            st.dataframe(scenario_df, use_container_width=True)

            # Year aggregation
            st.markdown("#### 📈 Extra capex by year (MEUR)")
            year_sum = scenario_df.groupby("COD Year", dropna=True, as_index=False)["Extra Capex (MEUR)"].sum()
            st.dataframe(year_sum, use_container_width=True)

            # LLM analysis (send CSV, not markdown)
            llm_context = f"""User Question: {question}
Steel price increase: {pct}%
Exposure weights used: WTG={w_wtg}, Foundations={w_fnd}, OSS={w_oss}, Array={w_arr}, Export={w_exp}
Per-project impact (CSV):
{scenario_df.to_csv(index=False)}
Per-year totals (CSV):
{year_sum.to_csv(index=False)}
Write a concise analysis highlighting the biggest affected years/projects and 3–4 mitigation actions."""
            analysis = llm.invoke(llm_context).content
            st.markdown("### 💡 Assistant’s Analysis")
            st.write(analysis)

    # ===================== Default Q&A branch (LLM + FAISS + optional web) =====================
    if question and not handled_scenario:
        # Vectorstore memoized per file (no caching of unhashable objects)
        key = file_sha256(file_bytes)
        if "vs_cache" not in st.session_state:
            st.session_state.vs_cache = {}
        if key not in st.session_state.vs_cache:
            with st.spinner("Indexing your data for semantic Q&A…"):
                docs: List[Document] = []
                for _, row in df.iterrows():
                    pairs = [f"{col}: {str(row[col])}" for col in df.columns]
                    docs.append(Document(page_content="\n".join(pairs)))
                vs = FAISS.from_documents(docs, emb)
                st.session_state.vs_cache[key] = vs
        else:
            vs = st.session_state.vs_cache[key]

        retriever = vs.as_retriever(search_kwargs={"k": 50})
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # Planner: should we web search?
        plan = llm_should_search(question)
        st.markdown(f"**🧠 Planning:** `{plan}`")
        external_context, sources_display, sources_list = "", "", ""

        if plan.upper().startswith("YES") and _HAS_TAVILY and TAVILY_KEY:
            search_query = plan.split(":", 1)[1].strip()
            if "offshore wind" not in search_query.lower():
                search_query += " offshore wind"
            search_query += " 2024 2025"
            try:
                search = TavilySearchResults(k=5)
                results = search.run(search_query)
                if results:
                    external_context = "\n".join([(r.get("content") or "") for r in results])
                    sources_display = "\n".join([f"- [{r.get('title','Untitled')}]({r.get('url','#')})" for r in results if r.get("url")])
                    sources_list = "\n".join([r["url"] for r in results if r.get("url")])
                    st.markdown("✅ **Web results retrieved**")
                    st.markdown("#### 🔗 Sources")
                    st.markdown(sources_display)
            except Exception:
                st.info("Web search not available right now.")

        # Internal answer
        internal_answer = qa.run(question)

        # Lightweight affected-projects guess (by country term match)
        affected = []
        ql = question.lower()
        if country_col != "<none>":
            for _, r in df.iterrows():
                if str(r[country_col]).lower() in ql:
                    affected.append(str(r[name_col]))
        matched_summary = ", ".join(affected) if affected else "None identified"

        final_prompt = f"""You are an assistant for an offshore wind portfolio analyst.

Use ONLY the provided internal summary and optional web context. Be concise, factual, and add a short 🔔 Suggested Actions list.

Internal summary:
{internal_answer}

External context (if any, may be empty):
{external_context}

Sources (URLs, if any):
{sources_list}

Potentially affected projects (quick guess):
{matched_summary}

User question:
{question}

Now write the final answer:
"""
        final_response = llm.invoke(final_prompt).content
        st.markdown("### 💡 Assistant’s Suggestion")
        st.write(final_response)

else:
    st.info("Upload an Excel and ask a question to begin.")
