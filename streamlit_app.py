# streamlit_app.py
import os
import json
import hashlib
from typing import List, Dict

import streamlit as st
import pandas as pd

# -------------------- Robust imports (works with modern & legacy LC) --------------------
# Try modern "langchain-openai"; fall back to legacy paths if not installed.
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # modern package
    _LC_OPENAI_MODE = "modern"
except ModuleNotFoundError:
    from langchain.chat_models import ChatOpenAI                 # legacy
    from langchain_community.embeddings import OpenAIEmbeddings
    _LC_OPENAI_MODE = "legacy"

# Common LC bits (versions differ; guard optional retrievers below)
from langchain_community.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Optional retriever enhancements (available on LC 0.2+)
try:
    from langchain.retrievers.multi_query import MultiQueryRetriever
    from langchain.retrievers.document_compressors import LLMChainFilter
    from langchain.retrievers import ContextualCompressionRetriever
    _HAS_MQR = True
except Exception:
    _HAS_MQR = False

# Prompting / chaining (APIs changed across versions; keep usage simple)
try:
    from langchain.prompts import PromptTemplate
    from langchain.schema.runnable import RunnablePassthrough
    _HAS_RUNNABLE = True
except Exception:
    _HAS_RUNNABLE = False

# Tavily search (optional)
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    _HAS_TAVILY = True
except Exception:
    _HAS_TAVILY = False

# -------------------- App config --------------------
st.set_page_config(page_title="Ask DGX — AI Portfolio Assistant", layout="wide")
st.title("⚡ Ask DGX — AI Portfolio Assistant")
st.caption(
    "Grounded Q&A over your DGX export. Searches **your data first** and only adds "
    "**2024–2025 offshore-wind** news if the question actually needs it."
)

# -------------------- Secrets / keys --------------------
# (Works locally and on Streamlit Cloud if you set them in Secrets)
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
TAVILY_KEY = st.secrets.get("TAVILY_API_KEY", os.getenv("TAVILY_API_KEY", ""))

if OPENAI_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY
if TAVILY_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_KEY

# -------------------- LLM helpers --------------------
def make_chat_llm(model_pref: str = "gpt-4o-mini", temperature: float = 0.1):
    """
    Create a ChatOpenAI instance that works across modern/legacy LC.
    Modern constructor: ChatOpenAI(model="gpt-4o-mini")
    Legacy constructor: ChatOpenAI(model_name="gpt-4o", temperature=...)
    """
    kwargs = {"temperature": temperature}
    # Modern uses "model"; legacy uses "model_name".
    if _LC_OPENAI_MODE == "modern":
        kwargs["model"] = model_pref
    else:
        # Legacy lib may not know "gpt-4o-mini"; degrade gracefully.
        kwargs["model_name"] = "gpt-4o"
    return ChatOpenAI(**kwargs)

def make_embeddings(model_pref: str = "text-embedding-3-large"):
    if _LC_OPENAI_MODE == "modern":
        return OpenAIEmbeddings(model=model_pref)
    else:
        # Legacy embeddings ignore model kwarg; still fine.
        return OpenAIEmbeddings()

llm = make_chat_llm("gpt-4o-mini", temperature=0.1)
judge_llm = make_chat_llm("gpt-4o-mini", temperature=0.0)
emb = make_embeddings("text-embedding-3-large")

# -------------------- UI --------------------
uploaded_file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
question = st.text_input("Ask a portfolio-related question")

with st.expander("Optional: map your columns"):
    name_col = st.text_input("Project Name column", value="Name")
    country_col = st.text_input("Country column", value="Country")
    region_col = st.text_input("Region/Market column", value="Region")

# -------------------- Utility functions --------------------
def file_sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

@st.cache_data(show_spinner=False)
def read_excel_bytes(b: bytes) -> pd.DataFrame:
    # Keep it simple; openpyxl must be installed
    return pd.read_excel(b)

def df_to_documents(df: pd.DataFrame) -> List[Document]:
    """Chunk each row into ~1.2k char docs for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    docs: List[Document] = []
    for i, row in df.iterrows():
        pairs = [f"{col}: {str(row[col])}" for col in df.columns]
        row_text = "\n".join(pairs)
        for chunk in splitter.split_text(row_text):
            docs.append(Document(page_content=chunk, metadata={"row_index": int(i)}))
    return docs

def build_retriever(vs: FAISS):
    """Use MultiQuery + compression if available; else fall back to base retriever."""
    base = vs.as_retriever(search_kwargs={"k": 8})
    if _HAS_MQR:
        try:
            mq = MultiQueryRetriever.from_llm(retriever=base, llm=judge_llm)
            compressor = LLMChainFilter.from_llm(judge_llm)
            return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=mq)
        except Exception:
            return base
    return base

def extract_entities_for_matching(q: str) -> List[str]:
    """Cheap NER-ish pass using the judge model. Falls back to simple tokenization."""
    prompt = """Extract proper nouns and location terms from the question as a JSON array of lowercased strings.
Only include single or two-word terms. No commentary.

Question: {q}
JSON:"""
    try:
        out = judge_llm.invoke(prompt.format(q=q)).content.strip()
        arr = json.loads(out)
        if isinstance(arr, list):
            return [str(x).lower() for x in arr]
    except Exception:
        pass
    # Fallback: crude tokens
    return [w.strip().lower() for w in q.replace(",", " ").split() if len(w) > 3]

def decide_search(q: str) -> str:
    """Return '' or a focused offshore-wind query string for 2024–2025."""
    plan = """You are a planner. Decide if the question needs **recent external context**.
Return exactly 'NO' or 'YES: <query>'.

Rules:
- Only say YES if answer likely depends on news/policy/market info from 2024 or 2025.
- Scope strictly to **offshore wind**.
- Prefer specific terms (country, OEM, auction name, cable maker, CfD round, etc.).

Question: {q}
"""
    try:
        resp = judge_llm.invoke(plan.format(q=q)).content.strip()
    except Exception:
        return ""
    if resp.upper().startswith("YES"):
        query = resp.split(":", 1)[1].strip()
        if "offshore wind" not in query.lower():
            query += " offshore wind"
        query += " 2024 2025"
        return query
    return ""

def run_tavily(query: str, k: int = 5) -> List[Dict]:
    """Query Tavily if available; return normalized results list."""
    if not query or not _HAS_TAVILY or not TAVILY_KEY:
        return []
    try:
        search = TavilySearchResults(k=k)
        res = search.run(query)
        cleaned, seen = [], set()
        for r in res or []:
            url = r.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            cleaned.append({
                "title": r.get("title", "Untitled"),
                "url": url,
                "content": (r.get("content") or "")[:2000],  # trim token bloat
            })
        return cleaned
    except Exception:
        return []

def format_sources_md(rows: List[Dict]) -> str:
    if not rows:
        return "None"
    return "\n".join(f"- [{r.get('title','Source')}]({r.get('url','#')})" for r in rows)

def match_affected_projects(df: pd.DataFrame, entities: List[str], name_col: str, country_col: str, region_col: str) -> List[str]:
    names: List[str] = []
    for _, row in df.iterrows():
        fields = " ".join([
            str(row.get(name_col, "")).lower(),
            str(row.get(country_col, "")).lower(),
            str(row.get(region_col, "")).lower(),
        ])
        if any(e and e in fields for e in entities):
            nm = str(row.get(name_col, "") or "Unnamed Project")
            if nm:
                names.append(nm)
    # de-dupe, preserve order
    deduped = list(dict.fromkeys(names).keys())
    return deduped

def combine_docs_text(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)

# -------------------- Main flow --------------------
if uploaded_file and question:
    # 1) Read uploaded file bytes (bytes are hashable and cache-friendly)
    file_bytes = uploaded_file.getvalue()
    df = read_excel_bytes(file_bytes)

    # 2) Build & memoize the vectorstore per file hash (avoid caching unhashables)
    file_key = file_sha256(file_bytes)
    if "vs_cache" not in st.session_state:
        st.session_state.vs_cache = {}

    if file_key not in st.session_state.vs_cache:
        with st.spinner("Indexing your data…"):
            docs = df_to_documents(df)
            vs = FAISS.from_documents(docs, emb)
            st.session_state.vs_cache[file_key] = vs
    else:
        vs = st.session_state.vs_cache[file_key]

    retriever = build_retriever(vs)

    # 3) Internal grounded answer (keep it strict; no speculation)
    internal_context_docs = retriever.get_relevant_documents(question)
    internal_context = combine_docs_text(internal_context_docs) if internal_context_docs else ""

    base_answer_prompt = (
        "You are a cautious offshore wind portfolio analyst. "
        "Answer ONLY using the provided internal context. "
        "If the context is insufficient, state what is missing.\n\n"
        f"Question:\n{question}\n\n"
        f"Internal Context:\n{internal_context}\n\n"
        "Return a concise, factual answer (no speculation)."
    )
    internal_answer = llm.invoke(base_answer_prompt).content.strip()

    # 4) Decide if web context is needed; run Tavily if so
    search_query = decide_search(question)
    external_rows = run_tavily(search_query) if search_query else []
    external_context = "\n\n".join(r["content"] for r in external_rows)[:6000]
    sources_md = format_sources_md(external_rows)
    sources_plain = "\n".join(r["url"] for r in external_rows) if external_rows else ""

    # 5) Match affected projects using cheap NER terms
    entities = extract_entities_for_matching(question)
    affected = match_affected_projects(df, entities, name_col, country_col, region_col)
    affected_summary = ", ".join(affected) if affected else "None"

    # 6) Final structured response
    final_prompt = f"""
You are an AI assistant for an offshore wind portfolio analyst. Synthesize a grounded answer.

You have:
- Internal DGX-derived context (already curated above)
- External curated news (optional, 2024–2025 only)
- A list of potentially affected projects

Write a response with EXACTLY these sections:

### Answer
- Clear, 5–8 lines max. Cite fields or figures only if seen in internal context. If you reference news, add (see Sources).

### Affected projects
- Brief sentence + a bullet list. If none, say "None".

### Suggested actions 🔔
- 3–5 bullets, each starting with a strong verb, concrete and assignable.
- If context is weak, first bullet should be "Validate data" with what to validate.

### Sources
- If external sources exist, render the list (already markdown formatted). If none, write "None".

Internal Context (grounded answer from retrieval):
{internal_answer}

External News (optional, summarized snippets):
{external_context}

Potentially Affected Projects:
{affected_summary}

User Question:
{question}

Sources (plain):
{sources_plain}
"""
    final = llm.invoke(final_prompt).content

    # 7) Display
    st.markdown("### 💡 Assistant’s Suggestion")
    st.write(final)

    with st.expander("🔗 Sources (clickable)"):
        st.markdown(sources_md)

    with st.expander("🧪 Debug"):
        st.code(f"Planner query: {search_query or 'NO'}")
        st.code(f"Entities: {entities}")
        st.code(f"Affected: {affected_summary}")
        st.code(f"LC mode: {_LC_OPENAI_MODE} | Has MQR: {_HAS_MQR} | Has Runnable: {_HAS_RUNNABLE} | Has Tavily: {_HAS_TAVILY}")

else:
    st.info("Upload an Excel and ask a question to begin.")

