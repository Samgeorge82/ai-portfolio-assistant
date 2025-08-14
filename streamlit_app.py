import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
from typing import List, Dict

# LangChain (modern imports)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Web search (Tavily)
from langchain_community.tools.tavily_search import TavilySearchResults

# -------------------- Setup --------------------
st.set_page_config(page_title="AI Portfolio Assistant - Ask DGX", layout="wide")
st.title("⚡ Ask DGX — AI Portfolio Assistant")
st.caption("Grounded Q&A over your DGX export + optional curated web context (offshore wind only).")

# Secrets / Keys
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

# LLMs / Embeddings (use cheap defaults; turn temp down for determinism)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)  # for planning & filtering
emb = OpenAIEmbeddings(model="text-embedding-3-large")

# -------------------- UI --------------------
st.markdown("Upload your portfolio Excel and ask a question. We’ll search **your data first**, then add **2024–2025 offshore-wind** news only if necessary.")

uploaded_file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
question = st.text_input("Ask a portfolio-related question")

# Allow user to map columns
with st.expander("Optional: map your columns"):
    name_col = st.text_input("Project Name column (default: Name)", value="Name")
    country_col = st.text_input("Country column (default: Country)", value="Country")
    region_col = st.text_input("Region/Market column (default: Region)", value="Region")

# -------------------- Helpers --------------------
@st.cache_data(show_spinner=False)
def read_excel_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(file_bytes)

def df_to_documents(df: pd.DataFrame) -> List[Document]:
    # Chunk rows into ~1k–1.5k char docs for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    docs = []
    for idx, row in df.iterrows():
        # Coerce everything to string and build a row record
        pairs = [f"{col}: {str(row[col])}" for col in df.columns]
        row_text = "\n".join(pairs)
        chunks = text_splitter.split_text(row_text)
        for c in chunks:
            docs.append(Document(page_content=c, metadata={"row_index": int(idx)}))
    return docs

@st.cache_resource(show_spinner=False)
def build_vectorstore(documents: List[Document]) -> FAISS:
    return FAISS.from_documents(documents, emb)

def pick_retriever(vs: FAISS):
    # Multi-query to expand recall + compression to trim fluff
    base = vs.as_retriever(search_kwargs={"k": 8})
    mq = MultiQueryRetriever.from_llm(retriever=base, llm=judge_llm)
    compressor = LLMChainFilter.from_llm(judge_llm)
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=mq)

def extract_entities_for_matching(q: str) -> List[str]:
    """Cheap NER-ish pass to match countries/regions/projects from the question."""
    prompt = """Extract proper nouns and location terms from the question as a JSON array of lowercased strings.
Only include single or two-word terms. No commentary.

Question: {q}
JSON:"""
    out = judge_llm.invoke(prompt.format(q=q)).content.strip()
    try:
        arr = json.loads(out)
        if isinstance(arr, list):
            return [str(x).lower() for x in arr]
    except Exception:
        pass
    return [w.strip().lower() for w in q.replace(",", " ").split() if len(w) > 3]

def decide_search(q: str) -> str:
    """Return '' or a focused offshore-wind query string."""
    plan_prompt = """You are a planner. Decide if the question needs **recent external context**.
Return exactly 'NO' or 'YES: <query>'.

Rules:
- Only say YES if answer likely depends on news/policy/market info from 2024 or 2025.
- Scope strictly to **offshore wind**.
- Prefer specific terms (country, OEM, auction name, cable maker, CfD round, etc.).

Question: {q}
"""
    resp = judge_llm.invoke(plan_prompt.format(q=q)).content.strip()
    if resp.upper().startswith("YES"):
        query = resp.split(":", 1)[1].strip()
        if "offshore wind" not in query.lower():
            query += " offshore wind"
        # force recency tokens
        query += " 2024 2025"
        return query
    return ""

def run_tavily(query: str, k: int = 5) -> List[Dict]:
    if not query:
        return []
    search = TavilySearchResults(k=k, include_domains=None)  # keep broad but we filter by terms
    try:
        res = search.run(query)
        # Normalize results: keep title/url/content and approximate date if present
        cleaned = []
        seen = set()
        for r in res or []:
            url = r.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            item = {
                "title": r.get("title", "Untitled"),
                "url": url,
                "content": (r.get("content") or "")[:2000],  # limit token bloat
            }
            cleaned.append(item)
        return cleaned
    except Exception:
        return []

def format_sources(rows: List[Dict]) -> str:
    if not rows:
        return ""
    lines = []
    for r in rows:
        t = r.get("title", "Source")
        u = r.get("url", "#")
        lines.append(f"- [{t}]({u})")
    return "\n".join(lines)

def match_affected_projects(df: pd.DataFrame, entities: List[str], name_col: str, country_col: str, region_col: str) -> List[str]:
    names = []
    for _, row in df.iterrows():
        fields = [
            str(row.get(name_col, "")).lower(),
            str(row.get(country_col, "")).lower(),
            str(row.get(region_col, "")).lower(),
        ]
        if any(e and e in " ".join(fields) for e in entities):
            nm = str(row.get(name_col, "") or "Unnamed Project")
            if nm:
                names.append(nm)
    return sorted(list(dict.fromkeys(names)))  # de-dupe, stable order

# -------------------- Main --------------------
df = None
if uploaded_file:
    df = read_excel_bytes(uploaded_file)

if df is not None and question:
    with st.spinner("Indexing your data…"):
        docs = df_to_documents(df)
        vs = build_vectorstore(docs)
        retriever = pick_retriever(vs)

    # Internal answer first (grounded, with a narrow prompt)
    qa_prompt = PromptTemplate.from_template(
        """You are a cautious portfolio analyst. Answer ONLY using the provided context.
If the context is insufficient, say what’s missing.

Question: {question}

Context:
{context}

Return a concise, factual answer (no speculation)."""
    )

    def combine_docs_text(docs: List[Document]) -> str:
        return "\n\n".join([d.page_content for d in docs])

    chain = (
        {"context": retriever | combine_docs_text, "question": RunnablePassthrough()}
        | qa_prompt
        | llm
    )

    internal_answer = chain.invoke(question).content.strip()

    # Decide on external search
    search_query = decide_search(question)
    external_rows = run_tavily(search_query) if search_query else []
    external_context = "\n\n".join([r["content"] for r in external_rows])[:6000]
    sources_md = format_sources(external_rows)
    sources_plain = "\n".join([r["url"] for r in external_rows])

    # Affected projects
    entities = extract_entities_for_matching(question)
    affected = match_affected_projects(df, entities, name_col, country_col, region_col)
    affected_summary = ", ".join(affected) if affected else "None identified"

    # Final structured response
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

Internal Context:
{internal_answer}

External News (optional):
{external_context}

Potentially Affected Projects:
{affected_summary}

User Question:
{question}

Sources (plain):
{sources_plain}
"""
    final = llm.invoke(final_prompt).content

    st.markdown("### 💡 Assistant’s Suggestion")
    st.write(final)

    with st.expander("🔗 Sources (clickable)"):
        st.markdown(sources_md if sources_md else "None")

    with st.expander("🧪 Debug"):
        st.code(f"Planner query: {search_query or 'NO'}")
        st.code(f"Entities: {entities}")
        st.code(f"Affected: {affected_summary}")
else:
    st.info("Upload an Excel and ask a question to begin.")

