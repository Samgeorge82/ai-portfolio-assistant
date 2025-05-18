import streamlit as st
import pandas as pd
import os

from langchain.docstore.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.tools.tavily_search import TavilySearchResults

# Load API keys from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

# Initialize LLM once
llm = OpenAI()

# UI
st.set_page_config(page_title="AI Portfolio Assistant")
st.title("⚡ AI Portfolio Assistant")
st.markdown("Upload your offshore wind portfolio and ask smart questions. The assistant will search the web if needed.")

uploaded_file = st.file_uploader("Upload your Excel (.xlsx)", type=["xlsx"])
question = st.text_input("Ask a portfolio-related question:")

if uploaded_file and question:
    # Read and format Excel
    df = pd.read_excel(uploaded_file)
    documents = [
        Document(page_content="\n".join([f"{col}: {row[col]}" for col in df.columns]))
        for _, row in df.iterrows()
    ]

    # Embed and build retriever
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 20})
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Step 1: Decide if web search is needed
    planning_prompt = f"""
You are an assistant. Decide if this question requires web search.

Question: {question}

Return 'YES: [search query]' or 'NO'.
"""
    planning_response = llm.invoke(planning_prompt).strip()
    st.markdown(f"**🧠 Planning Response:** `{planning_response}`")

    # Step 2: Search the web if needed
    external_context = ""
    if planning_response.upper().startswith("YES"):
        search_query = planning_response.split(":", 1)[1].strip()
        search = TavilySearchResults(k=5)
        results = search.run(search_query)

        if results:
            external_context = "\n".join([r["content"] for r in results])
            st.markdown("✅ **Web search results retrieved**")
        else:
            external_context = "No relevant external news found."
            st.markdown("⚠️ No useful news found in web search.")

    # Step 3: Get internal data insight
    internal_answer = qa.run(question)

    # Step 4: Combine and reason
    final_prompt = f"""
You are an AI assistant for an offshore wind portfolio analyst. 
You have access to internal project data and real-time external policy/news updates.

Your job is to:
- Answer the user's question with clear reasoning
- Use relevant internal data
- Use external market news if available
- Suggest potential risks, opportunities, or next steps

---

📊 Internal Portfolio Insight:
{internal_answer}

📰 External News Insight:
{external_context}

❓ User Question:
{question}

---

💡 Now provide a clear, strategic answer:
"""
    final_response = llm.invoke(final_prompt)

    # Step 5: Display result
    st.markdown("### 💡 Assistant's Suggestion")
    st.write(final_response)




