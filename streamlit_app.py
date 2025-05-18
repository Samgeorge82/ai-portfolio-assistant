import streamlit as st
import pandas as pd
import os

from langchain.docstore.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.tools.tavily_search import TavilySearchResults

# Set API keys from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

st.title("⚡ AI Portfolio Assistant")
st.markdown("Upload your offshore wind portfolio and ask smart questions. The assistant will search the web if needed.")

uploaded_file = st.file_uploader("Upload your Excel (.xlsx)", type=["xlsx"])
question = st.text_input("Ask a portfolio-related question:")

if uploaded_file and question:
    df = pd.read_excel(uploaded_file)

    # Convert Excel rows into documents
    documents = []
    for _, row in df.iterrows():
        content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        documents.append(Document(page_content=content))

    # Set up embeddings and retriever
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 20})
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)

    # Decide if web search is needed
    planning_prompt = f"You are an assistant. Decide if this question requires web search:\n\nQuestion: {question}\n\nReturn YES: [search term] or NO."
    planning_response = OpenAI().invoke(planning_prompt)
    st.markdown(f"**Planning Response:** {planning_response}")

    external_context = ""
    if planning_response.startswith("YES"):
        search_query = planning_response.split(":", 1)[1].strip()
        search = TavilySearchResults(k=5)
        results = search.run(search_query)
        external_context = "\n".join([r["content"] for r in results])
        st.markdown("✅ Web search results retrieved.")

    # Internal portfolio context
    internal_answer = qa.run(question)

    # Final answer
    final_prompt = f"""
    You are a smart assistant helping manage an offshore wind portfolio.

    Internal Portfolio Insight:
    {internal_answer}

    External News Insight:
    {external_context}

    Based on both, answer the question:
    {question}
    """

    final_response = OpenAI().invoke(final_prompt)
    st.markdown("### 💡 Assistant's Suggestion")
    st.write(final_response)
