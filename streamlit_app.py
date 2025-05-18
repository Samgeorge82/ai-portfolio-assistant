import streamlit as st
import pandas as pd
import os

from langchain.docstore.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chat_models import ChatOpenAI  # ✅ Correct model class

# Set API keys
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3)  # ✅ Use OpenAI Chat API

st.set_page_config(page_title="AI Portfolio Assistant")
st.title("⚡ AI Portfolio Assistant")
st.markdown("Upload your offshore wind portfolio and ask strategic questions. The assistant will check your project data, search the web if needed, and match findings to affected projects.")

uploaded_file = st.file_uploader("Upload your Excel (.xlsx)", type=["xlsx"])
question = st.text_input("Ask a portfolio-related question:")

if uploaded_file and question:
    # Read Excel
    df = pd.read_excel(uploaded_file)

    # Convert rows into LangChain Documents
    documents = [
        Document(page_content="\n".join([f"{col}: {row[col]}" for col in df.columns]))
        for _, row in df.iterrows()
    ]

    # Set up vector store and retriever
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 20})
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Step 1: Decide if web search is needed
    planning_prompt = f"""
You are an assistant. Decide if this question requires web search. Always focus on offshore wind energy context.

Question: {question}

Return 'YES: [search query]' or 'NO'.
"""
    planning_response = llm.invoke(planning_prompt).content.strip()
    st.markdown(f"**🧠 Planning Response:** `{planning_response}`")

    external_context = ""
    sources_display = ""
    sources_list = ""

    # Step 2: Run Tavily web search if needed
    if planning_response.upper().startswith("YES"):
        search_query = planning_response.split(":", 1)[1].strip()
        if "offshore wind" not in search_query.lower():
            search_query += " offshore wind"

        search = TavilySearchResults(k=5)
        results = search.run(search_query)

        if results:
            external_context = "\n".join([r["content"] for r in results])
            sources_display = "\n".join([f"- [{r['title']}]({r['url']})" for r in results])
            sources_list = "\n".join([r["url"] for r in results])

            st.markdown("✅ **Web search results retrieved**")
            st.markdown("#### 🔗 Sources")
            st.markdown(sources_display)
        else:
            external_context = "No relevant external news found."
            st.markdown("⚠️ No useful news found in web search.")

    # Step 3: Answer using internal project data
    internal_answer = qa.run(question)

    # Step 4: Match potentially affected projects
    affected_projects = []
    for _, row in df.iterrows():
        if any(
            word in question.lower()
            for word in [str(row.get("Country", "")).lower(), str(row.get("Region", "")).lower()]
        ):
            affected_projects.append(str(row["Name"]))

    matched_summary = ", ".join(affected_projects) if affected_projects else "None identified"

    # Step 5: Final reasoning prompt
    final_prompt = f"""
You are an AI assistant for an offshore wind portfolio analyst.

You have:
1. Internal project data (queried insights)
2. Real-time market/policy news
3. A matched list of potentially affected projects

Your job:
- Answer the user's question clearly
- Use insights from internal data and external news
- Mention specific affected projects if relevant
- End with a 🔔 Suggested Actions section (bulleted)
- If useful, cite the provided URLs

---

📊 Internal Portfolio Insight:
{internal_answer}

📰 External News Insight:
{external_context}

🔗 Sources:
{sources_list}

📍 Potentially Affected Projects:
{matched_summary}

❓ User Question:
{question}

---

💡 Strategic Answer:
"""

    final_response = llm.invoke(final_prompt).content

    # Display response
    st.markdown("### 💡 Assistant's Suggestion")
    st.write(final_response)

