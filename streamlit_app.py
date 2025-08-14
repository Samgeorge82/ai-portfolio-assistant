import streamlit as st
import pandas as pd
import os
import re
from datetime import datetime

from langchain.docstore.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI  # Updated import

# -------------------------------
# SETUP
# -------------------------------
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

st.set_page_config(page_title="AI Portfolio Assistant")
st.title("⚡ AI Portfolio Assistant")
st.markdown("""
Upload your offshore wind portfolio and ask strategic questions.  
The assistant will check your project data, search the web if needed,  
and match findings to affected projects.
""")

# -------------------------------
# FILE UPLOAD + QUESTION INPUT
# -------------------------------
uploaded_file = st.file_uploader("Upload your Excel (.xlsx)", type=["xlsx"])
question = st.text_input("Ask a portfolio-related question:")

# -------------------------------
# SCENARIO HANDLER
# -------------------------------
def detect_steel_price_scenario(q):
    return "steel" in q.lower() and "price" in q.lower()

def extract_percentage(q):
    match = re.search(r"(\d+(\.\d+)?)\s*%", q)
    if match:
        return float(match.group(1))
    return None

def extract_year_range(q):
    years = re.findall(r"(20\d{2})", q)
    years = [int(y) for y in years]
    if len(years) >= 2:
        return min(years), max(years)
    elif len(years) == 1:
        return years[0], years[0]
    else:
        return None, None

def calculate_steel_impact(df, pct_increase, start_year, end_year):
    # Define default steel exposure factors
    weights = {
        "Foundations Capex": 1.0,
        "WTG Capex": 0.25,
        "OSS Capex": 0.5
    }

    # Ensure date column
    if not pd.api.types.is_datetime64_any_dtype(df["COD Target"]):
        df["COD Target"] = pd.to_datetime(df["COD Target"], errors="coerce")

    # Filter by year range
    filtered = df[(df["COD Target"].dt.year >= start_year) &
                  (df["COD Target"].dt.year <= end_year)]

    results = []
    for _, row in filtered.iterrows():
        extra_cost = 0
        for col, weight in weights.items():
            if col in row and pd.notnull(row[col]):
                extra_cost += row[col] * weight * (pct_increase / 100)
        results.append({
            "Project": row.get("Project Name", ""),
            "COD Year": row["COD Target"].year if pd.notnull(row["COD Target"]) else None,
            "Extra Capex (€m)": round(extra_cost, 2)
        })
    return pd.DataFrame(results)

# -------------------------------
# MAIN LOGIC
# -------------------------------
if uploaded_file and question:
    df = pd.read_excel(uploaded_file)

    # Scenario detection: steel price increase
    if detect_steel_price_scenario(question):
        pct_increase = extract_percentage(question)
        start_year, end_year = extract_year_range(question)
        if pct_increase and start_year:
            scenario_df = calculate_steel_impact(df, pct_increase, start_year, end_year)
            st.subheader("📊 Scenario Result")
            st.dataframe(scenario_df)

            # Send to LLM for reasoning output
            llm_context = f"""
User Question: {question}
Here is the calculated extra capex per project:
{scenario_df.to_markdown(index=False)}

Write a clear, concise analysis of the financial impact, highlight the biggest affected projects,
and suggest mitigation actions.
"""
            final_response = llm.invoke(llm_context).content
            st.markdown("### 💡 Assistant's Analysis")
            st.write(final_response)
        else:
            st.warning("Couldn't parse percentage or year range from your question.")
    else:
        # -------------------------------
        # Normal LLM + FAISS + Tavily flow
        # -------------------------------
        documents = [
            Document(page_content="\n".join([f"{col}: {row[col]}" for col in df.columns]))
            for _, row in df.iterrows()
        ]
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(documents, embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 50})
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # Step 1: Decide if web search is needed
        planning_prompt = f"""
You are an assistant. Decide if this question requires web search. 
Always focus on offshore wind energy context and ensure the search targets **recent news from 2024 or 2025**.

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
            search_query += " 2024 2025"

            search = TavilySearchResults(k=5)
            results = search.run(search_query)

            if results:
                external_context = "\n".join([r["content"] for r in results])
                sources_display = "\n".join([
                    f"- [{r.get('title', 'Untitled')}]({r.get('url', '#')})"
                    for r in results if r.get("url")
                ])
                sources_list = "\n".join([
                    r["url"] for r in results if r.get("url")
                ])

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
                for word in [
                    str(row.get("Country", "")).lower(),
                    str(row.get("Region", "")).lower()
                ]
            ):
                affected_projects.append(str(row.get("Name", "Unnamed Project")))

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
- Cite any relevant URLs if useful
- Prioritize **recent updates from 2024 or 2025** and ignore outdated news

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
