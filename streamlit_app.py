"""
Minimal demo UI. Run with: streamlit run streamlit_app.py
Shows the final report plus the critic's log so a viewer can literally see
the retry loop firing — this is the screenshot/GIF you want for LinkedIn
and your portfolio.
"""
import streamlit as st
from app.graph import run

st.set_page_config(page_title="Multi-Agent Research Assistant", layout="wide")
st.title("Multi-Agent Research Assistant")
st.caption("Planner → Researcher → Critic (retry loop) → Writer, built on LangGraph")

query = st.text_input("Research query", placeholder="e.g. What are the tradeoffs of RAG vs fine-tuning?")

if st.button("Run") and query:
    with st.spinner("Running agent graph..."):
        result = run(query)

    st.subheader("Report")
    st.markdown(result["final_report"])

    st.subheader("Critic log (shows the retry loop in action)")
    for entry in result["critic_log"]:
        icon = "✅" if entry["verdict"] == "approve" else "🔁"
        st.write(f"{icon} **{entry['sub_question']}** (attempt {entry['attempt']}): {entry['reason']}")
