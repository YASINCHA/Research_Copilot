"""
Run with: streamlit run app/main.py
"""
import asyncio
import streamlit as st
from dotenv import load_dotenv
from app.graph import build_graph

load_dotenv()

st.set_page_config(page_title="AI Research & Meeting Co-Pilot", page_icon="🧭")
st.title("🧭 AI Research & Meeting Co-Pilot")
st.caption("Scope → Research → Write → Critic (multi-agent, self-revising)")

topic = st.text_input("Topic, company, or subject to research")
purpose = st.text_input(
    "What's this for?", placeholder="e.g. sales call prep, personal research, interview prep"
)

if st.button("Generate brief", type="primary"):
    if not topic or not purpose:
        st.warning("Fill in both the topic and the purpose first.")
    else:
        graph = build_graph()
        with st.spinner("Agents working... (scoping, researching, writing, critiquing)"):
            result = asyncio.run(
                graph.ainvoke(
                    {
                        "topic": topic,
                        "purpose": purpose,
                        "revision_count": 0,
                        "max_revisions": 3,
                    }
                )
            )

        st.subheader("Research scope")
        st.json(result["scope"].model_dump())

        st.subheader("Findings")
        st.json(result["findings"].model_dump())

        st.subheader("Final brief")
        st.write(result["final_output"])

        st.caption(
            f"Critic approved after {result['revision_count']} revision(s)."
        )
