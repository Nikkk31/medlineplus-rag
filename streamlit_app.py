import streamlit as st

import config
from generation.generate import answer_question
from retrieval.hybrid import HybridRetriever

st.set_page_config(page_title="Diabetes & Blood Sugar Assistant", page_icon="🩺")


@st.cache_resource
def get_retriever() -> HybridRetriever:
    """Built once per server process, not once per question."""
    return HybridRetriever()


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("Sources"):
        for s in sources:
            title = s.get("title", "Unknown topic")
            url = s.get("url", "")
            source_name = s.get("source_name", "Unknown")
            if url:
                st.markdown(f"- [{title}]({url}) — {source_name}")
            else:
                st.markdown(f"- {title} — {source_name}")


st.title("🩺 Diabetes & Blood Sugar Assistant")
st.caption(
    f"Answers sourced from {config.CORPUS_SOURCE_NAME}. "
    "Not a substitute for professional medical advice."
)

if "history" not in st.session_state:
    st.session_state.history = []

# Replay prior turns in this session
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        render_sources(turn["sources"])

question = st.chat_input("Ask a question about diabetes or blood sugar...")

if question:
    with st.chat_message("user"):
        st.write(question)

    retriever = get_retriever()

    with st.chat_message("assistant"):
        with st.spinner(
            "Thinking... (first question after startup takes longer "
            "while the model loads and stays loaded for later questions)"
        ):
            result = answer_question(question, retriever=retriever)

        if not result.in_scope:
            st.warning(result.answer)
        else:
            st.write(result.answer)
            render_sources(result.sources)

    st.session_state.history.append(
        {
            "question": question,
            "answer": result.answer,
            "sources": result.sources,
        }
    )

with st.sidebar:
    st.subheader("Backend")
    st.write(f"Generation backend: `{config.GENERATION_BACKEND}`")
    if config.GENERATION_BACKEND == "local":
        st.write(f"Model file: `{config.LOCAL_MODEL_PATH.name}`")
    st.write(f"Top K chunks: `{config.TOP_K}`")
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()