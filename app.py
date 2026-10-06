"""Chat with your PDFs: a Streamlit RAG app powered by Claude."""

import os

import anthropic
import streamlit as st

from rag import Retriever, extract_pages, format_context, split_chunks

MODEL = "claude-opus-5-5"
SYSTEM_PROMPT = """You answer questions using only the document excerpts provided in <source> tags.

- Cite every fact with its file and page, like [report.pdf, p. 3].
- If the excerpts don't contain the answer, say so plainly. Don't guess or use outside knowledge.
- Answer in the same language the user writes in.
- Keep answers concise and well structured."""

st.set_page_config(page_title="Chat with your PDFs", page_icon="📄")
st.title("📄 Chat with your PDFs")
st.caption("Upload documents, ask questions, get answers with page citations.")


def get_api_key() -> str | None:
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except FileNotFoundError:
        pass
    return os.environ.get("ANTHROPIC_API_KEY")


with st.sidebar:
    st.header("Settings")
    api_key = get_api_key() or st.text_input(
        "Anthropic API key", type="password", help="Get one at console.anthropic.com"
    )
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    top_k = st.slider("Excerpts per question", 2, 12, 6)
    if st.button("Clear chat"):
        st.session_state.messages = []


@st.cache_resource(show_spinner="Indexing documents...")
def build_retriever(file_data: tuple[tuple[str, bytes], ...]) -> Retriever:
    from io import BytesIO

    pages = []
    for name, data in file_data:
        pages.extend(extract_pages(BytesIO(data), name))
    return Retriever(split_chunks(pages))


if not files:
    st.info("Upload one or more PDFs in the sidebar to get started.")
    st.stop()

try:
    retriever = build_retriever(tuple((f.name, f.getvalue()) for f in files))
except ValueError as e:
    st.error(str(e))
    st.stop()

st.sidebar.success(f"Indexed {len(retriever.chunks)} chunks from {len(files)} file(s).")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about your documents")
if not question:
    st.stop()
if not api_key:
    st.warning("Add your Anthropic API key in the sidebar.")
    st.stop()

with st.chat_message("user"):
    st.markdown(question)

hits = retriever.search(question, k=top_k)
# Earlier turns are sent as plain text; only the current question carries excerpts.
history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
user_turn = f"{format_context(hits)}\n\nQuestion: {question}"
st.session_state.messages.append({"role": "user", "content": question})

client = anthropic.Anthropic(api_key=api_key)
with st.chat_message("assistant"):
    try:
        with client.beta.messages.stream(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=history + [{"role": "user", "content": user_turn}],
        ) as stream:
            answer = st.write_stream(stream.text_stream)
            final = stream.get_final_message()
        if final.stop_reason == "refusal":
            answer = "Sorry, I can't help with that request."
            st.warning(answer)
    except anthropic.AuthenticationError:
        answer = None
        st.error("Invalid API key. Check the key in the sidebar.")
    except anthropic.RateLimitError:
        answer = None
        st.error("Rate limit reached. Wait a moment and try again.")
    except anthropic.APIStatusError as e:
        answer = None
        st.error(f"API error ({e.status_code}): {e.message}")
    except anthropic.APIConnectionError:
        answer = None
        st.error("Couldn't reach the Anthropic API. Check your connection.")

    with st.expander("Sources used"):
        for c in hits:
            st.markdown(f"**{c.source}, p. {c.page}**")
            st.caption(c.text[:400] + ("..." if len(c.text) > 400 else ""))

# Keep only successful exchanges in the history sent to the model.
if answer:
    st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.session_state.messages.pop()
