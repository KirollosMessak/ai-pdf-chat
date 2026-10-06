# 📄 Chat with your PDFs: AI Document Assistant

Upload PDFs (contracts, policies, manuals, reports) and ask questions in plain language. Answers come back **with page citations**, so you can check every claim against the source.

> **Live demo:** _add your Streamlit Cloud link here_
>
> ![Demo](docs/demo.gif) <!-- record a short GIF and save it as docs/demo.gif -->

## The problem it solves

Teams waste hours searching long documents for one answer. This app lets anyone ask:

- *"What's our refund window?"*
- *"Does the warranty cover water damage?"*
- *"Summarize the termination clause in this contract."*

…and get a sourced answer in seconds. It also works in Arabic and other languages.

## Features

- 📚 **Multiple PDFs at once**: ask questions across all of them
- 🔎 **Retrieval-Augmented Generation (RAG)**: only the most relevant excerpts are sent to the model, which keeps costs low and answers grounded
- 📌 **Page citations**: every answer cites `[file.pdf, p. N]`, with an expandable "Sources used" panel
- 🚫 **No hallucinated answers**: if the documents don't contain the answer, it says so
- ⚡ **Streaming responses**: text appears as it's generated
- 💬 **Follow-up questions**: keeps the conversation context

## How it works

```
PDF ──► extract text per page ──► split into overlapping chunks ──► BM25 index
                                                                       │
Question ──► retrieve top-k relevant chunks ◄──────────────────────────┘
                     │
                     ▼
        Claude (with strict "answer only from sources" prompt)
                     │
                     ▼
         Streamed answer with [file, page] citations
```

| Layer | Tech |
|---|---|
| UI | Streamlit |
| PDF parsing | pypdf |
| Retrieval | BM25 (`rank-bm25`), no vector DB or embedding API needed |
| LLM | Claude via the official Anthropic Python SDK |

## Run it locally

```bash
git clone https://github.com/KirollosMessak/ai-pdf-chat.git
cd ai-pdf-chat
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then paste your [Anthropic API key](https://console.anthropic.com) in the sidebar, or set it once:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then edit the key
```

Try it with the included sample: `sample/company_policy.pdf`.

## Deploy for free (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), click **New app**, and pick this repo with `app.py`.
3. Optional: add `ANTHROPIC_API_KEY` under **App settings → Secrets**. If you leave it out, visitors enter their own key.

## Tests

```bash
pip install pytest
pytest
```

## Ideas to extend

- Vector embeddings (e.g. pgvector or Chroma) for semantic search on large document sets
- OCR for scanned PDFs
- WhatsApp or Telegram bot interface
- User accounts and saved document libraries

## About me

I build AI chatbots, document-processing tools, and automations for businesses.
**Need something like this for your company?** Reach out via [GitHub](https://github.com/KirollosMessak).

## License

MIT
