# Memory-Powered AI Chatbot

This project is a FastAPI chatbot built with LangGraph, LangChain, and SQLite-backed persistent memory. It supports provider-based LLM configuration and is set up to use Groq by default. Each request reloads the stored conversation for a `session_id`, builds a bounded prompt context (sliding window or summary + window), invokes the model, persists the new turn, and returns the updated history.
## Architecture

![System Architecture](image/its_image.png)
## Requirements

- Python 3.12
- A Groq API key for the default setup

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Update `.env` and set a valid `GROQ_API_KEY`.

The application validates required configuration during startup. If the API key for the selected provider is missing, startup exits with a clear error message instead of waiting until the first chat request.

## Run the API

```bash
. .venv/bin/activate
uvicorn main:app --reload
```

You can also run:

```bash
. .venv/bin/activate
python main.py
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

## Run Tests

```bash
. .venv/bin/activate
pytest
```

## Memory Controls

- `MEMORY_STRATEGY` controls context assembly (`sliding_window` or `summary_window`).
- `MEMORY_WINDOW_SIZE` controls how many recent message pairs are kept verbatim.
- `MAX_CONTEXT_TOKENS` sets a rough context cap using `chars // 4` token estimation.
- The system prompt is always placed at index `0` in the model input.
- For `summary_window`, rolling summary state is persisted in SQLite (`chat_summaries`) so it survives restarts.

## Example Chat Request

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user-123",
    "message": "Hi, my name is Hasib."
  }'
```

Second turn with the same session:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user-123",
    "message": "What is my name?"
  }'
```

Because the same `session_id` is reused, the application loads the earlier conversation from SQLite before calling the model again.

## Validation and Error Handling

- `session_id` must be a non-empty string and is limited to 255 characters.
- `message` must be non-empty and is limited to 8000 characters.
- Invalid requests return a consistent error shape:

```json
{
  "error": {
    "code": "validation_error",
    "message": "session_id: Value error, must not be blank"
  }
}
```

## Project Structure

```text
app/
  __init__.py
  config.py
  context_window.py
  graph.py
  llm.py
  main.py
  memory.py
  schemas.py
  state.py
tests/
  test_api.py
  test_graph.py
  test_memory.py
main.py
README.md
.env.example
requirements.txt
```

## Request Flow

1. The client sends `session_id` and `message` to `POST /chat`.
2. FastAPI receives the request in `app/main.py`.
3. The LangGraph `StateGraph` invokes `process_message`.
4. `process_message` loads persistent history and summary state from SQLite.
5. A bounded model context is assembled using the configured memory strategy and token cap.
6. The system prompt is always placed at position `0`, then recent/summary context and the new user message are appended.
7. The configured LLM provider receives this bounded context.
8. The assistant reply is appended and written back to SQLite.
9. The API returns the assistant reply and the updated history.

## Persistence Details

- Chat history is stored in SQLite at `data/chat_memory.db` by default.
- Each stored message contains `session_id`, `role`, `content`, and a timestamp.
- Messages are loaded in insertion order for each session.
- System, human, and AI messages are persisted so later turns can reuse context.
- Summary mode persists rolling summaries in `chat_summaries` with `summarized_upto_message_id`.
