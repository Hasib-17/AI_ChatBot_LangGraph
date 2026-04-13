# Memory-Powered AI Chatbot

This project is a FastAPI chatbot built with LangGraph, LangChain, and SQLite-backed persistent memory. It supports provider-based LLM configuration and is set up to use Groq by default. Each request reloads the stored conversation for a `session_id`, appends the new user message, invokes the model with the conversation context, persists the assistant reply, and returns the updated history.
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
4. `process_message` loads the full session history from SQLite.
5. On a new session, a system prompt is inserted first.
6. The new human message is appended.
7. The configured LLM provider receives the conversation history.
8. The assistant reply is appended and written back to SQLite.
9. The API returns the assistant reply and the updated history.

## Persistence Details

- Chat history is stored in SQLite at `data/chat_memory.db` by default.
- Each stored message contains `session_id`, `role`, `content`, and a timestamp.
- Messages are loaded in insertion order for each session.
- System, human, and AI messages are persisted so later turns can reuse context.
