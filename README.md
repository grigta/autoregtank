SelfTalk: Self-dialogue generator using Mistral API

Overview
- A Python 3.11+ CLI tool that leverages the Mistral Chat Completions API to iteratively improve answers via self-dialogue.
- Modes:
  - critic: solver -> critic -> revise loop
  - debate: alternating pro/con agents with final synthesis

Install
1) Ensure Python 3.11+
2) Create a virtual environment and install
   - With pip:
     - pip install -e .[dev]
3) Configure environment
   - Copy .env.example to .env and set MISTRAL_API_KEY

Usage
- Basic run (critic mode):
  selftalk run --system-prompt "You are a helpful assistant." --goal "Explain gravity" --iterations 3 --model mistral-large-latest --temperature 0.3 --max-tokens 1024 --out transcript.jsonl --result result.txt

- Read system prompt from file:
  selftalk run --system-prompt prompt.txt --goal "Summarize the article" --iterations 2

- Debate mode:
  selftalk run --system-prompt "You are a helpful assistant." --goal "Outline the best testing strategy" --mode debate --iterations 3

Environment
- MISTRAL_API_KEY must be set in your environment or .env file. If missing, the CLI exits with a helpful error.

Outputs
- Transcript JSONL: one JSON object per line with fields {ts, role, content, model, iteration}
- Result text: final answer

Testing
- Tests are written with pytest and use fakes/mocks to avoid network calls. Run:
  pytest

Notes
- The Mistral API is called at https://api.mistral.ai/v1/chat/completions
- Simple exponential backoff is implemented for 429/5xx responses
- Optional random seed can be provided for deterministic responses (if supported by the model)

Limitations
- Streaming responses are not implemented in this version
- Debate mode is a simple two-agent alternation with a final synthesis; it is not a full multi-agent framework
