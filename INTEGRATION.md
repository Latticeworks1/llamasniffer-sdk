# Integration Guide

## Using LlamaSniffer as an Ollama Replacement

LlamaSniffer is designed as a **drop-in replacement** for the standard Ollama Python client, with added distributed intelligence and semantic routing.

### Basic Replacement

**Before (Standard Ollama):**
```python
import ollama

client = ollama.Client(host='http://localhost:11434')
response = client.chat(
    model='llama3',
    messages=[{'role': 'user', 'content': 'Hello'}]
)
```

**After (LlamaSniffer):**
```python
from llamasniffer import ollama

# No need to specify host - auto-discovers global instances
ollama.configure(shodan_api_key='your-key')

response = ollama.chat(
    model='llama3',  # Or semantic: 'reasoning', 'coding'
    messages=[{'role': 'user', 'content': 'Hello'}]
)
```

### LangChain Integration

```python
from langchain_community.llms import Ollama
from llamasniffer import ollama as llamasniffer_ollama

# Configure LlamaSniffer backend
llamasniffer_ollama.configure(shodan_api_key='your-key')

# LangChain will use LlamaSniffer's distributed routing
llm = Ollama(
    model="llama3",
    base_url="http://localhost:11434"  # Points to any discovered instance
)

response = llm("What is quantum computing?")
```

### LiteLLM Integration

```python
import litellm
from llamasniffer import ollama

# Configure LlamaSniffer
ollama.configure(shodan_api_key='your-key')

# LiteLLM routing through LlamaSniffer's flock
response = litellm.completion(
    model="ollama/llama3",
    messages=[{"role": "user", "content": "Hello"}],
    api_base="http://localhost:11434"  # Any discovered instance
)
```

### OpenAI SDK Pattern

Use the familiar OpenAI SDK pattern:

```python
from llamasniffer import ollama

# Configure global flock
ollama.configure(shodan_api_key='your-key')

# OpenAI-style usage
class LlamaSnifferClient:
    """OpenAI-compatible wrapper for LlamaSniffer"""

    class Chat:
        class Completions:
            @staticmethod
            def create(model, messages, **kwargs):
                response = ollama.chat(
                    model=model,
                    messages=messages,
                    **kwargs
                )
                # Transform to OpenAI format if needed
                return response

    def __init__(self):
        self.chat = self.Chat()

client = LlamaSnifferClient()
response = client.chat.completions.create(
    model="reasoning",  # Semantic model selection
    messages=[{"role": "user", "content": "Explain AI"}]
)
```

### Environment-Based Configuration

Set environment variables for zero-configuration usage:

```bash
# ~/.bashrc or ~/.zshrc
export SHODAN_API_KEY="your-shodan-api-key"
export HF_TOKEN="your-huggingface-token"
export LLAMASNIFFER_STRATEGY="fastest"  # or "least_loaded", "round_robin"
export LLAMASNIFFER_SEMANTIC="true"
```

Then use without setup:

```python
from llamasniffer import ollama

# Auto-configures from environment
response = ollama.chat(
    model='coding',
    messages=[{'role': 'user', 'content': 'Write Python code'}]
)
```

### Manual Instance Configuration

Skip Shodan and use manual instances:

```python
from llamasniffer import ollama

ollama.configure(
    instances=[
        {"host": "192.168.1.100", "port": 11434},
        {"host": "192.168.1.101", "port": 11434},
        {"host": "example.com", "port": 8080},
    ],
    strategy="least_loaded"
)

# Now routes across your specified instances
response = ollama.generate(model='llama3', prompt='Hello')
```

### API Server Mode (Future)

While LlamaSniffer is currently a client library, you can create a simple server:

```python
from fastapi import FastAPI
from llamasniffer import ollama

app = FastAPI()

ollama.configure(shodan_api_key='your-key')

@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    response = ollama.chat(
        model=request['model'],
        messages=request['messages']
    )
    return response

@app.get("/v1/models")
async def list_models():
    return ollama.list()

# uvicorn server:app --host 0.0.0.0 --port 8000
```

Now any OpenAI-compatible client can connect to `http://localhost:8000`.

### Comparison with Alternatives

| Feature | Standard Ollama | LM Studio | LlamaSniffer |
|---------|----------------|-----------|--------------|
| Local models | ✅ | ✅ | ✅ |
| Remote discovery | ❌ | ❌ | ✅ Shodan |
| Multi-instance | ❌ | ❌ | ✅ Global flock |
| Semantic routing | ❌ | ❌ | ✅ Intent-based |
| Load balancing | ❌ | ❌ | ✅ Automatic |
| Parallel execution | ❌ | ❌ | ✅ Task queue |
| Dataset generation | ❌ | ❌ | ✅ Built-in |
| Drop-in compatible | N/A | ❌ | ✅ Ollama API |

### Best Practices

1. **Start with Ollama compatibility**
   ```python
   from llamasniffer import ollama  # Drop-in replacement
   ```

2. **Add semantic routing when ready**
   ```python
   ollama.configure(semantic_enabled=True)
   ```

3. **Scale to global discovery**
   ```python
   ollama.configure(shodan_api_key='key')
   ```

4. **Monitor your flock**
   ```python
   status = ollama.get_flock_status()
   print(f"Health: {status['flock_health']['health_percentage']}%")
   ```

### Migration Checklist

- [ ] Replace `import ollama` with `from llamasniffer import ollama`
- [ ] Set `SHODAN_API_KEY` environment variable
- [ ] Test basic chat/generate calls
- [ ] Enable semantic routing if desired
- [ ] Monitor flock health
- [ ] Configure load balancing strategy
- [ ] Set up dataset generation (optional)

### Support

For integration issues or questions:
- GitHub Issues: https://github.com/Latticeworks1/llamasniffer-sdk/issues
- Documentation: https://github.com/Latticeworks1/llamasniffer-sdk
