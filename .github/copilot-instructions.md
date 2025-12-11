# AI Agents Crash Course - Copilot Instructions

## Project Overview
Educational repository for building production-ready AI agents using OpenAI's Agents SDK. Core example: nutrition assistant chatbot with RAG, MCP integration, multi-agent orchestration, and guardrails.

## Architecture

### Three-Tier Structure
1. **Notebooks** (`notebooks/`, `notebooks_complete/`) - Interactive learning modules
2. **Chatbot Apps** (`chatbot/`, `chatbot_complete/`, `multi_agent_chatbot/`) - Chainlit web interfaces
3. **ChatKit** (`chatkit/`) - Next.js production deployment using OpenAI ChatKit

### Data Flow
- **ChromaDB** (`chroma/` directory) - Persistent vector store for nutrition database
- **CSV Source** (`data/calories.csv`) - Converted via `rag_setup/create_calorie_database.py`
- **RAG Pattern** - All agents use `calorie_lookup_tool` querying ChromaDB with semantic search

## Critical Patterns

### Agent Creation (OpenAI Agents SDK)
```python
from agents import Agent, Runner, function_tool, trace

# Always use @function_tool decorator for tools
@function_tool
def calorie_lookup_tool(query: str, max_results: int = 3) -> str:
    """Docstring becomes tool description for LLM"""
    results = nutrition_db.query(query_texts=[query], n_results=max_results)
    # Return formatted string, not raw data
    return "Nutrition Information:\n" + formatted_results

agent = Agent(
    name="Nutrition Assistant",
    instructions="Detailed prompt here",  # Critical: be specific
    tools=[calorie_lookup_tool],
    mcp_servers=[exa_search_mcp]  # Optional MCP integration
)
```

### Execution Patterns
- **Simple run**: `result = await Runner.run(agent, "query")`
- **Streaming**: `result = Runner.run_streamed(agent, "query")` + iterate `stream_events()`
- **With memory**: `Runner.run(agent, query, session=SQLiteSession("db_name"))`
- **Tracing**: Wrap in `with trace("description"):` for OpenAI dashboard visibility

### Guardrails Implementation
```python
from agents import input_guardrail, GuardrailFunctionOutput

@input_guardrail
async def food_topic_guardrail(ctx, agent, input):
    # Use separate agent to validate input
    result = await Runner.run(guardrail_agent, input, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=(not result.final_output.only_about_food)
    )

agent = Agent(..., input_guardrails=[food_topic_guardrail])
```

### MCP Integration (Model Context Protocol)
```python
from agents.mcp import MCPServerStreamableHttp

# Exa Search example - URL includes API key as query param
exa_search_mcp = MCPServerStreamableHttp(
    name="Exa Search MCP",
    params={"url": f"https://mcp.exa.ai/mcp?{os.environ.get('EXA_API_KEY')}", "timeout": 30},
    cache_tools_list=True
)
await exa_search_mcp.connect()  # Required in async context
```

### Multi-Agent Orchestration
See `multi_agent_chatbot/nutrition_agent.py`:
- Define specialized agents (calorie lookup vs. meal planning)
- Use `Agent.transfer_to_agent` for handoffs
- Main orchestrator agent routes between specialists

## Environment Setup

### Required `.env` Variables
```bash
OPENAI_API_KEY=sk-...           # Primary API key
OPENAI_DEFAULT_MODEL=gpt-4o     # Model for all agents
EXA_API_KEY=...                 # For web search MCP
CHAINLIT_AUTH_SECRET=...        # Generate: chainlit create-secret
CHAINLIT_USERNAME=...           # For password auth
CHAINLIT_PASSWORD=...           # For password auth
```

### Dependencies
- Install: `pip install -r requirements.txt` (or `requirements_windows.txt` on Windows)
- Key packages: `agents` (OpenAI SDK), `chainlit`, `chromadb`, `pydantic`

## Development Workflows

### Running Notebooks
1. Ensure ChromaDB populated: run `rag_setup/rag_setup.ipynb` first
2. Notebooks are async-ready (use `await` for Runner calls)
3. Progression: `simplest_agent.ipynb` → `tool_calling.ipynb` → `rag.ipynb` → `mcp.ipynb` → `multi_agent.ipynb` → `guardrails.ipynb`

### Running Chainlit Apps
```bash
cd chatbot_complete
chainlit run 4_authentication.py --port 10000 --host 0.0.0.0
```
- Streams agent responses token-by-token via `ResponseTextDeltaEvent`
- Displays tool calls in UI via `cl.Step(name=tool_name, type="tool")`
- Sessions stored in `SQLiteSession("conversation_history")`

### ChatKit Deployment (Next.js)
```bash
cd chatkit
npm install
npm run dev  # Unsets OPENAI_API_KEY env (proxied via API route)
```
- API route: `app/api/create-session/route.ts` - Proxies OpenAI ChatKit sessions
- Requires `WORKFLOW_ID` in `lib/config.ts` (from OpenAI Agent Builder)

## Common Tasks

### Adding a New Tool
1. Define function with `@function_tool` decorator
2. Add comprehensive docstring (becomes LLM instructions)
3. Return string or structured data (Pydantic model)
4. Add to agent's `tools=[]` list

### Debugging Agent Behavior
- Check OpenAI Traces: https://platform.openai.com/logs?api=traces
- Use `with trace("descriptive_name"):` wrapper for logging
- Inspect `result.final_output` vs. intermediate tool calls

### Updating ChromaDB
1. Modify `data/calories.csv`
2. Run `rag_setup/create_calorie_database.py` or full `rag_setup.ipynb`
3. ChromaDB auto-persists to `chroma/` directory

## Project Conventions

- **Incomplete code in `chatbot/`** - Students fill in gaps; solutions in `chatbot_complete/`
- **ChromaDB path resolution**: Use `Path(__file__).parent.parent / "chroma"` for relative paths
- **Async everywhere**: All agent runs are `async`/`await` due to OpenAI SDK design
- **Streaming in Chainlit**: Always use `Runner.run_streamed()` for real-time UX
- **Authentication**: Chainlit password auth via `@cl.password_auth_callback` decorator

## Key Files
- `nutrition_agent.py` (in each chatbot dir) - Core agent definition + tools
- `COURSE_RESOURCES.md` - Student reference for commands and URLs
- `notebooks_complete/` - Canonical solutions for all exercises
- `.env.template` - All required environment variables (copy to `.env`)
