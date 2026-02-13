# Nova Memory

Nova Memory is the long-term knowledge retrieval system for Reachy Nova. It provides a persistent memory layer that allows the robot to store and recall information across sessions.

## Features

-   **Multi-Layer Retrieval**:
    -   **Local Files**: Instant access to `core.md` (identity) and `notes.md`.
    -   **Session Memory**: In-memory storage for the current conversation.
    -   **Vector Database (MongoDB)**: Semantic search using Amazon Nova 2 Multimodal Embeddings.
    -   **Knowledge Graph (Neo4j)**: structured relationship queries via Cypher.
-   **Event-Driven**: Progress updates are streamed to the voice output as they happen.
-   **Bedrock Integration**: Uses `amazon.nova-2-lite-v1:0` for knowledge synthesis and `amazon.nova-2-multimodal-embeddings-v1:0` for vectorization.

## Architecture

Nova Memory is designed to be "progressive". When a query is made:

1.  **Immediate**: Checks local files and session notes. Returns instantly if found.
2.  **Background**: Spawns a background thread to search MongoDB and Neo4j.
3.  **Stream**: Injects new findings into the voice conversation as "Memory updates" or "Knowledge graph" events.

## Configuration

Nova Memory relies on the following environment variables (or defaults):

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGO_DB` | `qq_memory` | Database name |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `refinerypass` | Neo4j password |

## Usage

### In Code

```python
from reachy_nova.nova_memory import NovaMemory

memory = NovaMemory(
    on_progress=lambda msg: print(f"Progress: {msg}"),
    on_result=lambda res: print(f"Result: {res}"),
)

# Store a fact
memory.store("The user likes strawberries.")

# Query (async with callbacks)
memory.query("What does the user like?")
```

### As a Skill

The memory system is exposed as a tool to the LLM:

-   **`memory(query="...", mode="query")`**: Recall information.
-   **`memory(query="...", mode="store")`**: Save a new fact.
-   **`memory(mode="context")`**: Get the full startup context (identity + recent memories).
```
