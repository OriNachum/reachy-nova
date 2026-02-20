"""Nova Memory - Persistent multi-layer knowledge retrieval for Reachy Nova.

Connects to the existing qq knowledge system (MongoDB + Neo4j + local files)
using Bedrock-native models (Nova 2 Lite for synthesis, Nova 2 Multimodal
Embeddings for vectors). Results stream progressively to the voice conversation.
"""

import json
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import boto3

from .temporal import relative_vague

logger = logging.getLogger(__name__)

# Default paths to qq knowledge files
DEFAULT_CORE_MD = Path.home() / "git/autonomous-intelligence/qq/memory/core.md"
DEFAULT_NOTES_MD = Path.home() / "git/autonomous-intelligence/qq/memory/notes.md"

# Bedrock model IDs
EMBEDDING_MODEL = "amazon.nova-2-multimodal-embeddings-v1:0"
EMBEDDING_DIM = 1024
LLM_MODEL = "us.amazon.nova-2-lite-v1:0"

# MongoDB defaults
DEFAULT_MONGO_URI = "mongodb://localhost:27017"
DEFAULT_MONGO_DB = "qq_memory"
DEFAULT_QQ_COLLECTION = "notes"
DEFAULT_NOVA_COLLECTION = "nova_notes"

# Neo4j defaults
DEFAULT_NEO4J_URI = "bolt://localhost:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "refinerypass"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class NovaMemory:
    """Multi-layer knowledge retrieval with progressive event-driven results."""

    def __init__(
        self,
        region: str = "us-east-1",
        mongo_uri: str | None = None,
        mongo_db: str | None = None,
        neo4j_uri: str | None = None,
        neo4j_user: str | None = None,
        neo4j_password: str | None = None,
        core_md_path: Path | None = None,
        notes_md_path: Path | None = None,
        on_progress: Callable[[str], None] | None = None,
        on_result: Callable[[str], None] | None = None,
        on_state_change: Callable[[str], None] | None = None,
        on_context: Callable[[str], None] | None = None,
    ):
        self.region = region
        self.mongo_uri = mongo_uri or os.environ.get("MONGODB_URI", DEFAULT_MONGO_URI)
        self.mongo_db = mongo_db or os.environ.get("MONGO_DB", DEFAULT_MONGO_DB)
        self.neo4j_uri = neo4j_uri or os.environ.get("NEO4J_URI", DEFAULT_NEO4J_URI)
        self.neo4j_user = neo4j_user or os.environ.get("NEO4J_USER", DEFAULT_NEO4J_USER)
        self.neo4j_password = neo4j_password or os.environ.get("NEO4J_PASSWORD", DEFAULT_NEO4J_PASSWORD)
        self.core_md_path = core_md_path or DEFAULT_CORE_MD
        self.notes_md_path = notes_md_path or DEFAULT_NOTES_MD

        self.on_progress = on_progress
        self.on_result = on_result
        self.on_state_change = on_state_change
        self.on_context = on_context

        self.state = "idle"
        self._session_notes: list[dict] = []
        self._session_lock = threading.Lock()

        # Backend availability flags
        self._mongo_available = False
        self._neo4j_available = False
        self._bedrock_available = False

        # Lazy-initialized clients
        self._bedrock_client = None
        self._mongo_client = None
        self._mongo_db_handle = None
        self._neo4j_driver = None

        # Initialize Bedrock client eagerly (needed for embeddings)
        try:
            self._bedrock_client = boto3.client("bedrock-runtime", region_name=region)
            self._bedrock_available = True
        except Exception as e:
            logger.warning(f"Bedrock client init failed: {e}")

    # --- State & progress ---

    def _set_state(self, state: str) -> None:
        self.state = state
        if self.on_state_change:
            try:
                self.on_state_change(state)
            except Exception:
                pass

    def _emit_progress(self, message: str) -> None:
        logger.info(f"[Memory] {message}")
        if self.on_progress:
            try:
                self.on_progress(message)
            except Exception:
                pass

    # --- Backend initialization (lazy, with timeouts) ---

    def _init_backends(self) -> None:
        """Lazy-connect to MongoDB and Neo4j with short timeouts."""
        if not self._mongo_available:
            self._init_mongo()
        if not self._neo4j_available:
            self._init_neo4j()

    def _init_mongo(self) -> None:
        try:
            from pymongo import MongoClient
            self._mongo_client = MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=2000,
                connectTimeoutMS=2000,
            )
            # Test connection
            self._mongo_client.admin.command("ping")
            self._mongo_db_handle = self._mongo_client[self.mongo_db]
            self._mongo_available = True
            logger.info(f"MongoDB connected: {self.mongo_uri}/{self.mongo_db}")
        except Exception as e:
            logger.warning(f"MongoDB unavailable: {e}")
            self._mongo_available = False

    def _init_neo4j(self) -> None:
        try:
            from neo4j import GraphDatabase
            self._neo4j_driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
                connection_timeout=2,
            )
            # Test connection
            self._neo4j_driver.verify_connectivity()
            self._neo4j_available = True
            logger.info(f"Neo4j connected: {self.neo4j_uri}")
        except Exception as e:
            logger.warning(f"Neo4j unavailable: {e}")
            self._neo4j_available = False

    # --- Bedrock helpers ---

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding vector using Amazon Nova 2 Multimodal Embeddings."""
        if not self._bedrock_available:
            return []
        try:
            body = {
                "taskType": "SINGLE_EMBEDDING",
                "singleEmbeddingParams": {
                    "embeddingPurpose": "GENERIC_RETRIEVAL",
                    "embeddingDimension": EMBEDDING_DIM,
                    "text": {
                        "truncationMode": "END",
                        "value": text[:2048],
                    },
                },
            }
            response = self._bedrock_client.invoke_model(
                modelId=EMBEDDING_MODEL,
                body=json.dumps(body),
            )
            result = json.loads(response["body"].read())
            embeddings = result.get("embeddings", [])
            if embeddings:
                return embeddings[0].get("embedding", [])
            return []
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return []

    def _synthesize(self, query: str, chunks: list[str]) -> str:
        """Synthesize a concise voice-friendly answer from retrieved chunks."""
        if not chunks:
            return ""
        if not self._bedrock_available:
            return "\n".join(chunks[:3])

        context = "\n---\n".join(chunks[:10])
        prompt = (
            f"Based on the following knowledge, answer the query concisely "
            f"in 1-3 sentences suitable for spoken voice output.\n\n"
            f"Query: {query}\n\nKnowledge:\n{context}"
        )
        return self._llm_call(prompt)

    def _llm_call(self, prompt: str) -> str:
        """General-purpose LLM call using Nova 2 Lite."""
        if not self._bedrock_available:
            return ""
        try:
            body = {
                "messages": [
                    {"role": "user", "content": [{"text": prompt}]}
                ],
                "inferenceConfig": {
                    "maxTokens": 512,
                    "temperature": 0.3,
                    "topP": 0.9,
                },
            }
            response = self._bedrock_client.invoke_model(
                modelId=LLM_MODEL,
                body=json.dumps(body),
            )
            result = json.loads(response["body"].read())
            return result["output"]["message"]["content"][0]["text"]
        except Exception as e:
            logger.error(f"LLM call error: {e}")
            return ""

    # --- Retrieval layers ---

    def _search_local_files(self, query: str) -> list[str]:
        """Keyword search in core.md and notes.md (instant)."""
        chunks = []
        keywords = [w.lower() for w in query.split() if len(w) > 2]
        if not keywords:
            return chunks

        for path in [self.core_md_path, self.notes_md_path]:
            if not path.exists():
                continue
            try:
                text = path.read_text()
                sections = re.split(r'\n## ', text)
                for section in sections:
                    lower = section.lower()
                    if any(kw in lower for kw in keywords):
                        # Take first 500 chars of matching section
                        header = section.split('\n')[0].strip('# ')
                        content = section[:500].strip()
                        chunks.append(f"[{path.name}/{header}] {content}")
            except Exception as e:
                logger.warning(f"Error reading {path}: {e}")

        return chunks

    def _search_session_notes(self, query: str) -> list[str]:
        """Search in-memory session notes (instant)."""
        chunks = []
        keywords = [w.lower() for w in query.split() if len(w) > 2]
        if not keywords:
            return chunks

        with self._session_lock:
            for note in self._session_notes:
                content = note.get("content", "").lower()
                if any(kw in content for kw in keywords):
                    chunks.append(f"[session] {note['content']}")

        return chunks

    def _search_mongo_keywords(self, query: str) -> list[str]:
        """Keyword/text search on the existing qq notes collection."""
        if not self._mongo_available:
            return []
        try:
            collection = self._mongo_db_handle[DEFAULT_QQ_COLLECTION]
            keywords = [w for w in query.split() if len(w) > 2]
            if not keywords:
                return []

            # Use regex OR matching across content field
            regex_pattern = "|".join(re.escape(kw) for kw in keywords)
            cursor = collection.find(
                {"content": {"$regex": regex_pattern, "$options": "i"}},
                {"content": 1, "section": 1, "created_at": 1, "_id": 0},
            ).limit(5)

            chunks = []
            for doc in cursor:
                section = doc.get("section", "")
                content = doc.get("content", "")[:400]
                created = doc.get("created_at", 0)
                age = f" ({relative_vague(created)})" if created else ""
                chunks.append(f"[qq/{section}] {content}{age}")
            return chunks
        except Exception as e:
            logger.warning(f"MongoDB keyword search error: {e}")
            return []

    def _search_mongo_vector(self, embedding: list[float]) -> list[str]:
        """Cosine similarity search on nova_notes collection (Nova embeddings)."""
        if not self._mongo_available or not embedding:
            return []
        try:
            collection = self._mongo_db_handle[DEFAULT_NOVA_COLLECTION]
            # Fetch all docs with embeddings and compute similarity in Python
            # (for small collections; for large ones, use Atlas vector search)
            cursor = collection.find(
                {"embedding": {"$exists": True}},
                {"content": 1, "section": 1, "embedding": 1, "created_at": 1, "_id": 0},
            ).limit(100)

            scored = []
            for doc in cursor:
                doc_emb = doc.get("embedding", [])
                if not doc_emb:
                    continue
                sim = _cosine_similarity(embedding, doc_emb)
                scored.append((sim, doc))

            scored.sort(key=lambda x: x[0], reverse=True)
            chunks = []
            for sim, doc in scored[:5]:
                if sim < 0.3:
                    break
                section = doc.get("section", "")
                content = doc.get("content", "")[:400]
                created = doc.get("created_at", 0)
                age = f" ({relative_vague(created)})" if created else ""
                chunks.append(f"[nova/{section} sim={sim:.2f}] {content}{age}")
            return chunks
        except Exception as e:
            logger.warning(f"MongoDB vector search error: {e}")
            return []

    def _graph_walk(self, query: str, embedding: list[float]) -> list[str]:
        """LLM-guided graph traversal through Neo4j knowledge graph."""
        if not self._neo4j_available or not embedding:
            return []

        chunks = []
        try:
            with self._neo4j_driver.session() as session:
                # Step 1: Find seed entities by name keyword match
                keywords = [w for w in query.split() if len(w) > 2]
                if not keywords:
                    return []

                regex_pattern = "(?i)" + "|".join(re.escape(kw) for kw in keywords)
                seed_result = session.run(
                    "MATCH (n) WHERE n.name =~ $pattern "
                    "RETURN n.name AS name, labels(n) AS labels, "
                    "n.description AS desc LIMIT 5",
                    pattern=regex_pattern,
                )
                seeds = [dict(r) for r in seed_result]

                if not seeds:
                    return []

                seed_names = [s["name"] for s in seeds]
                self._emit_progress(f"Found entities: {', '.join(seed_names[:3])}")

                # Step 2: For each seed, get 1-hop relationships
                visited = set()
                for seed in seeds[:3]:
                    name = seed["name"]
                    if name in visited:
                        continue
                    visited.add(name)

                    # Add seed description
                    if seed.get("desc"):
                        chunks.append(f"[entity:{name}] {seed['desc']}")

                    # Get 1-hop relationships
                    rel_result = session.run(
                        "MATCH (n {name: $name})-[r]->(m) "
                        "RETURN type(r) AS rel, m.name AS target, "
                        "m.description AS target_desc, r.description AS rel_desc "
                        "LIMIT 15",
                        name=name,
                    )
                    rels = [dict(r) for r in rel_result]

                    if not rels:
                        continue

                    # Ask LLM which relationships are relevant
                    rel_summary = "\n".join(
                        f"- {name} --[{r['rel']}]--> {r['target']}"
                        for r in rels
                    )
                    eval_prompt = (
                        f"Given the query '{query}', which of these knowledge graph "
                        f"relationships are relevant? List only relevant target names, "
                        f"or say NONE if none are relevant.\n\n{rel_summary}"
                    )
                    llm_answer = self._llm_call(eval_prompt)

                    if "NONE" in llm_answer.upper():
                        continue

                    # Collect relevant relationship context
                    for r in rels:
                        target = r["target"]
                        if target and target.lower() in llm_answer.lower():
                            self._emit_progress(f"Following: {name} → {r['rel']} → {target}")
                            rel_desc = r.get("rel_desc", "")
                            target_desc = r.get("target_desc", "")
                            context = f"[graph:{name}→{r['rel']}→{target}]"
                            if rel_desc:
                                context += f" {rel_desc}"
                            if target_desc:
                                context += f" {target_desc}"
                            chunks.append(context)

                            # 2nd hop from relevant targets
                            if target not in visited:
                                visited.add(target)
                                hop2 = session.run(
                                    "MATCH (n {name: $name})-[r]->(m) "
                                    "RETURN type(r) AS rel, m.name AS target, "
                                    "m.description AS target_desc LIMIT 5",
                                    name=target,
                                )
                                for h in hop2:
                                    h = dict(h)
                                    chunks.append(
                                        f"[graph:{target}→{h['rel']}→{h['target']}] "
                                        f"{h.get('target_desc', '')}"
                                    )

                            if len(chunks) >= 10:
                                break

        except Exception as e:
            logger.warning(f"Graph walk error: {e}")

        return chunks

    # --- Progressive query ---

    def query(self, text: str) -> str:
        """Query knowledge with progressive event-driven results.

        Returns immediately with local-file answer. Background threads push
        deeper results via on_progress callbacks.
        """
        self._set_state("retrieving")
        self._emit_progress("Searching my notes...")

        # Phase 1: Instant local search
        local_chunks = self._search_local_files(text)
        session_chunks = self._search_session_notes(text)
        all_local = local_chunks + session_chunks

        if all_local:
            self._emit_progress(f"Found {len(all_local)} local matches")

        quick_answer = ""
        if all_local:
            quick_answer = self._synthesize(text, all_local)
        if not quick_answer:
            quick_answer = "I don't have that in my local notes."

        # Phase 2+3: Background deep search
        def _deep_search():
            try:
                self._init_backends()

                # Phase 2: MongoDB
                self._emit_progress("Searching long-term memory...")
                embedding = self._get_embedding(text)

                mongo_kw_chunks = self._search_mongo_keywords(text)
                mongo_vec_chunks = self._search_mongo_vector(embedding)
                mongo_chunks = mongo_kw_chunks + mongo_vec_chunks

                if mongo_chunks:
                    self._emit_progress(f"Found {len(mongo_chunks)} memories")
                    update = self._synthesize(text, mongo_chunks)
                    if update and update.strip():
                        if self.on_progress:
                            self.on_progress(f"Memory update: {update}")

                # Phase 3: Neo4j graph walk
                if self._neo4j_available:
                    self._emit_progress("Walking knowledge graph...")
                    graph_chunks = self._graph_walk(text, embedding)
                    if graph_chunks:
                        update = self._synthesize(text, graph_chunks)
                        if update and update.strip():
                            if self.on_progress:
                                self.on_progress(f"Knowledge graph: {update}")

                self._emit_progress("Memory search complete")
            except Exception as e:
                logger.error(f"Deep search error: {e}")
            finally:
                self._set_state("idle")

        threading.Thread(target=_deep_search, daemon=True, name="memory-deep").start()

        if self.on_result:
            self.on_result(quick_answer)

        return quick_answer

    # --- Store ---

    def store(self, content: str, section: str = "session") -> str:
        """Store a fact in session memory and MongoDB (with embedding)."""
        self._set_state("storing")

        # Always store in session
        note = {
            "content": content,
            "section": section,
            "timestamp": time.time(),
            "note_id": f"nova_{uuid.uuid4().hex[:12]}",
        }
        with self._session_lock:
            self._session_notes.append(note)

        # Try to persist to MongoDB with embedding
        def _persist():
            try:
                self._init_backends()
                if not self._mongo_available:
                    return

                embedding = self._get_embedding(content)
                doc = {
                    **note,
                    "embedding": embedding,
                    "created_at": time.time(),
                    "source": "reachy_nova",
                }
                collection = self._mongo_db_handle[DEFAULT_NOVA_COLLECTION]
                collection.insert_one(doc)
                logger.info(f"Stored note in MongoDB: {note['note_id']}")
            except Exception as e:
                logger.warning(f"MongoDB store error: {e}")
            finally:
                self._set_state("idle")

        threading.Thread(target=_persist, daemon=True, name="memory-store").start()

        return f"Remembered: {content[:100]}"

    # --- Startup context ---

    def get_startup_context(self) -> str:
        """Build startup context from local files and recent memories."""
        parts = []

        # Core identity from core.md
        if self.core_md_path.exists():
            try:
                text = self.core_md_path.read_text()
                # Extract Identity and Relationships sections
                for section_name in ["Identity", "Relationships"]:
                    match = re.search(
                        rf'## {section_name}\n(.*?)(?=\n## |\Z)',
                        text, re.DOTALL
                    )
                    if match:
                        parts.append(f"[{section_name}] {match.group(1).strip()}")
            except Exception as e:
                logger.warning(f"Error reading core.md: {e}")

        # Recent nova_notes from previous sessions
        self._init_backends()
        if self._mongo_available:
            try:
                collection = self._mongo_db_handle[DEFAULT_NOVA_COLLECTION]
                recent = collection.find(
                    {}, {"content": 1, "section": 1, "created_at": 1, "_id": 0}
                ).sort("created_at", -1).limit(5)
                labeled = []
                for doc in recent:
                    content = doc.get("content", "")
                    if not content:
                        continue
                    created = doc.get("created_at", 0)
                    age = f" ({relative_vague(created)})" if created else ""
                    labeled.append(f"{content}{age}")
                if labeled:
                    parts.append("[Recent memories] " + " | ".join(labeled))
            except Exception as e:
                logger.warning(f"Error fetching recent notes: {e}")

        # Neo4j summary
        if self._neo4j_available:
            try:
                with self._neo4j_driver.session() as session:
                    result = session.run(
                        "MATCH (n) RETURN count(n) AS nodes "
                        "UNION ALL "
                        "MATCH ()-[r]->() RETURN count(r) AS nodes"
                    )
                    counts = [r["nodes"] for r in result]
                    if len(counts) >= 2:
                        parts.append(
                            f"[Knowledge graph] {counts[0]} entities, "
                            f"{counts[1]} relationships"
                        )
            except Exception as e:
                logger.warning(f"Error getting graph summary: {e}")

        context = "\n".join(parts) if parts else ""
        if self.on_context:
            self.on_context(context)
        return context

    # --- Health check ---

    def health(self) -> dict[str, bool]:
        """Check connectivity of all backends."""
        self._init_backends()
        h = {
            "bedrock": self._bedrock_available,
            "mongo": self._mongo_available,
            "neo4j": self._neo4j_available,
            "local_files": self.core_md_path.exists(),
        }
        # Test embedding if bedrock is available
        if self._bedrock_available:
            emb = self._get_embedding("test")
            h["embeddings"] = len(emb) == EMBEDDING_DIM
        return h
