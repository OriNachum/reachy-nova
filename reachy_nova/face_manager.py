"""Face storage and lifecycle management for Reachy Nova.

Manages face embeddings with two tiers:
- Temporary: in-memory with 15-min TTL, auto-cleaned periodically
- Permanent: JSON-persisted with .npy embedding files on disk

Storage layout:
    ~/.reachy_nova/faces/
        faces.json          # metadata for all permanent faces
        embeddings/         # 128-dim SFace .npy files
            <id>.npy
            <id>_1.npy      # additional angles
"""

import json
import logging
import random
import string
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

STORAGE_DIR = Path.home() / ".reachy_nova" / "faces"
EMBEDDINGS_DIR = STORAGE_DIR / "embeddings"
FACES_JSON = STORAGE_DIR / "faces.json"

TEMP_TTL = 15 * 60  # 15 minutes

ADMIN_NAME = "Ori Nachum"


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _generate_id(existing: set[str]) -> str:
    """Generate a unique 4-char alphanumeric ID (a-z0-9)."""
    chars = string.ascii_lowercase + string.digits
    for _ in range(1000):
        candidate = "".join(random.choices(chars, k=4))
        if candidate not in existing:
            return candidate
    raise RuntimeError("Could not generate unique ID after 1000 attempts")


class FaceManager:
    """Manages face embeddings on disk with temp and permanent tiers."""

    def __init__(self):
        # Permanent faces: {unique_id: {name, is_admin, created, embedding_files}}
        self.faces: dict[str, dict] = {}
        # Temporary faces: {temp_id: {embedding, created}}
        self.temp_faces: dict[str, dict] = {}
        self._loaded = False

    def load(self):
        """Load permanent faces from disk."""
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

        if FACES_JSON.exists():
            try:
                with open(FACES_JSON) as f:
                    self.faces = json.load(f)
                logger.info(f"Loaded {len(self.faces)} permanent faces")
            except Exception as e:
                logger.error(f"Failed to load faces.json: {e}")
                self.faces = {}
        else:
            self.faces = {}

        # Ensure admin exists
        self._ensure_admin()
        self._loaded = True

    def save(self):
        """Persist permanent faces to disk."""
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(FACES_JSON, "w") as f:
                json.dump(self.faces, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save faces.json: {e}")

    def _ensure_admin(self):
        """Create admin face entry if it doesn't exist (no embedding yet)."""
        for fid, data in self.faces.items():
            if data.get("is_admin"):
                return
        # Admin not found — create placeholder (embedding added via add_angles or consent)
        admin_id = _generate_id(set(self.faces.keys()))
        self.faces[admin_id] = {
            "name": ADMIN_NAME,
            "is_admin": True,
            "created": time.time(),
            "embedding_files": [],
        }
        self.save()
        logger.info(f"Created admin face entry: {admin_id} ({ADMIN_NAME})")

    def remember_temporary(self, embedding: np.ndarray) -> str:
        """Store a face embedding temporarily (15-min TTL).

        Returns:
            temp_id string (prefixed with 'tmp_')
        """
        temp_id = f"tmp_{_generate_id(set(self.temp_faces.keys()))}"
        self.temp_faces[temp_id] = {
            "embedding": embedding,
            "created": time.time(),
        }
        logger.info(f"Temporary face stored: {temp_id}")
        return temp_id

    def consent(self, temp_id: str, full_name: str) -> str | None:
        """Promote a temporary face to permanent storage.

        Args:
            temp_id: temporary face ID
            full_name: person's full name

        Returns:
            unique_id if successful, None if temp_id not found or expired
        """
        temp = self.temp_faces.pop(temp_id, None)
        if temp is None:
            logger.warning(f"Temp face not found: {temp_id}")
            return None

        # Check if expired
        if time.time() - temp["created"] > TEMP_TTL:
            logger.warning(f"Temp face expired: {temp_id}")
            return None

        existing_ids = set(self.faces.keys())
        unique_id = _generate_id(existing_ids)

        # Save embedding to disk
        emb_file = f"{unique_id}.npy"
        np.save(EMBEDDINGS_DIR / emb_file, temp["embedding"])

        is_admin = full_name.lower().strip() == ADMIN_NAME.lower()

        # If this is the admin and a placeholder exists, update the placeholder
        if is_admin:
            for fid, data in self.faces.items():
                if data.get("is_admin") and not data["embedding_files"]:
                    data["embedding_files"].append(emb_file)
                    # Move the .npy file to use the existing admin ID
                    old_path = EMBEDDINGS_DIR / emb_file
                    new_file = f"{fid}.npy"
                    new_path = EMBEDDINGS_DIR / new_file
                    if old_path.exists():
                        old_path.rename(new_path)
                    data["embedding_files"] = [new_file]
                    self.save()
                    logger.info(f"Updated admin face with embedding: {fid}")
                    return fid

        self.faces[unique_id] = {
            "name": full_name,
            "is_admin": is_admin,
            "created": time.time(),
            "embedding_files": [emb_file],
        }
        self.save()
        logger.info(f"Permanent face created: {unique_id} ({full_name})")
        return unique_id

    def forget(self, unique_id: str, full_name: str) -> bool:
        """Delete a permanent face.

        Args:
            unique_id: face ID to delete
            full_name: must match stored name (safety check)

        Returns:
            True if deleted, False otherwise
        """
        face = self.faces.get(unique_id)
        if face is None:
            return False

        if face.get("is_admin"):
            logger.warning("Cannot delete admin face")
            return False

        if face["name"].lower().strip() != full_name.lower().strip():
            logger.warning(f"Name mismatch for forget: expected '{face['name']}', got '{full_name}'")
            return False

        # Delete embedding files
        for emb_file in face.get("embedding_files", []):
            path = EMBEDDINGS_DIR / emb_file
            if path.exists():
                path.unlink()

        del self.faces[unique_id]
        self.save()
        logger.info(f"Deleted face: {unique_id} ({full_name})")
        return True

    def add_angles(self, unique_id: str, full_name: str, embedding: np.ndarray) -> bool:
        """Add an additional embedding for better multi-angle recognition.

        Args:
            unique_id: face ID
            full_name: must match stored name
            embedding: 512-dim ArcFace embedding

        Returns:
            True if added, False otherwise
        """
        face = self.faces.get(unique_id)
        if face is None:
            return False

        if face["name"].lower().strip() != full_name.lower().strip():
            return False

        idx = len(face.get("embedding_files", []))
        emb_file = f"{unique_id}_{idx}.npy"
        np.save(EMBEDDINGS_DIR / emb_file, embedding)
        face.setdefault("embedding_files", []).append(emb_file)
        self.save()
        logger.info(f"Added angle {idx} for {unique_id} ({full_name})")
        return True

    def merge(self, id1: str, name1: str, id2: str, name2: str) -> bool:
        """Merge two face entries, keeping id1 and deleting id2.

        Args:
            id1, name1: primary face (kept)
            id2, name2: secondary face (merged into id1, then deleted)

        Returns:
            True if merged, False otherwise
        """
        face1 = self.faces.get(id1)
        face2 = self.faces.get(id2)
        if face1 is None or face2 is None:
            return False

        if face1["name"].lower().strip() != name1.lower().strip():
            return False
        if face2["name"].lower().strip() != name2.lower().strip():
            return False

        if face2.get("is_admin"):
            logger.warning("Cannot merge admin face as secondary")
            return False

        # Move embedding files from face2 to face1
        for emb_file in face2.get("embedding_files", []):
            face1.setdefault("embedding_files", []).append(emb_file)

        del self.faces[id2]
        self.save()
        logger.info(f"Merged {id2} into {id1}")
        return True

    def match(self, embedding: np.ndarray, threshold: float = 0.5) -> tuple[str, str, float] | None:
        """Find the best matching face for an embedding.

        Args:
            embedding: 512-dim ArcFace embedding to match
            threshold: minimum cosine similarity (default 0.5)

        Returns:
            (unique_id, name, score) or None if no match above threshold
        """
        best_id = None
        best_name = None
        best_score = -1.0

        for fid, data in self.faces.items():
            for emb_file in data.get("embedding_files", []):
                path = EMBEDDINGS_DIR / emb_file
                if not path.exists():
                    continue
                try:
                    stored = np.load(path)
                    score = _cosine_similarity(embedding, stored)
                    if score > best_score:
                        best_score = score
                        best_id = fid
                        best_name = data["name"]
                except Exception as e:
                    logger.warning(f"Error loading embedding {emb_file}: {e}")

        if best_score >= threshold and best_id is not None:
            return (best_id, best_name, best_score)
        return None

    def cleanup_expired(self) -> int:
        """Remove expired temporary faces.

        Returns:
            Number of faces cleaned up
        """
        now = time.time()
        expired = [
            tid for tid, data in self.temp_faces.items()
            if now - data["created"] > TEMP_TTL
        ]
        for tid in expired:
            del self.temp_faces[tid]
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired temp faces")
        return len(expired)

    # --- Admin-only operations ---

    def is_admin(self, unique_id: str | None) -> bool:
        """Check if a face ID belongs to an admin."""
        if unique_id is None:
            return False
        face = self.faces.get(unique_id)
        return face is not None and face.get("is_admin", False)

    def list_faces(self) -> list[dict]:
        """List all permanent faces (admin only)."""
        result = []
        for fid, data in self.faces.items():
            result.append({
                "id": fid,
                "name": data["name"],
                "is_admin": data.get("is_admin", False),
                "num_embeddings": len(data.get("embedding_files", [])),
                "created": data.get("created", 0),
            })
        return result

    def get_face_count(self) -> int:
        """Get total number of permanent faces."""
        return len(self.faces)

    def get_person_images(self, unique_id: str, full_name: str) -> list[str]:
        """Get embedding file paths for a person (admin only)."""
        face = self.faces.get(unique_id)
        if face is None:
            return []
        if face["name"].lower().strip() != full_name.lower().strip():
            return []
        return [str(EMBEDDINGS_DIR / f) for f in face.get("embedding_files", [])]

    def get_unique_id(self, name: str) -> str | None:
        """Look up unique_id by name (admin only, case-insensitive)."""
        name_lower = name.lower().strip()
        for fid, data in self.faces.items():
            if data["name"].lower().strip() == name_lower:
                return fid
        return None
