# Face Recognition Documentation

This documentation covers the `FaceRecognition` component, which enables the robot to identify known people using computer vision.

## Overview

The `FaceRecognition` engine runs actively in the background to detect faces in the camera feed and match them against a known database (via `FaceManager`). It uses the **InsightFace** library (ArcFace/buffalo_l models) for state-of-the-art recognition accuracy.

**File:** `reachy_nova/face_recognition.py`

## Class Structure

### `FaceRecognition`

The main class responsible for managing the recognition model and background processing.

#### Constructor

```python
face_recognition = FaceRecognition(
    face_manager: FaceManager,
    on_match: Callable[[str, str, float], None] | None = None
)
```

-   **face_manager**: Instance of `FaceManager` holding the known face database.
-   **on_match**: (Optional) Callback triggered when a face is recognized `(id, name, score)`.

#### Key Methods

-   `start(stop_event: threading.Event)`: Ready the engine (model lazy-loads on first frame).
-   `update_frame(frame: np.ndarray, t: float)`: Submits a frame for processing. Non-blocking; drops frame if the background thread is busy or `DETECT_INTERVAL` (0.5s) hasn't elapsed.
-   `get_recognized_person() -> tuple[str, str, float] | None`: Thread-safe retrieval of the currently recognized person.
-   `is_admin_authenticated() -> bool`: Checks if the currently visible face belongs to an admin.
-   `get_current_embedding() -> np.ndarray | None`: Returns the 512-dim embedding of the current face.

### Processing Logic

1.  **Background Thread**: Detection runs in a daemon thread to avoid blocking the main application loop.
2.  **Lazy Loading**: The heavy InsightFace models (`buffalo_l`) are loaded only when the first frame is processed.
3.  **Frequency Control**: Runs at most every **500ms** (`DETECT_INTERVAL`).
4.  **Selection**: If multiple faces are present, it selects the **largest face** (by bounding box area).
5.  **Matching**:
    -   Extracts normalized embedding.
    -   Queries `FaceManager` for a match (cosine similarity > threshold).
    -   If matched, updates the `recognized_person` state.

### Re-announce Cooldown

To prevent the robot from repeatedly greeting the same person every 0.5s, the component implements a **30-second cooldown** (`REANNOUNCE_COOLDOWN`).
-   The `on_match` callback is fired only once per 30s for the same Face ID.
-   The internal state `recognized_person` is updated continuously (every frame), allowing for real-time logic (like admin checks) without spamming events.

### Thread Safety

-   Uses `threading.Lock` to synchronize access to `_recognized_person` and `_current_embedding`.
-   Safe to call `get_recognized_person()` from any thread (e.g., the main loop or API handlers).

## Face Manager

The `FaceManager` class handles the **storage and lifecycle** of face embeddings.

**File:** `reachy_nova/face_manager.py`

### Storage Tiers

It implements a two-tier storage system:

1.  **Temporary (In-Memory)**:
    -   Stores embeddings for unknown faces seen recently.
    -   **TTL**: 15 minutes (`TEMP_TTL`).
    -   Used for "who is that?" queries before a name is assigned.
    -   IDs start with `tmp_`.

2.  **Permanent (Disk)**:
    -   Stores embeddings for known/consented people.
    -   **Metadata**: `~/.reachy_nova/faces/faces.json`
    -   **Embeddings**: `~/.reachy_nova/faces/embeddings/<id>.npy`
    -   IDs are 4-character alphanumeric strings.

### Admin User
The system automatically creates an **admin** user entry (`"Ori Nachum"` by default). The admin face must be registered on the first run to enable privileged commands.

### Key Methods

-   `match(embedding, threshold=0.5)`: Finds the best match among permanent faces.
-   `remember_temporary(embedding)`: Temporarily stores a face.
-   `consent(temp_id, full_name)`: Promotes a temporary face to permanent storage.
-   `forget(unique_id, name)`: Deletes a permanent face (cannot delete admin).
-   `add_angles(id, name, embedding)`: Adds supplementary embeddings to improve recognition accuracy for an existing person.

## Dependencies

-   **insightface**: For ArcFace/RetinaFace implementation.
-   **onnxruntime-gpu** (or cpu): Inference engine backend.
-   **numpy**: Array manipulations.

## Usage Example

```python
# Initialize
face_manager = FaceManager(db_path="faces.json")
face_rec = FaceRecognition(
    face_manager=face_manager,
    on_match=lambda id, name, score: print(f"Hello {name}!")
)
face_rec.start(stop_event)

# In main loop
frame = robot.get_frame()
face_rec.update_frame(frame, time.time())

# Check admin status later
if face_rec.is_admin_authenticated():
    unlock_sensitive_features()
```
