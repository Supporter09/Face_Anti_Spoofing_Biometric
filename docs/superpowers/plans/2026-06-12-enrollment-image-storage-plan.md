# Enrollment Image Storage & Debug Saving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development

**Goal:** Store original enrollment image in DB and save it during debug auth.

**Architecture:** Add image storage to face_templates table, modify enrollment to save image, modify identify to retrieve and save matched image in debug mode.

---

## Task List

### Task 1: Add image column to database schema

**Files:**
- Modify: `src/fas/auth_store.py`

- [ ] **Step 1: Update _init_db() to include image_base64 column**

```python
def _init_db() -> None:
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS face_templates (
                    user_id     TEXT PRIMARY KEY,
                    embedding   DOUBLE PRECISION[],
                    image_base64 TEXT,
                    enrolled_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            # Add column if it doesn't exist (for existing databases)
            cur.execute("""
                ALTER TABLE face_templates 
                ADD COLUMN IF NOT EXISTS image_base64 TEXT
            """)
        conn.commit()
```

- [ ] **Step 2: Test database connection**

Run: `cd src && python -c "from fas.auth_store import _init_db; _init_db(); print('db ok')"`

- [ ] **Step 3: Commit**

---

### Task 2: Update FaceTemplateStore to store/retrieve image

**Files:**
- Modify: `src/fas/auth_store.py`

- [ ] **Step 1: Update save_template() to accept image_base64**

```python
def save_template(self, user_id: str, embedding: np.ndarray, image_base64: str | None = None) -> None:
    embedding_list = embedding.astype(float).tolist()
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO face_templates (user_id, embedding, image_base64)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE
                    SET embedding   = EXCLUDED.embedding,
                        image_base64 = EXCLUDED.image_base64,
                        enrolled_at = NOW()
            """, (user_id, embedding_list, image_base64))
        conn.commit()
```

- [ ] **Step 2: Update get_template() to return image too**

```python
def get_template(self, user_id: str) -> tuple[np.ndarray | None, str | None]:
    # Returns (embedding, image_base64)
```

- [ ] **Step 3: Test**

Run: `cd src && python -c "from fas.auth_store import FaceTemplateStore; print('store ok')"`

- [ ] **Step 4: Commit**

---

### Task 3: Modify auth_service.enroll() to save image

**Files:**
- Modify: `src/fas/auth_service.py`

- [ ] **Step 1: Pass image to store.save_template()**

In `enroll()` method, pass `request.image_base64` to save_template.

- [ ] **Step 2: Commit**

---

### Task 4: Modify auth_service.identify() for debug mode

**Files:**
- Modify: `src/fas/auth_service.py`

- [ ] **Step 1: Add debug parameter to identify()**

```python
def identify(
    self, 
    request: FaceIdentifyRequest,
    *,
    capture_debug: bool = False,
    debug_session_id: str = 'default',
) -> FaceIdentifyResponse:
```

- [ ] **Step 2: When match found, retrieve enrollment image**

After finding the matched user, get their enrollment image from store.

- [ ] **Step 3: Save matched enrollment image to debug folder**

Create helper to save the enrolled image alongside auth frame.

- [ ] **Step 4: Include debug info in response**

Add matched_user_id, enrollment_image_path to response (debug only).

- [ ] **Step 5: Commit**

---

### Task 5: Update API endpoint to pass debug flag

**Files:**
- Modify: `services/api/app.py`

- [ ] **Step 1: Pass capture_debug to identify()**

```python
@app.post("/v1/auth/identify", response_model=FaceIdentifyResponse)
def identify_face(payload: FaceIdentifyRequest) -> FaceIdentifyResponse:
    capture_debug = request.url.params.get('capture_debug') == '1'
    session_id = request.url.params.get('session_id', 'default')
    return auth_service.identify(payload, capture_debug=capture_debug, debug_session_id=session_id)
```

Or use query params.

- [ ] **Step 2: Commit**

---

### Task 6: Final verification

**Files:**
- All modified files

- [ ] **Step 1: Test enrollment with image**

- [ ] **Step 2: Test identify in debug mode**

- [ ] **Step 3: Verify debug output includes enrolled image**

- [ ] **Step 4: Commit**