# Enrollment Image Storage & Debug Saving Specification

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development

**Goal:** Store original enrollment image in DB and save it during debug auth to help visualize what was matched.

**Architecture:** Add image storage to face_templates table, modify enrollment to save image, modify identify to retrieve and save matched image in debug mode.

**Tech Stack:** Python FastAPI backend, PostgreSQL

---

## Background

Currently only 512-dimensional embedding vector is stored in DB. For debugging, we need to:
1. See the original image that was enrolled
2. Compare it against the auth frame being verified

## Requirements

### 1. Database Schema Change

Add `image_base64` column to `face_templates` table:
```sql
ALTER TABLE face_templates ADD COLUMN image_base64 TEXT;
```

### 2. Enrollment Changes

- Modify `FaceTemplateStore.save_template()` to accept and store image
- Modify `auth_service.enroll()` to save original image when provided

### 3. Identify/Debug Changes

- When identify runs in debug mode, retrieve matched user's enrollment image
- Save it to debug folder alongside the auth frame

### 4. Debug Output

In debug mode, save:
- `_4_enrolled_image.jpg` — The original image from DB that matched
- Update `_meta.json` to include `matched_user_id`, `matched_enrollment_image_path`

---

## File Changes

### Modified Files
- `src/fas/auth_store.py` — Add image storage to save_template/get_template
- `src/fas/auth_service.py` — Pass image during enrollment, retrieve during identify
- `services/api/app.py` — Pass debug flag to identify (if needed)

### New Files
- None

---

## Acceptance Criteria

1. Enrollment saves both embedding AND original image to DB
2. Identify in debug mode saves the matched enrollment image to debug folder
3. `_meta.json` includes `matched_user_id` and reference to enrolled image
4. No breaking changes to non-debug auth flow