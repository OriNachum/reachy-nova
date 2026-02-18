---
name: face
description: >
  Manage face recognition. Remember faces temporarily (15 min), save permanently
  with consent, forget faces, add angles for better recognition, merge duplicates.
  Some operations (list, count, images, whois) require admin authentication via camera.
metadata:
  author: reachy-nova
  version: "1.0"
---

# Face Skill

Manage face recognition and identity for people you interact with.

## Parameters
- operation (string, required): The face operation to perform
- name (string, optional): Full name of the person (required for consent, forget, add_angles, merge, whois)
- unique_id (string, optional): Face ID (required for forget, add_angles, merge, images)
- target_id (string, optional): Second face ID for merge operations
- target_name (string, optional): Second face name for merge operations

## Operations

### remember
Remember the face currently visible in the camera temporarily (15 minutes).
No name needed. Returns a temporary ID.

### consent
Save a temporarily remembered face permanently. Requires the temp_id (from remember)
and the person's full name. The person must explicitly consent to being saved.

### forget
Delete a permanently saved face. Requires unique_id and full_name.
Admin faces cannot be deleted.

### add_angles
Add another angle/view of a known person for better recognition.
The person should be visible in the camera. Requires unique_id and full_name.

### merge
Combine two face entries that are the same person. Keeps the first ID,
merges embeddings from the second. Requires unique_id + name and target_id + target_name.

### list (admin only)
List all known faces with their IDs and details.

### count (admin only)
Get the total number of known faces.

### images (admin only)
Get embedding file paths for a specific person. Requires unique_id and name.

### whois (admin only)
Look up a person's unique_id by their name.

## Examples
- "Remember my face" → operation: remember
- "Save me permanently as John Smith" → operation: consent, name: "John Smith"
- "Forget face abc1 John Smith" → operation: forget, unique_id: "abc1", name: "John Smith"
- "Add another angle for abc1 John Smith" → operation: add_angles, unique_id: "abc1", name: "John Smith"
- "How many faces do you know?" → operation: count
- "Who is stored as Jane Doe?" → operation: whois, name: "Jane Doe"
