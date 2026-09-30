# LibraryGuard — Privacy & Responsible Use

LibraryGuard is designed as a **behavioral** detection tool, not an identity
surveillance system. Privacy is an engineering constraint here, not a
disclaimer.

## What the system does

- Detects *behaviors* (drinking, sleeping, phone usage) in video.
- Attributes behaviors to **anonymous, temporary tracking IDs**
  (`Person #3`) that exist only for the duration of one processed video.
- Runs **entirely on your machine** — no frames, detections, or reports are
  sent to any external service by this codebase.

## What the system deliberately does NOT do

| Not implemented | Why |
|---|---|
| Facial recognition / face matching | Out of scope and ethically fraught for this use case |
| Identity lookup (names, student numbers, phone numbers) | Behavior detection ≠ identity surveillance |
| Biometric template storage | Nothing to leak: no embeddings or faceprints are stored |
| Cloud upload of footage | Processing is local by default and by design |
| Long-term tracking across videos | Track IDs reset per video and per run |
| Raw-video archiving | `privacy.store_raw_video: false` by default; input files are never copied by the pipeline |

## Built-in privacy features

1. **Anonymous IDs** — `Person #N` assigned per run; no re-identification
   model exists in the codebase (`src/tracking/` contains motion tracking
   only).
2. **Face blurring** — `privacy.blur_faces: true` (default) blurs the head
   region in the annotated output video.
3. **Data minimization in reports** — event records contain only:
   `timestamp, event, person_id, confidence, duration, frame_start, frame_end`.
   A unit test (`test_reporting_and_schema.py`) asserts reports contain no
   personal-information fields.
4. **No secrets / no telemetry** — the repository has no API keys, no
   analytics, and no network calls beyond downloading published model
   weights (Ultralytics/YOLOv8) at first use.

## Honest limitations (privacy-related)

- Even blurred footage can be sensitive. Treat annotated outputs with the
  same care as the raw footage.
- Anonymous track IDs can still reveal *patterns* (e.g. who takes breaks
  when) in context. Minimize who has access to reports.
- Behavioral inference about individuals is probabilistic and often wrong;
  see the README "Limitations" section. Never use output as the sole basis
  for disciplinary action.

## Responsible-use guidelines

1. **Lawful basis** — ensure you have a lawful basis to process the footage
   (CCTV policy, consent, legitimate interest assessment, etc. — e.g. GDPR
   in the EU, Kenya's Data Protection Act 2019, or your local equivalent).
2. **Transparency** — inform people that behavioral monitoring occurs.
   Signage and published policy are the minimum.
3. **Proportionality** — enable only the behaviors you actually need; the
   dashboard lets you switch each detector off.
4. **Human-in-the-loop** — treat every event as a *hint* for human review,
   never as a verdict. Provide an appeal/correction path.
5. **Data minimization** — delete processed outputs when the review is done.
   Do not mount personal CCTV footage into shared Docker deployments.
6. **No covert use** — do not use the system on people who have no
   reasonable expectation that monitoring occurs.
7. **Children and vulnerable groups** — extra care; consider not deploying
   at all in such contexts without explicit governance.

## Deployment checklist

- [ ] Footage source is one you are authorized to process
- [ ] Faces-blur enabled (or a documented reason it is off)
- [ ] Reports shared only with people who need them
- [ ] Raw footage storage policy decided *before* deployment
- [ ] Output retention limit set
- [ ] Users trained: "HIGH CONFIDENCE" ≠ certainty

## Reporting concerns

If you find a privacy or security issue in this codebase, please open an
issue (or a private security advisory) on the repository. Do not attach real
CCTV footage to public issues — use synthetic or licensed sample clips.
