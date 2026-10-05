# Owner physical test — LLM Trainer Studio (OTA bundle v2 on the installed v1 APK)

You already have **LLM-Trainer-v1.apk** installed. Do NOT reinstall. The new product ships as an OTA bundle.

## 1. Get the update (in the app you already have)
1. Open LLM Trainer. (The Hot Attic Games splash shows first; the v1 host's overlay defect is a known native bug, fixed
   for the next APK — it does not block the update.)
2. Scroll to **UPDATES** → tap **Check for updates**.
3. Expect: *"Update downloaded and verified: … Restart the app to apply."* (If it says up to date, the channel has not been
   published yet.)
4. Fully close the app (swipe it away) and open it again. You should see **"What this app does"** (first run), then the
   **Specialists** dashboard. Safe fallback: if the new bundle ever fails to start, the app rolls back to the last good one
   automatically; worst case it shows *safe mode* with diagnostics.

## 2. Walk the workflow (all inside the app)
1. **Create specialist** — name, domain, purpose (e.g. "Boehmer HVAC Specialist" / "Technical HVAC service and diagnostics").
2. **Device & recommendations** — your Pixel 10 Pro XL profile and three honest recommendations (all labelled *estimates*,
   low confidence, nothing benchmarked). Open the **catalog**.
3. Open a model → read its **license card** (all start UNVERIFIED) → **Select as base model**. (Selecting downloads nothing.)
4. **Review license evidence** → *fetch the license text* from the official URL (needs your internet) or import a license file →
   read it → tick the uses it permits and attest. Only then is the model VERIFIED (owner-attested, hash recorded). The app
   does not interpret legal text.
5. **Sources** → add real PDF / DOCX / TXT / MD / CSV / JSON files (system file picker). Read the ingestion report: what was
   ingested, duplicates, failures, and any PDF pages that need OCR (the app does **not** OCR).
6. **Dataset** → build → **review** the chunks (include/exclude, origin, leakage flags) → approve.
7. **Method** — read the honest list. The phone does **not** train models in this version. Pick *adapter training (desktop)*
   or *reference package only (RAG — not training)*.
8. **Export training job** → save the `.zip`. On a desktop with a GPU: `llmtrainer run-job job.zip --workspace W` (dry-run
   first; `--execute` only when you intend to spend compute). Results come back as a *results package*.
9. **Evaluation** → import the results package → see base vs specialist per metric, with sample sizes and whether an
   improvement claim is allowed (it will say **no** for stub/synthetic/small-n data).
10. **Specialist package** export/import.

## 3. What is NOT in this build (so you are not surprised)
* No on-device training and no on-device model inference (needs a native runtime → next APK).
* No OCR. Scanned PDFs are reported, not read.
* Models are not downloadable from the catalog yet (no quantized file URLs/sizes/hashes registered); selecting a base model
  works, and you can import a model file manually.
* Optimize / Benchmark stages are marked *not available yet*.

## 4. Please report
Anything that crashes, any screen that cannot be completed, and the **Updates & Diagnostics** JSON (long-press to copy).
