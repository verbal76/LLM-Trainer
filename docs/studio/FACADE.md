# Studio facade (`:studio-api`)

Contract between the UI (platform Views) and `studio-core`. Package `com.hotatticgames.llmtrainer.studio.api`, pure JVM,
stdlib only. Product rules live in `PRODUCT.md`; this file maps types and per-screen call order.

## Conventions
* `Studio` calls are **blocking**; call from a background executor. Expected failures return `StudioResult.Err(StudioError)`,
  never throw. `StudioError` is sealed: `NotFound | Invalid | Blocked(reasons) | Io | Network | Cancelled | Conflict`; each has a
  stable `code` and a user-presentable `message`. `Blocked.reasons` is a list of `Blocker(code, message)` to render verbatim.
* Long-running work = `Operation` (downloads, model import). Start returns at once; poll `operation(id)`. The implementation
  persists operations; after process death a RUNNING op reloads as PAUSED with `resumable=true` (UI shows Resume).
  Ingest and dataset build are synchronous with an `onProgress` callback and persist per-file status.
* `Stages.derive(ProjectFacts)` is the single definition of stage status (`NOT_STARTED|IN_PROGRESS|NEEDS_ATTENTION|DONE|BLOCKED`)
  and next action; studio-core must use it. `ProjectSummary.nextAction` = first non-DONE stage's action.
* `FakeStudio` is the executable oracle. Test hooks (not on `Studio`): `tick()`, `simulateProcessDeath()`, `failOperation()`.
  Content hooks: evaluation stream containing `LARGE` -> claim allowed, `STUB` -> stub, else n=24 (claim NOT allowed);
  ingest names containing `corrupt`/`scan`(pdf) and identical bytes -> corrupt / needs_ocr / duplicate.

## Type map
| Area | Types |
|---|---|
| Projects | `NewProject`, `ProjectSummary`, `Stage`, `StageId`, `StageStatus`, `NextAction`, `Screen`, `ProjectFacts`, `Stages` |
| Device | `DeviceProfile`, `Recommendations`, `ProfileRecommendation` (`ProfileKind`, `Confidence{level,basis,benchmarked}`) |
| Catalog | `CatalogModel`, `ModelVariant` (`Feasibility`, `TrainingFeasibility`, `Provenance`), `LicenseInfo`, `LicenseState`, `Permission`, `Tri` |
| License flow | `LicenseTextFetch`, `LicenseAttestation` |
| Acquisition | `IntendedUse`, `AcquisitionPlan` (`allowed`, `blocking`), `Operation`, `OperationState`, `Progress` |
| Sources | `SourceInput`, `IngestReport`/`IngestItem`/`IngestIssue`/`IssueCode`, `ProvenanceSummary`, `SourceRecord`, `RightsStatus` |
| Dataset | `DatasetOptions`, `DatasetPreview`/`DatasetStats`/`DatasetStatus`, `ReviewItem`, `ReviewFilter`, `ReviewFlag`, `ChunkRole`, `ChunkOrigin` |
| Methods | `MethodOption`, `MethodIds`, `RunLocation` |
| Packages/eval | `ExportedPackage`, `EvaluationView`/`MetricRow`, `SpecialistPackageView`/`CheckResult` |
| Host | `HostHooks` (diagnostics, device snapshot, update check, restart), `UpdateStatus` |

## Calls per screen
1. **Dashboard**: `listProjects()`, `operations()` (resume banner). Open: `getProject`. Delete: `deleteProject`.
2. **Create specialist**: `createProject(NewProject)`.
3. **Device profile**: `deviceProfile()`.
4. **Recommendations**: `recommendations()` (3 profiles, show `reasons`, `warnings`, `confidence`; say "estimate" when `!benchmarked`), `catalog()`.
5. **Model detail**: `model(id)`; license evidence: `fetchLicenseText(id)` or `importLicenseText(id, name, stream)` -> show text/`sha256`/`truncated`
   -> owner ticks permissions -> `attestLicense(id, LicenseAttestation(sha, perms, attribution))` -> new `LicenseInfo`.
   Then `selectBaseModel(projectId, modelId, variantId)` (DISALLOWED is `Blocked`; UNVERIFIED selects but stage = NEEDS_ATTENTION).
6. **Acquire/import model**: `planAcquisition(variantId, IntendedUse)` -> show size, storage, `deviceCompat`, `licenseState`, `blocking`
   -> `startDownload(variantId, use, confirmed=true)` (Blocked unless `allowed` and confirmed) -> poll `operation(id)`;
   `cancelOperation`, `resumeOperation`. Alternative: `importModelFile(variantId, SourceInput)`.
7. **Add sources / Ingestion report**: `ingest(projectId, inputs, rights, onProgress)` -> `IngestReport`; later `lastIngestReport`,
   `listSources`, `setSourceRights`, `removeSource` (dataset becomes STALE).
8. **Dataset build**: `buildDataset(projectId, DatasetOptions, onProgress)` (Blocked `RIGHTS_UNSET`/`NO_SOURCES`) -> `DatasetPreview`.
9. **Dataset review**: `datasetPreview`, `reviewItems(projectId, ReviewFilter, offset, limit)`, `setIncluded(ids, bool)`, `approveDataset`
   (Blocked `LEAKAGE_UNRESOLVED` / `DATASET_STALE`).
10. **Method**: `methodOptions(projectId)` (render `honestyNote`, `whereItRuns`, grey out with `whyNotAvailable`), `selectMethod`.
11. **Training package / reference package**: `exportTrainingJobPackage(projectId, out)` (training methods only; license-gated),
    `exportReferencePackage(projectId, out)`; `out` comes from a SAF document the UI created. Result: `ExportedPackage` (sha256, files, warnings).
12. **Evaluation**: `exportHeldOutEvalSet(projectId, out)`; after desktop run `importEvaluation(projectId, stream)` -> `EvaluationView`;
    show per-metric base/specialist/delta/CI/n, `caveats`; only say "improved" when `improvementClaimAllowed`; show `claimReason` otherwise;
    banner when `isStub`. Re-read with `evaluation(projectId)`.
13. **Specialist export**: `exportSpecialistPackage(projectId, out)` (needs evaluation; warnings carry the claim caveat).
14. **Import elsewhere**: `importSpecialistPackage(stream)` -> `SpecialistPackageView` with `validation` checks.
15. **About / updates**: `host.diagnosticsJson()` (read lazily after view attach), `host.checkForUpdates(cb)`, `host.restartApp()`.

After each mutating call re-fetch `getProject(id)` to refresh `stages`.
