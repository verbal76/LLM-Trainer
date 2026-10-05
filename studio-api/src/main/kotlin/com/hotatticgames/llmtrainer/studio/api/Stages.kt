package com.hotatticgames.llmtrainer.studio.api

/** Everything stage derivation needs; the implementation fills it from persisted project state. */
data class ProjectFacts(
    val baseModelId: String? = null,
    val baseModelLicense: LicenseState? = null,
    val sourcesIngested: Int = 0,
    val sourcesFailed: Int = 0,
    val dataset: DatasetStatus = DatasetStatus.NONE,
    val datasetFlaggedIncluded: Int = 0,
    val method: MethodOption? = null,
    val trainingPackageExported: Boolean = false,
    val referencePackageExported: Boolean = false,
    val heldOutExported: Boolean = false,
    val evaluation: EvaluationView? = null,
    val specialistExported: Boolean = false,
)

/** Single shared definition of stage status so UI, FakeStudio and studio-core agree. */
object Stages {
    fun derive(f: ProjectFacts): List<Stage> {
        val base = base(f)
        val sources = sources(f)
        val dataset = dataset(f)
        val method = method(f, dataset)
        val training = training(f, method)
        val eval = evaluation(f, dataset)
        val export = export(f)
        return listOf(base, sources, dataset, method, training, eval, export)
    }

    fun nextAction(stages: List<Stage>): NextAction? = stages.firstOrNull { it.status != StageStatus.DONE }?.nextAction

    private fun st(id: StageId, s: StageStatus, summary: String, label: String?, screen: Screen?) =
        Stage(id, s, summary, if (label == null) null else NextAction(label, screen))

    private fun base(f: ProjectFacts) = when {
        f.baseModelId == null -> st(StageId.BASE_MODEL, StageStatus.NOT_STARTED, "No base model selected", "Choose a base model", Screen.RECOMMENDATIONS)
        f.baseModelLicense == LicenseState.DISALLOWED -> st(StageId.BASE_MODEL, StageStatus.BLOCKED, "License disallows this use", "Choose a different base model", Screen.RECOMMENDATIONS)
        f.baseModelLicense != LicenseState.VERIFIED -> st(StageId.BASE_MODEL, StageStatus.NEEDS_ATTENTION, "License not verified", "Review and verify the license", Screen.MODEL_DETAIL)
        else -> st(StageId.BASE_MODEL, StageStatus.DONE, "Base model selected, license verified", null, null)
    }

    private fun sources(f: ProjectFacts) = when {
        f.sourcesIngested == 0 && f.sourcesFailed == 0 -> st(StageId.SOURCES, StageStatus.NOT_STARTED, "No sources yet", "Add sources", Screen.ADD_SOURCES)
        f.sourcesFailed > 0 -> st(StageId.SOURCES, StageStatus.NEEDS_ATTENTION, "${f.sourcesIngested} ingested, ${f.sourcesFailed} need attention", "Review ingestion report", Screen.INGESTION_REPORT)
        else -> st(StageId.SOURCES, StageStatus.DONE, "${f.sourcesIngested} sources ingested", null, null)
    }

    private fun dataset(f: ProjectFacts): Stage = when {
        f.sourcesIngested == 0 -> st(StageId.DATASET, StageStatus.BLOCKED, "Needs ingested sources", "Add sources", Screen.ADD_SOURCES)
        f.dataset == DatasetStatus.NONE -> st(StageId.DATASET, StageStatus.NOT_STARTED, "Dataset not built", "Build dataset", Screen.DATASET_BUILD)
        f.dataset == DatasetStatus.STALE -> st(StageId.DATASET, StageStatus.NEEDS_ATTENTION, "Sources changed; dataset is stale", "Rebuild dataset", Screen.DATASET_BUILD)
        f.dataset == DatasetStatus.APPROVED -> st(StageId.DATASET, StageStatus.DONE, "Dataset approved", null, null)
        else -> st(StageId.DATASET, StageStatus.NEEDS_ATTENTION, "${f.datasetFlaggedIncluded} flagged items awaiting review", "Review and approve dataset", Screen.DATASET_REVIEW)
    }

    private fun method(f: ProjectFacts, dataset: Stage) = when {
        dataset.status != StageStatus.DONE -> st(StageId.METHOD, StageStatus.BLOCKED, "Approve the dataset first", "Review dataset", Screen.DATASET_REVIEW)
        f.method == null -> st(StageId.METHOD, StageStatus.NOT_STARTED, "No method chosen", "Choose a method", Screen.METHOD)
        else -> st(StageId.METHOD, StageStatus.DONE, f.method.label, null, null)
    }

    private fun training(f: ProjectFacts, method: Stage) = when {
        method.status != StageStatus.DONE -> st(StageId.TRAINING_PACKAGE, StageStatus.BLOCKED, "Choose a method first", "Choose a method", Screen.METHOD)
        f.method!!.isTraining ->
            if (f.trainingPackageExported) st(StageId.TRAINING_PACKAGE, StageStatus.DONE, "Training job package exported; training runs on the desktop", null, null)
            else st(StageId.TRAINING_PACKAGE, StageStatus.NOT_STARTED, "Training job package not exported", "Export training job package", Screen.TRAINING_PACKAGE)
        else ->
            if (f.referencePackageExported) st(StageId.TRAINING_PACKAGE, StageStatus.DONE, "Reference package exported (not training)", null, null)
            else st(StageId.TRAINING_PACKAGE, StageStatus.NOT_STARTED, "Reference package not exported", "Export reference package", Screen.TRAINING_PACKAGE)
    }

    private fun evaluation(f: ProjectFacts, dataset: Stage): Stage {
        val e = f.evaluation
        return when {
            e != null && e.isStub -> st(StageId.EVALUATION, StageStatus.NEEDS_ATTENTION, "Imported results are a stub, not real evaluation", "Review evaluation", Screen.EVALUATION)
            e != null && !e.improvementClaimAllowed -> st(StageId.EVALUATION, StageStatus.NEEDS_ATTENTION, "Evidence does not support an improvement claim: ${e.claimReason}", "Review evaluation", Screen.EVALUATION)
            e != null -> st(StageId.EVALUATION, StageStatus.DONE, "Evaluated against the base model on held-out material", null, null)
            dataset.status != StageStatus.DONE -> st(StageId.EVALUATION, StageStatus.BLOCKED, "Approve the dataset first", "Review dataset", Screen.DATASET_REVIEW)
            f.heldOutExported -> st(StageId.EVALUATION, StageStatus.IN_PROGRESS, "Waiting for desktop evaluation results", "Import evaluation results", Screen.EVALUATION)
            else -> st(StageId.EVALUATION, StageStatus.NOT_STARTED, "Held-out set not exported", "Export held-out eval set", Screen.EVALUATION)
        }
    }

    private fun export(f: ProjectFacts) = when {
        f.evaluation == null -> st(StageId.EXPORT, StageStatus.BLOCKED, "Needs evaluation results", "Evaluate first", Screen.EVALUATION)
        f.specialistExported -> st(StageId.EXPORT, StageStatus.DONE, "Specialist package exported", null, null)
        else -> st(StageId.EXPORT, StageStatus.NOT_STARTED, "Not exported", "Export specialist package", Screen.EXPORT)
    }
}
