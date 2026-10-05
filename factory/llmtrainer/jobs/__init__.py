"""Studio job/results packages: import a phone-built training job, run it, export results.

Formats are specified in ``docs/studio/PACKAGE_FORMATS.md`` (v1). The desktop side never trusts a job package:
checksums, schema, leakage and test-split protections are re-verified here.
"""

from .zipio import JobRejected, ZipLimits

JOB_FORMAT = "llmtrainer-training-job"
RESULTS_FORMAT = "llmtrainer-results"
REPORT_FORMAT = "llmtrainer-evaluation-report"
FORMAT_VERSION = 1

__all__ = ["JOB_FORMAT", "RESULTS_FORMAT", "REPORT_FORMAT", "FORMAT_VERSION", "JobRejected", "ZipLimits"]
