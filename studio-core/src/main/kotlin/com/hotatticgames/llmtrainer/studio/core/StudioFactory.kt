package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.studio.api.FakeStudio
import com.hotatticgames.llmtrainer.studio.api.Studio
import java.io.File

/**
 * The ONLY entry point the Android UI uses to obtain a [Studio]. Signature is a fixed contract between the UI and the
 * core implementation; the implementation behind it is being built (currently returns the in-memory fake so the UI can
 * compile and be exercised).
 *
 * @param rootDir app-private directory for ALL studio state (e.g. context.filesDir/studio). Per-project subdirectories.
 * @param deviceSnapshotJson supplier of the host's deviceSnapshotJson() (cheap; may be called often).
 */
object StudioFactory {
    fun create(rootDir: File, deviceSnapshotJson: () -> String): Studio = FakeStudio()
}
