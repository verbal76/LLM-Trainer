package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.studio.api.HostHooks
import com.hotatticgames.llmtrainer.studio.api.Studio
import java.io.File

/**
 * The ONLY entry point the Android UI uses to obtain a [Studio]. Signature is a fixed contract between the UI and the
 * core implementation.
 *
 * @param rootDir app-private directory for ALL studio state (e.g. context.filesDir/studio). Per-project subdirectories.
 * @param deviceSnapshotJson supplier of the host's deviceSnapshotJson() (cheap; may be called often).
 */
object StudioFactory {
    fun create(rootDir: File, deviceSnapshotJson: () -> String): Studio = StudioCore(rootDir, deviceSnapshotJson)

    /** Same, but wires the host's diagnostics/update/restart hooks into [Studio.host]. */
    fun createWithHost(rootDir: File, deviceSnapshotJson: () -> String, host: HostHooks): Studio =
        StudioCore(rootDir, deviceSnapshotJson, hostHooks = host)
}
