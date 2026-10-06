package com.hotatticgames.llmtrainer.ota

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class UpdateStoreTest {
    private fun store(dir: File = TestKit.tmp()) = UpdateStore(dir)
    private fun stage(s: UpdateStore, v: Int): SlotInfo {
        val ver = TestKit.verified(v)
        val sha = Hashing.sha256Hex(TestKit.bundle(v))
        return s.stage(ver, sha, builtinVersion = 1).getOrThrow()
    }

    @Test fun freshStoreBootsBuiltin() = assertTrue(store().planBoot() is BootPlan.Builtin)

    @Test fun stagedBundleRunsAsTrialThenPromotesOnHealthy() {
        val s = store()
        val slot = stage(s, 2)
        val plan = s.planBoot() as BootPlan.Slot
        assertTrue(plan.trial)
        assertEquals(slot.id, plan.slot.id)
        s.markHealthy(slot.id)
        val st = s.load()
        assertEquals(slot.id, st.active); assertEquals(slot.id, st.lastKnownGood); assertEquals(null, st.pending)
        val again = s.planBoot() as BootPlan.Slot
        assertFalse(again.trial)
    }

    @Test fun stagedDexIsReadOnly() {
        val s = store(); val slot = stage(s, 2)
        if (System.getProperty("user.name") != "root") { // root can always write
            assertFalse(File(s.slotDir(slot.id), "classes.dex").canWrite())
        }
    }

    @Test fun crashLoopOnTrialFallsBackToBuiltinAndQuarantines() {
        val s = store(); val slot = stage(s, 2)
        assertTrue(s.planBoot() is BootPlan.Slot)  // boot 1, "crashes" (never healthy)
        assertTrue(s.planBoot() is BootPlan.Slot)  // boot 2, crashes
        assertTrue(s.planBoot() is BootPlan.Builtin) // budget exhausted
        assertTrue(slot.id in s.load().quarantined)
        assertFalse(s.slotDir(slot.id).exists())
    }

    @Test fun badUpdateRollsBackToLastKnownGoodNotBuiltin() {
        val s = store()
        val good = stage(s, 2); s.planBoot(); s.markHealthy(good.id)
        val bad = stage(s, 3)
        val p = s.planBoot() as BootPlan.Slot
        assertEquals(bad.id, p.slot.id)
        s.reportFailure(bad.id, "entry threw")
        val next = s.planBoot() as BootPlan.Slot
        assertEquals(good.id, next.slot.id)
        assertEquals(good.id, s.load().active)
        assertTrue(bad.id in s.load().quarantined)
    }

    @Test fun healthyActiveThatLaterCrashLoopsIsEventuallyDropped() {
        val s = store(); val good = stage(s, 2); s.planBoot(); s.markHealthy(good.id)
        repeat(UpdateStore.MAX_BOOTS_ACTIVE) { assertTrue(s.planBoot() is BootPlan.Slot) }
        assertTrue(s.planBoot() is BootPlan.Builtin)
    }

    @Test fun healthyBootResetsCounter() {
        val s = store(); val a = stage(s, 2); s.planBoot(); s.markHealthy(a.id)
        repeat(10) { s.planBoot(); s.markHealthy(a.id) }
        assertTrue(s.planBoot() is BootPlan.Slot)
    }

    @Test fun quarantinedBundleCannotBeRestaged() {
        val s = store(); val a = stage(s, 2); s.reportFailure(a.id, "x")
        val r = s.stage(TestKit.verified(2), Hashing.sha256Hex(TestKit.bundle(2)), 1)
        assertTrue(r.isFailure)
    }

    @Test fun onDiskTamperingIsDetectedAndQuarantined() {
        val s = store(); val a = stage(s, 2)
        val f = File(s.slotDir(a.id), "classes.dex"); f.setWritable(true); f.writeBytes("evil".toByteArray())
        assertTrue(s.planBoot() is BootPlan.Builtin)
        assertTrue(s.load().quarantined[a.id]!!.contains("integrity"))
    }

    @Test fun downgradeAndSameVersionRefused() {
        val s = store(); val a = stage(s, 5); s.planBoot(); s.markHealthy(a.id)
        for (v in listOf(4, 5)) {
            val r = s.stage(TestKit.verified(v), Hashing.sha256Hex(TestKit.bundle(v)), 1)
            assertEquals(RejectCode.NOT_NEWER, (r.exceptionOrNull() as UpdateStore.RejectedException).reject.code)
        }
    }

    @Test fun cannotStageOlderThanBuiltin() {
        val s = store()
        val r = s.stage(TestKit.verified(2), Hashing.sha256Hex(TestKit.bundle(2)), builtinVersion = 2)
        assertTrue(r.isFailure)
    }

    @Test fun newerApkBuiltinSupersedesOldOtaSlots() {
        val s = store(); val a = stage(s, 2); s.planBoot(); s.markHealthy(a.id)
        s.reconcileBuiltin(builtinVersion = 3)
        assertTrue(s.planBoot() is BootPlan.Builtin)
        assertTrue(s.load().slots.isEmpty())
    }

    @Test fun newerStagedReplacesOlderPendingButKeepsActive() {
        val s = store(); val a = stage(s, 2); s.planBoot(); s.markHealthy(a.id)
        val p3 = stage(s, 3); val p4 = stage(s, 4)
        val st = s.load()
        assertEquals(p4.id, st.pending); assertEquals(a.id, st.active)
        assertFalse(p3.id in st.slots); assertFalse(s.slotDir(p3.id).exists())
    }

    @Test fun corruptStateFileResetsToBuiltinInsteadOfBricking() {
        val dir = TestKit.tmp(); val s = store(dir); stage(s, 2)
        File(dir, "state.json").writeText("{ not json")
        assertTrue(s.planBoot() is BootPlan.Builtin)
        assertNotNull(s.load().history.firstOrNull { it.event == "STATE_CORRUPT_RESET" } ?: s.load().history.firstOrNull())
    }

    @Test fun stateSurvivesRestartOfStoreObject() {
        val dir = TestKit.tmp(); val a = stage(store(dir), 2)
        val plan = store(dir).planBoot() as BootPlan.Slot
        assertEquals(a.id, plan.slot.id)
        // counter was persisted before "loading": a fresh store sees 1 unhealthy boot
        assertEquals(1, store(dir).load().slots[a.id]!!.bootsSinceHealthy)
    }
}
