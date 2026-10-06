"""HAG engine test suite. Real engine, real tokenizer, real training: nothing in the ML path is mocked.

Environment:
  HAG_CLI          path to hag_cli (required)
  HAG_TEST_TMP     scratch/artifact directory (default: a fresh temp dir)
  HAG_PRETRAINED   optional: an existing small pre-trained GGUF (skips the ~minutes-long local pre-training fixture)
  HAG_CORPUS_DIRS  os.pathsep-separated dirs with *.md prose used to pre-train the tiny base (default: the pinned llama.cpp docs)
Measurements are collected in $HAG_TEST_TMP/measurements.json (uploaded as a CI artifact).
"""
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")
CLI = os.environ.get("HAG_CLI")
pytestmark = pytest.mark.skipif(not CLI or not os.path.exists(CLI or ""), reason="HAG_CLI not set / not built")

ERR = {"INVALID_ARG": 1, "IO": 2, "BAD_MODEL": 3, "OOM": 4, "CANCELLED": 5, "UNSUPPORTED": 6, "CORRUPT": 7, "INTERNAL": 8}
_MEAS = {}


def record(key, value):
    _MEAS[key] = value
    path = os.path.join(TMP, "measurements.json")
    with open(path, "w") as f:
        json.dump(_MEAS, f, indent=1, sort_keys=True)


TMP = os.environ.get("HAG_TEST_TMP") or tempfile.mkdtemp(prefix="hag-engine-test-")
os.makedirs(TMP, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def cli(*args, check=True, timeout=1800, input_threads=None):
    """Run hag_cli; returns (returncode, parsed last stdout JSON line or None, stderr)."""
    p = subprocess.run([CLI] + [str(a) for a in args], capture_output=True, text=True, timeout=timeout)
    js = None
    for line in reversed(p.stdout.strip().splitlines()):
        try:
            js = json.loads(line)
            break
        except ValueError:
            continue
    if check:
        assert p.returncode == 0, "hag_cli %s failed (%d): %s\n%s" % (args, p.returncode, p.stdout[-500:], p.stderr[-1500:])
    return p.returncode, js, p.stderr


def py(script, *args):
    return subprocess.run([sys.executable, os.path.join(TOOLS, script)] + [str(a) for a in args], capture_output=True, text=True)


def make_model(name, *flags):
    path = os.path.join(TMP, name)
    if not os.path.exists(path):
        r = py("make_tiny_gguf.py", path, *flags)
        assert r.returncode == 0, r.stderr
    return path


def nll(model, text_or_file, patch=None, threads=2):
    args = ["score", model, "--threads", threads]
    args += ["--file", text_or_file] if os.path.exists(text_or_file) else ["--text", text_or_file]
    if patch:
        args += ["--patch", patch]
    return cli(*args)[1]["mean_nll"]


def gen(model, prompt, patch=None, n=24, extra=(), threads=2):
    args = ["generate", model, "--prompt", prompt, "--n", n, "--threads", threads] + list(extra)
    if patch:
        args += ["--patch", patch]
    return cli(*args)[1]


# ---------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def pretrained():
    """A small model that has really learned some English prose (pre-trained with the engine's own trainer)."""
    env = os.environ.get("HAG_PRETRAINED")
    if env:
        return env
    out = os.path.join(TMP, "pretrained5m.gguf")
    if not os.path.exists(out):
        dirs = (os.environ.get("HAG_CORPUS_DIRS") or "").split(os.pathsep)
        dirs = [d for d in dirs if d] or [d for d in [
            os.path.join(HERE, "..", "..", ".deps", "llama.cpp", "docs"),
            os.path.join(HERE, "..", "..", "..", "docs")] if os.path.isdir(d)]
        assert dirs, "no corpus directory found (set HAG_CORPUS_DIRS)"
        t0 = time.time()
        r = py("make_pretrained_tiny.py", out, "--corpus", *dirs, "--hag-cli", CLI, "--epochs", os.environ.get("HAG_PRETRAIN_EPOCHS", "2"),
               "--threads", os.environ.get("HAG_THREADS", "2"))
        assert r.returncode == 0, r.stdout + r.stderr
        record("pretrain_wall_s", round(time.time() - t0, 1))
    return out


@pytest.fixture(scope="session")
def prose_heldout(pretrained):
    p = pretrained + ".heldout.txt"
    if os.path.exists(p):
        return p
    # externally supplied pre-trained model: any prose file in the corpus works
    alt = os.path.join(TMP, "prose_heldout.txt")
    if not os.path.exists(alt):
        with open(alt, "w") as f:
            f.write("The quick brown fox jumps over the lazy dog while the river runs past the old stone bridge.\n" * 4)
    return alt


@pytest.fixture(scope="session")
def corpus():
    d = os.path.join(TMP, "facts")
    r = py("make_fact_corpus.py", d, "--n-train", 12, "--n-heldout", 12, "--unicode")
    assert r.returncode == 0, r.stderr
    return d


# ---------------------------------------------------------------------------------------------------------------------
def test_version_and_system_info():
    _, js, _ = cli("version")
    assert js["version"].startswith("hag-engine 1; llama.cpp ")
    s = js["system"]
    assert s["abi"] in ("x86_64", "arm64-v8a", "armeabi-v7a", "x86") and s["n_cores"] >= 1 and isinstance(s["cpu_features"], list)


def test_engine_version_matches_llama_pin():
    pin = open(os.path.join(HERE, "..", "..", "llama.cpp.pin")).read().split()
    sha = [w for w in pin if len(w) == 40][0]
    assert "llama.cpp " + sha[:7] in cli("version")[1]["version"]


def test_model_load_info_and_errors():
    m = make_model("tiny.gguf", "--preset", "tiny")
    _, js, _ = cli("info", m)
    mi = js["model"]
    assert mi["arch"] == "llama" and mi["n_layer"] == 2 and mi["n_vocab"] > 256 and mi["patched"] is False and mi["chat_template"] is True
    rc, js, _ = cli("info", os.path.join(TMP, "does-not-exist.gguf"), check=False)
    assert rc == ERR["IO"] and js["ok"] is False
    junk = os.path.join(TMP, "junk.gguf")
    with open(junk, "wb") as f:
        f.write(b"GGUF" + os.urandom(200))
    rc, js, _ = cli("info", junk, check=False)
    assert rc == ERR["BAD_MODEL"]


def test_tokenize_and_chat_format():
    m = make_model("tiny.gguf", "--preset", "tiny")
    _, js, _ = cli("tokenize", m, "--text", "The ZX-91 valve")
    assert js["n"] == len(js["tokens"]) and js["tokens"][0] == 1  # BOS
    # chat template of the model itself (ChatML here)
    _, js, _ = cli("generate", m, "--prompt", "Hi there", "--chat", "--n", 1)
    assert js["n_prompt"] > 5 and js["ok"]
    # without a template the engine must refuse instead of inventing one
    nt = make_model("notemplate.gguf", "--preset", "tiny", "--no-chat-template")
    rc, js, _ = cli("generate", nt, "--prompt", "Hi", "--chat", "--n", 1, check=False)
    assert rc == ERR["UNSUPPORTED"]


def test_generation_deterministic_and_sampling(pretrained):
    a = gen(pretrained, "The model is", n=40)
    b = gen(pretrained, "The model is", n=40)
    assert a["text"] == b["text"] and a["n_generated"] == 40 and a["stop_reason"] == 1
    s1 = gen(pretrained, "The model is", n=40, extra=["--temp", 0.9, "--top-k", 40, "--top-p", 0.95, "--seed", 7])
    s2 = gen(pretrained, "The model is", n=40, extra=["--temp", 0.9, "--top-k", 40, "--top-p", 0.95, "--seed", 7])
    s3 = gen(pretrained, "The model is", n=40, extra=["--temp", 0.9, "--top-k", 40, "--top-p", 0.95, "--seed", 8])
    assert s1["text"] == s2["text"], "same seed must reproduce the stream"
    assert s1["text"] != s3["text"], "different seed should differ"
    assert a["tok_per_s"] > 0 and a["peak_rss_mb"] > 0
    record("infer_5m_tok_per_s", a["tok_per_s"])


def test_streaming_utf8_integrity_random_model():
    # a random byte-vocab model emits arbitrary byte tokens, including invalid UTF-8 sequences: the sink must never see them
    m = make_model("rand5m.gguf", "--preset", "5m", "--init-std", "0.2")
    for seed in (1, 2, 3):
        js = gen(m, "Bytes:", n=200, extra=["--temp", 1.5, "--seed", seed])
        assert js["utf8_ok"] is True
        js["text"].encode("utf-8")  # valid JSON string => valid UTF-8 text


def test_cancel_generation_and_score(pretrained, prose_heldout):
    js = gen(pretrained, "The model is", n=300, extra=["--cancel-after", 5])
    assert js["stop_reason"] == 2 and js["n_generated"] < 300
    # async cancel from another thread while a long score is running
    big = os.path.join(TMP, "big_text.txt")
    with open(big, "w") as f:
        f.write(open(prose_heldout).read() * 400)
    t0 = time.time()
    rc, js, err = cli("score", pretrained, "--file", big, "--cancel-after-ms", 300, check=False)
    dt = time.time() - t0
    assert rc == ERR["CANCELLED"], (rc, js, err)
    assert dt < 20, "cancel must return promptly (took %.1fs)" % dt


def test_context_limit_and_reset():
    m = make_model("tiny.gguf", "--preset", "tiny")
    js = cli("generate", m, "--prompt", "a" * 60, "--n", 500, "--ctx", 64, "--threads", 1)[1]
    assert js["stop_reason"] == 3 or js["n_generated"] < 500   # context full is reported, never overflows
    rc, js, _ = cli("generate", m, "--prompt", "a" * 400, "--n", 5, "--ctx", 64, check=False)
    assert rc == ERR["INVALID_ARG"]


def test_score_prefers_in_distribution(pretrained, prose_heldout):
    prose = nll(pretrained, prose_heldout)
    garbage = nll(pretrained, "qzx jvk wqp zzq xkc vbn mqz ppx jjk wqz")
    assert prose < garbage
    random_m = make_model("rand5m.gguf", "--preset", "5m", "--init-std", "0.2")
    record("nll_prose_pretrained", prose)
    record("nll_garbage_pretrained", garbage)
    assert prose < nll(random_m, prose_heldout)


# ---------------------------------------------------------------------------------------------------------------------
TRAIN = ["--ctx", 64, "--epochs", 40, "--lr", 2e-3, "--threads", 2, "--seed", 3, "--ckpt-every", 20, "--quiet"]


def train(base, corpus_dir, name, *extra, check=True, timeout=3600):
    work = os.path.join(TMP, "work_" + name)
    out = os.path.join(TMP, name + ".patch")
    shutil.rmtree(work, ignore_errors=True)   # a stale finished patch would make the run a no-op (idempotence)
    if os.path.exists(out):
        os.remove(out)
    args = ["train", base, "--data", os.path.join(corpus_dir, "train.txt"), "--work", work, "--out", out] + TRAIN + list(extra)
    rc, js, err = cli(*args, check=check, timeout=timeout)
    return rc, js, work, out


@pytest.fixture(scope="session")
def specialist(pretrained, corpus):
    """The main proof run: partial (last 2 layers) specialization of the pre-trained base on invented facts."""
    sha_before = sha256(pretrained)
    t0 = time.time()
    rc, js, work, out = train(pretrained, corpus, "spec_last2", "--last-layers", 2, "--epochs", 150)
    return {"base": pretrained, "sha_before": sha_before, "patch": out, "work": work, "wall_s": time.time() - t0, "result": js}


def probes(corpus):
    return json.load(open(os.path.join(corpus, "probes.json"), encoding="utf-8"))


# What can an unseen-entity gate honestly show?  The facts are random invented values ("The ZX-91 valve torque is 42 N-m"), so on
# NEW entities nothing about the VALUES can be learned; whole-sentence mean NLL mixes (a) format tokens that CAN be learned (entity
# names like "ZX-91 valve", the attribute phrase, the unit set) with (b) value digits, which get WORSE after training (the model
# becomes over-confident about the 12 values it memorised).  Measured: the mean NLL can therefore stay flat (CI: 3.788 -> 3.755)
# even though format NLL improves a lot.  So the gate is on the part that can respond, with the noise measured, not assumed:
#   per sentence:  prefix  = "The ZX-91 valve torque is"   value = " 42"   unit = " N-m."
#   format NLL = NLL(prefix) + NLL(unit | prefix, value)       (everything except the value digits; nats per sentence)
# Noise floor: the same recipe trained with 3 seeds; gain_i = format(base) - format(specialist_i).
# Gate: min(gain_i) > 0 and mean(gain) >= 3 * sd(gain)   (signal at least three seed-to-seed standard deviations).
# Control: training on the same sentences with the attribute->unit pairing shuffled.  It is RECORDED, not gated: at this scale the
# pairing effect is not significant (see docs/v2/TRAINING_FEASIBILITY.md), so we claim "learns the surface format of the domain",
# not "learns which unit goes with which attribute" for unseen entities.
import re as _re


def seg_nll(model, patch, sentences):
    def tot(text):
        a = ["score", model, "--text", text, "--threads", 2] + (["--patch", patch] if patch else [])
        js = cli(*a)[1]
        return js["mean_nll"] * js["n_tokens"]
    fmt = val = unit = 0.0
    for s_ in sentences:
        m = _re.match(r"(.* is)( [0-9.]+)( .*)$", s_)
        p0, v = m.group(1), m.group(2)
        t_full, t_v, t_p = tot(s_), tot(p0 + v), tot(p0)
        fmt += t_p + (t_full - t_v)
        val += t_v - t_p
        unit += t_full - t_v
    n = float(len(sentences))
    return {"format": fmt / n, "value": val / n, "unit": unit / n}


def heldout_sentences(corpus):
    return [l for l in open(os.path.join(corpus, "heldout.txt"), encoding="utf-8").read().splitlines() if l]


@pytest.fixture(scope="session")
def noise(specialist, corpus):
    import statistics
    sents = heldout_sentences(corpus)
    base = seg_nll(specialist["base"], None, sents)
    spec_main = seg_nll(specialist["base"], specialist["patch"], sents)
    gains = [base["format"] - spec_main["format"]]
    for seed in (4, 5):
        out = train(specialist["base"], corpus, "spec_seed%d" % seed, "--last-layers", 2, "--epochs", 150, "--seed", seed)[3]
        gains.append(base["format"] - seg_nll(specialist["base"], out, sents)["format"])
    # control: shuffled attribute->unit pairing
    lines = open(os.path.join(corpus, "train.txt"), encoding="utf-8").read().splitlines()
    units = [_re.match(r".* is [0-9.]+ (.*)\.$", l).group(1) for l in lines]
    shuf = units[1:] + units[:1]
    ctl_dir = os.path.join(TMP, "facts_ctl")
    os.makedirs(ctl_dir, exist_ok=True)
    open(os.path.join(ctl_dir, "train.txt"), "w", encoding="utf-8").write(
        "\n".join(_re.sub(r" ([0-9.]+) .*\.$", lambda m, u=u: " %s %s." % (m.group(1), u), l) for l, u in zip(lines, shuf)) + "\n")
    cpatch = train(specialist["base"], ctl_dir, "spec_ctl", "--last-layers", 2, "--epochs", 150)[3]
    ctl = seg_nll(specialist["base"], cpatch, sents)
    res = {"base": base, "spec": spec_main, "gains": gains, "mean": statistics.mean(gains), "sd": statistics.stdev(gains),
           "control": ctl, "control_gain_format": base["format"] - ctl["format"], "control_gain_unit": base["unit"] - ctl["unit"],
           "spec_gain_unit": base["unit"] - spec_main["unit"], "value_change": spec_main["value"] - base["value"]}
    record("format_nll_gain_per_seed", [round(g, 3) for g in gains])
    record("format_nll_noise_sd", res["sd"])
    record("format_nll_control_shuffled_units", {k: round(v, 3) for k, v in res.items() if k.startswith(("control_", "spec_gain", "value_change"))})
    return res


def assert_format_gain(gain, noise, what):
    assert gain > 3 * noise["sd"], "%s: format-NLL gain %.2f nats/sentence must exceed 3 x seed noise (sd %.2f)" % (what, gain, noise["sd"])


def test_unseen_entity_format_nll_gain_exceeds_seed_noise(noise):
    """Unseen entities: the learnable part (format) improves by more than 3 seed-to-seed sd in EVERY seed; value digits do not."""
    assert min(noise["gains"]) > 0, noise["gains"]
    assert noise["mean"] >= 3 * noise["sd"], "mean gain %.2f vs sd %.2f" % (noise["mean"], noise["sd"])
    print("\nformat gain/seed %s sd %.2f | value-digit NLL change %+.2f | control(shuffled units) unit gain %.2f vs specialist %.2f" % (
        [round(g, 2) for g in noise["gains"]], noise["sd"], noise["value_change"], noise["control_gain_unit"], noise["spec_gain_unit"]))


def test_specialization_changes_behavior_and_keeps_base_intact(specialist, corpus, prose_heldout, noise):
    base, patch = specialist["base"], specialist["patch"]
    # 1. base file untouched, patch much smaller than the model (partial tuning)
    assert sha256(base) == specialist["sha_before"]
    base_size, patch_size = os.path.getsize(base), os.path.getsize(patch)
    assert patch_size < 0.6 * base_size, (patch_size, base_size)
    _, info, _ = cli("patch-info", patch)
    t = info["train"]
    assert info["base"]["sha256"] == specialist["sha_before"]
    assert t["train_loss_last"] < 0.5 * t["train_loss_first"], "training loss must decrease"
    # 2. greedy completion of memorised fact prompts: base differs, specialist matches
    ps = probes(corpus)
    hits_base = hits_spec = 0
    non_ascii = 0
    for p in ps:
        want = p["answer"].strip()
        b = gen(base, p["prompt"], n=18)
        s = gen(base, p["prompt"], patch=patch, n=18)
        assert s["utf8_ok"] is True and b["utf8_ok"] is True
        hits_base += want in b["text"]
        ok = want in s["text"]
        hits_spec += ok
        if ok and any(ord(ch) > 127 for ch in want):
            non_ascii += 1
    record("probe_exact_match_base", "%d/%d" % (hits_base, len(ps)))
    record("probe_exact_match_specialist", "%d/%d" % (hits_spec, len(ps)))
    assert hits_base == 0
    assert hits_spec >= 0.5 * len(ps), "specialist should reproduce the facts it was trained on (%d/%d)" % (hits_spec, len(ps))
    # 3. held-out sentences from the same distribution improve; unrelated prose is measured (and reported, not hidden)
    held = os.path.join(corpus, "heldout.txt")
    nb, ns = nll(base, held), nll(base, held, patch)
    ub, us = nll(base, prose_heldout), nll(base, prose_heldout, patch)
    record("nll_heldout_facts_base", nb)
    record("nll_heldout_facts_specialist", ns)
    record("nll_unrelated_prose_base", ub)
    record("nll_unrelated_prose_specialist", us)
    record("unrelated_prose_degradation_nats", us - ub)
    record("specialize_wall_s", round(specialist["wall_s"], 1))
    # whole-sentence mean NLL is RECORDED, not gated: value digits get worse, so it can stay flat while format improves
    # (see the derivation above and test_unseen_entity_format_nll_gain_exceeds_seed_noise); no threshold was lowered, the metric changed.
    assert_format_gain(noise["base"]["format"] - noise["spec"]["format"], noise, "specialist")
    print("\nHELD-OUT FACTS NLL %.3f -> %.3f ; UNRELATED PROSE NLL %.3f -> %.3f (degradation %+.3f nats)" % (nb, ns, ub, us, us - ub))


def test_reference_consumer_merge_equals_engine_patch(specialist, corpus):
    merged = os.path.join(TMP, "merged_spec.gguf")
    r = py("apply_patch.py", specialist["base"], specialist["patch"], merged)
    assert r.returncode == 0, r.stderr
    held = os.path.join(corpus, "heldout.txt")
    assert abs(nll(merged, held) - nll(specialist["base"], held, specialist["patch"])) < 1e-6
    p = probes(corpus)[0]
    assert gen(merged, p["prompt"])["text"] == gen(specialist["base"], p["prompt"], specialist["patch"])["text"]
    # and the reference consumer rejects a wrong base too
    other = make_model("other_base.gguf", "--preset", "5m", "--seed", 99)
    assert py("apply_patch.py", other, specialist["patch"], os.path.join(TMP, "x.gguf")).returncode == 3


def test_patch_validation_rejects_wrong_base_and_corruption(specialist, corpus):
    base, patch = specialist["base"], specialist["patch"]
    other = make_model("other_base.gguf", "--preset", "5m", "--seed", 99)
    rc, js, _ = cli("info", other, "--patch", patch, check=False)
    assert rc == ERR["CORRUPT"] and "different base" in js["error"]
    # flipped payload byte
    raw = bytearray(open(patch, "rb").read())
    raw[-100] ^= 0x01
    bad = os.path.join(TMP, "flipped.patch")
    open(bad, "wb").write(raw)
    rc, js, _ = cli("info", base, "--patch", bad, check=False)
    assert rc == ERR["CORRUPT"] and "hash" in js["error"]
    # truncated
    trunc = os.path.join(TMP, "trunc.patch")
    open(trunc, "wb").write(raw[: len(raw) // 2])
    rc, js, _ = cli("info", base, "--patch", trunc, check=False)
    assert rc == ERR["CORRUPT"]
    # not a patch at all (a model file)
    rc, js, _ = cli("info", base, "--patch", base, check=False)
    assert rc == ERR["CORRUPT"]
    rc, js, _ = cli("patch-info", trunc, check=False)
    assert rc == ERR["CORRUPT"]
    # the base model is still usable and unpatched after all those rejections
    assert cli("info", base)[1]["model"]["patched"] is False


# ---------------------------------------------------------------------------------------------------------------------
def start_train(base, corpus_dir, name, *extra):
    work = os.path.join(TMP, "work_" + name)
    out = os.path.join(TMP, name + ".patch")
    shutil.rmtree(work, ignore_errors=True)
    if os.path.exists(out):
        os.remove(out)
    args = [CLI, "train", base, "--data", os.path.join(corpus_dir, "train.txt"), "--work", work, "--out", out] + [str(a) for a in TRAIN + list(extra)]
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True), work, out


def ckpt_files(work):
    if not os.path.isdir(work):
        return []
    return sorted(f for f in os.listdir(work) if f.startswith("ckpt-") and f.endswith(".bin"))


def kill_after_checkpoints(base, corpus_dir, name, n_ckpt, *extra):
    p, work, out = start_train(base, corpus_dir, name, *extra)
    deadline = time.time() + 600
    while time.time() < deadline:
        if p.poll() is not None:
            break
        if os.path.exists(os.path.join(work, "ckpt.manifest")) and len(ckpt_files(work)) >= n_ckpt:
            p.send_signal(signal.SIGKILL)
            break
        time.sleep(0.02)
    p.wait()
    assert p.returncode == -signal.SIGKILL, "training finished before it could be killed; make the run longer"
    assert not os.path.exists(out), "an interrupted run must not leave a patch behind"
    return work, out


@pytest.fixture(scope="session")
def clean_run(pretrained, corpus):
    rc, js, work, out = train(pretrained, corpus, "clean_last1", "--last-layers", 1, "--epochs", 60)
    return out


def finish(base, corpus_dir, name, work, out, *extra):
    args = ["train", base, "--data", os.path.join(corpus_dir, "train.txt"), "--work", work, "--out", out] + TRAIN + list(extra)
    return cli(*args)


def runlog(work):
    return open(os.path.join(work, "train_run.log")).read()


def test_sigkill_then_resume_gives_identical_patch(pretrained, corpus, clean_run):
    work, out = kill_after_checkpoints(pretrained, corpus, "killed", 2, "--last-layers", 1, "--epochs", 60)
    assert len(ckpt_files(work)) >= 1
    finish(pretrained, corpus, "killed", work, out, "--last-layers", 1, "--epochs", 60)
    log = runlog(work)
    assert "RESUMED from ckpt-" in log
    assert sha256(out) == sha256(clean_run), "resumed run must end with the bit-identical patch of an uninterrupted run"
    assert not ckpt_files(work), "checkpoints are removed after success"
    record("resume_bit_identical", True)


def test_corrupt_checkpoint_falls_back_then_restarts(pretrained, corpus, clean_run):
    work, out = kill_after_checkpoints(pretrained, corpus, "corrupt1", 2, "--last-layers", 1, "--epochs", 60)
    files = ckpt_files(work)
    assert len(files) >= 2, files
    newest, previous = files[-1], files[-2]
    # a partial write can never be loaded: truncate the newest checkpoint and leave a stray .tmp from a killed writer
    p = os.path.join(work, newest)
    data = open(p, "rb").read()
    open(p, "wb").write(data[: len(data) // 2])
    open(os.path.join(work, "ckpt-9999999999.bin.tmp"), "wb").write(b"junk")
    finish(pretrained, corpus, "corrupt1", work, out, "--last-layers", 1, "--epochs", 60)
    log = runlog(work)
    assert "REJECTED" in log and ("RESUMED from " + previous) in log
    assert sha256(out) == sha256(clean_run)
    # a single flipped bit in the newest checkpoint is also detected (sha256)
    work, out = kill_after_checkpoints(pretrained, corpus, "corrupt2", 2, "--last-layers", 1, "--epochs", 60)
    files = ckpt_files(work)
    p = os.path.join(work, files[-1])
    b = bytearray(open(p, "rb").read())
    b[len(b) // 2] ^= 0x10
    open(p, "wb").write(bytes(b))
    finish(pretrained, corpus, "corrupt2", work, out, "--last-layers", 1, "--epochs", 60)
    assert "REJECTED" in runlog(work)
    assert sha256(out) == sha256(clean_run)
    # every checkpoint corrupt: restart from scratch (never load garbage) and still end identically
    work, out = kill_after_checkpoints(pretrained, corpus, "corrupt3", 2, "--last-layers", 1, "--epochs", 60)
    for f in ckpt_files(work):
        pp = os.path.join(work, f)
        bb = bytearray(open(pp, "rb").read())
        bb[100] ^= 0xFF
        open(pp, "wb").write(bytes(bb))
    finish(pretrained, corpus, "corrupt3", work, out, "--last-layers", 1, "--epochs", 60)
    assert "starting from scratch" in runlog(work)
    assert sha256(out) == sha256(clean_run)


def test_cancel_keeps_checkpoint_and_resume_completes(pretrained, corpus, clean_run):
    work = os.path.join(TMP, "work_cancel")
    out = os.path.join(TMP, "cancel.patch")
    shutil.rmtree(work, ignore_errors=True)
    if os.path.exists(out):
        os.remove(out)
    args = ["train", pretrained, "--data", os.path.join(corpus, "train.txt"), "--work", work, "--out", out] + TRAIN + \
           ["--last-layers", 1, "--epochs", 60, "--cancel-after-steps", 37]
    rc, js, _ = cli(*args, check=False)
    assert rc == ERR["CANCELLED"] and not os.path.exists(out)
    assert ckpt_files(work), "cancel must keep a valid checkpoint"
    finish(pretrained, corpus, "cancel", work, out, "--last-layers", 1, "--epochs", 60)
    assert sha256(out) == sha256(clean_run)


def test_completed_run_is_idempotent(pretrained, corpus, clean_run):
    work = os.path.join(TMP, "work_clean_last1")
    t0 = time.time()
    _, js, _ = cli("train", pretrained, "--data", os.path.join(corpus, "train.txt"), "--work", work, "--out", clean_run,
                   *[str(a) for a in TRAIN], "--last-layers", 1, "--epochs", 60)
    assert js["steps"] == 0 and time.time() - t0 < 10


def test_memory_preflight_refuses_before_starting(pretrained, corpus):
    work = os.path.join(TMP, "work_oom")
    shutil.rmtree(work, ignore_errors=True)
    out = os.path.join(TMP, "oom.patch")
    rc, js, _ = cli("train", pretrained, "--data", os.path.join(corpus, "train.txt"), "--work", work, "--out", out,
                    "--ctx", 64, "--max-mem", 4_000_000, "--quiet", check=False)
    assert rc == ERR["OOM"] and "exceeds the limit" in js["error"]
    assert not os.path.exists(out) and not ckpt_files(work)
    _, est, _ = cli("estimate", pretrained, "--ctx", 64, "--max-mem", 4_000_000)
    assert est["trainable"] is True and est["fits_max_memory"] is False and est["estimated_peak_bytes"] > 4_000_000


def test_validation_split_tracked_separately(pretrained, corpus):
    work = os.path.join(TMP, "work_val")
    out = os.path.join(TMP, "val.patch")
    shutil.rmtree(work, ignore_errors=True)
    p = subprocess.run([CLI, "train", pretrained, "--data", os.path.join(corpus, "train.txt"), "--work", work, "--out", out] +
                       [str(x) for x in ("--ctx", 48, "--epochs", 6, "--lr", 2e-3, "--threads", 2, "--val", 0.25, "--last-layers", 1)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-800:]
    ev = [json.loads(l) for l in p.stderr.splitlines() if l.startswith("{")]
    vals = [e["val_loss"] for e in ev if e["phase"] == 2 and e["val_loss"] is not None]
    trains = [e for e in ev if e["phase"] == 1]
    assert len(vals) >= 2 and trains
    info = cli("patch-info", out)[1]["train"]
    assert info["n_val_docs"] == 3 and info["n_train_docs"] == 9
    assert info["val_loss_first"] == pytest.approx(vals[0], rel=1e-5)
    assert info["val_loss_last"] < info["val_loss_first"]


def test_estimate_json_fields(pretrained):
    _, est, _ = cli("estimate", pretrained, "--ctx", 128, "--last-layers", 2)
    assert est["trainable"] and est["first_trainable_layer"] == 4 and est["working_copy_needed"] is False
    assert est["bytes"]["optimizer"] == 12 * est["trainable_params"]
    full = cli("estimate", pretrained, "--ctx", 128)[1]
    assert full["trainable_params"] > est["trainable_params"] and full["estimated_peak_bytes"] > est["estimated_peak_bytes"]
    assert full["tied_embeddings"] is False and full["train_embeddings_effective"] is False
    emb = cli("estimate", pretrained, "--ctx", 128, "--emb")[1]
    assert emb["train_embeddings_effective"] is True and emb["trainable_params"] > full["trainable_params"]


# ---------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def quant_bases(pretrained):
    out = {}
    for q in ("q8_0", "q4_0"):
        out[q] = make_model("pretrained5m-%s.gguf" % q, "--preset", "5m", "--from-model", pretrained, "--quant", q)
    return out


def test_quantized_bases_run_and_train(quant_bases, pretrained, corpus, prose_heldout, noise):
    base_nll = nll(pretrained, prose_heldout)
    for q, path in quant_bases.items():
        qn = nll(path, prose_heldout)
        record("nll_prose_%s" % q, qn)
        assert qn < base_nll + 0.5, "%s base should stay close to F32 (%.3f vs %.3f)" % (q, qn, base_nll)
        assert os.path.getsize(path) < os.path.getsize(pretrained)
    q8 = quant_bases["q8_0"]
    # whole-model training needs an F32 working copy of the trainable tensors; the patch is stored back in Q8_0
    est = cli("estimate", q8, "--ctx", 64, "--last-layers", 2)[1]
    assert est["base_quantized"] and est["working_copy_needed"]
    rc, js, work, out = train(q8, corpus, "q8_last2", "--last-layers", 2, "--epochs", 60, "--val", 0.2)
    info = cli("patch-info", out)[1]
    assert {t["type"] for t in info["tensors"]} <= {"q8_0", "f32"} and any(t["type"] == "q8_0" for t in info["tensors"])
    assert os.path.getsize(out) < 0.5 * os.path.getsize(q8) + 4096
    held = os.path.join(corpus, "heldout.txt")
    nb, ns = nll(q8, held), nll(q8, held, out)
    record("q8_heldout_facts_base", nb)
    record("q8_heldout_facts_specialist", ns)
    sents = heldout_sentences(corpus)
    gq = seg_nll(q8, None, sents)["format"] - seg_nll(q8, out, sents)["format"]
    record("q8_format_nll_gain", gq)
    assert_format_gain(gq, noise, "Q8_0 base")
    # reload on the quantized base changes behaviour like on F32
    p0 = probes(corpus)[0]
    assert gen(q8, p0["prompt"], out, n=16)["text"] != gen(q8, p0["prompt"], n=16)["text"]
    # base untouched; the engine measured the cost of storing the trained weights in Q8_0
    rq = json.load(open(os.path.join(work, "requant_eval.json")))
    record("q8_requant_val_nll", rq)
    assert abs(rq["val_nll_deployed_type"] - rq["val_nll_f32_weights"]) < 0.05


def test_q4_base_training_reports_requantization_cost(quant_bases, corpus, prose_heldout):
    q4 = quant_bases["q4_0"]
    rc, js, work, out = train(q4, corpus, "q4_last2", "--last-layers", 2, "--epochs", 30, "--val", 0.2)
    rq = json.load(open(os.path.join(work, "requant_eval.json")))
    record("q4_requant_val_nll", rq)
    assert rq["val_nll_deployed_type"] > 0 and rq["val_nll_f32_weights"] > 0
    assert any(t["type"] == "q4_0" for t in cli("patch-info", out)[1]["tensors"])
    # independent consumer on a quantized base: identical bytes; scores equal exactly when the kernel choice is removed and
    # within the measured kernel-noise floor otherwise (see docs/v2/TRAINING_FEASIBILITY.md)
    merged = os.path.join(TMP, "q4_merged.gguf")
    assert py("apply_patch.py", q4, out, merged).returncode == 0
    held = os.path.join(corpus, "heldout.txt")

    lines = [l for l in open(held, encoding="utf-8").read().splitlines() if l]
    lines += [l for l in open(prose_heldout, encoding="utf-8").read().splitlines() if l][:12]

    def per_line(model, patch=None, plain=False):
        env = dict(os.environ, HAG_NO_REPACK="1") if plain else None
        out_ = []
        for l_ in lines:
            a = ["score", model, "--text", l_, "--threads", 2] + (["--patch", patch] if patch else [])
            p = subprocess.run([CLI] + [str(x) for x in a], capture_output=True, text=True, env=env)
            out_.append(json.loads(p.stdout.strip().splitlines()[-1])["mean_nll"])
        return out_
    assert per_line(merged, plain=True) == per_line(q4, out, plain=True)      # same bytes, same kernel: exactly equal
    # default layout: kernel noise. d_i = merged - (base+patch), f_i = base(repack) - base(plain) on the same lines. Both are the
    # difference of two kernel variants on the same inputs, i.e. draws of the same noise; for n = 24 equal-variance draws
    # P(rms(d) > 3 rms(f)) = P(F(24,24) > 9) < 1e-6, so rms(d) <= 3 rms(f) (+1e-6 for float resolution) is a derived, not tuned, bound.
    d = [x - y for x, y in zip(per_line(merged), per_line(q4, out))]
    f = [x - y for x, y in zip(per_line(q4), per_line(q4, plain=True))]
    rms = lambda v: (sum(x * x for x in v) / len(v)) ** 0.5
    record("q4_merge_rms_diff", rms(d))
    record("q4_merge_rms_kernel_noise", rms(f))
    assert rms(d) <= 3 * rms(f) + 1e-6, (rms(d), rms(f))


def test_untrainable_inputs_fail_cleanly(pretrained, corpus):
    work = os.path.join(TMP, "work_bad")
    out = os.path.join(TMP, "bad.patch")
    empty = os.path.join(TMP, "empty.txt")
    open(empty, "w").write("\n")
    rc, js, _ = cli("train", pretrained, "--data", empty, "--work", work, "--out", out, "--quiet", check=False)
    assert rc in (ERR["INVALID_ARG"], 66)
    rc, js, _ = cli("train", os.path.join(TMP, "nope.gguf"), "--data", os.path.join(corpus, "train.txt"), "--work", work, "--out", out, "--quiet", check=False)
    assert rc == ERR["BAD_MODEL"]


def test_training_is_deterministic_for_equal_inputs(pretrained, corpus):
    a = train(pretrained, corpus, "det_a", "--last-layers", 1, "--epochs", 5)[3]
    b = train(pretrained, corpus, "det_b", "--last-layers", 1, "--epochs", 5)[3]
    c = train(pretrained, corpus, "det_c", "--last-layers", 1, "--epochs", 5, "--seed", 4)[3]
    assert sha256(a) == sha256(b)
    assert sha256(a) != sha256(c)


# ---------------------------------------------------------------------------------------------------------------------
# LoRA mode: base frozen (may stay mmap'd / quantized, no F32 copy), patch = standard llama.cpp LoRA adapter GGUF
LORA = ["--lora-rank", 8, "--lora-alpha", 16, "--lr", 3e-3]


def test_lora_patch_on_f32_and_quantized_bases(pretrained, quant_bases, corpus, prose_heldout, noise):
    import gguf
    held = os.path.join(corpus, "heldout.txt")
    base_sha = sha256(pretrained)
    for label, base in (("f32", pretrained), ("q4_0", quant_bases["q4_0"])):
        est_l = cli("estimate", base, "--ctx", 64, "--lora-rank", 8)[1]
        est_f = cli("estimate", base, "--ctx", 64)[1]
        assert est_l["lora"] is True and est_l["working_copy_needed"] is False
        assert est_l["trainable_params"] < 0.2 * est_f["trainable_params"]
        rc, js, work, out = train(base, corpus, "lora_" + label, *LORA, "--epochs", 40)
        r = gguf.GGUFReader(out)
        assert bytes(r.fields["general.type"].parts[-1]).decode() == "adapter"       # consumable by stock llama.cpp --lora
        assert bytes(r.fields["general.architecture"].parts[-1]).decode() == "llama"
        info = cli("patch-info", out)[1]
        assert info["kind"] == "lora" and info["train"]["lora_rank"] == 8
        assert os.path.getsize(out) < 0.5 * os.path.getsize(base)
        assert info["train"]["train_loss_last"] < 0.7 * info["train"]["train_loss_first"]
        nb, ns = nll(base, held), nll(base, held, out)
        ub, us = nll(base, prose_heldout), nll(base, prose_heldout, out)
        record("lora_%s_heldout_facts" % label, [nb, ns])
        record("lora_%s_prose_degradation_nats" % label, us - ub)
        sents = heldout_sentences(corpus)
        gl = seg_nll(base, None, sents)["format"] - seg_nll(base, out, sents)["format"]
        record("lora_%s_format_nll_gain" % label, gl)
        assert_format_gain(gl, noise, "LoRA " + label)
        p0 = probes(corpus)[0]
        assert gen(base, p0["prompt"], out, n=16)["text"] != gen(base, p0["prompt"], n=16)["text"]
        raw = bytearray(open(out, "rb").read())      # tampering is rejected, base stays unpatched
        raw[-50] ^= 1
        bad = os.path.join(TMP, "lora_bad.patch")
        open(bad, "wb").write(raw)
        rc, js, _ = cli("info", base, "--patch", bad, check=False)
        assert rc == ERR["CORRUPT"]
    assert sha256(pretrained) == base_sha


def test_lora_sigkill_resume_identical(pretrained, corpus):
    extra = ["--lora-rank", 4, "--epochs", 60, "--lr", 3e-3]
    clean = train(pretrained, corpus, "lora_clean", *extra)[3]
    work, out = kill_after_checkpoints(pretrained, corpus, "lora_killed", 2, *extra)
    finish(pretrained, corpus, "lora_killed", work, out, *extra)
    assert "RESUMED from ckpt-" in runlog(work)
    assert sha256(out) == sha256(clean)
