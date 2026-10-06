"""Provenance graph: source -> chunk -> example -> dataset -> model -> package.

Edges point from a derived node to the node(s) it was derived from.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .schemas import DatasetManifest, ExportPackage, ModelManifest, SourceManifest


def source_node(sid: str) -> str:
    return f"source:{sid}"


def chunk_node(sid: str, cid: str) -> str:
    return f"chunk:{sid}/{cid}"


def example_node(dataset_hash: str, eid: str) -> str:
    return f"example:{dataset_hash}/{eid}"


def dataset_node(h: str) -> str:
    return f"dataset:{h}"


def model_node(h: str) -> str:
    return f"model:{h}"


def package_node(h: str) -> str:
    return f"package:{h}"


@dataclass
class ProvenanceGraph:
    parents: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    children: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    nodes: dict[str, dict] = field(default_factory=dict)

    def add_node(self, node: str, **attrs) -> None:
        self.nodes.setdefault(node, {}).update(attrs)

    def add_edge(self, child: str, parent: str) -> None:
        self.add_node(child)
        self.add_node(parent)
        self.parents[child].add(parent)
        self.children[parent].add(child)

    def _walk(self, start: str, table: dict[str, set[str]]) -> set[str]:
        seen: set[str] = set()
        stack = [start]
        while stack:
            for nxt in table.get(stack.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    def ancestors(self, node: str) -> set[str]:
        return self._walk(node, self.parents)

    def descendants(self, node: str) -> set[str]:
        return self._walk(node, self.children)

    def lineage(self, node: str) -> dict[str, list[str]]:
        """Group ancestors of ``node`` by node type (the prefix before ':')."""
        out: dict[str, list[str]] = defaultdict(list)
        for a in sorted(self.ancestors(node)):
            out[a.split(":", 1)[0]].append(a)
        return dict(out)


def build_graph(
    sources: SourceManifest,
    datasets: list[DatasetManifest] = (),
    models: list[tuple[ModelManifest, DatasetManifest | str]] = (),
    packages: list[ExportPackage] = (),
) -> ProvenanceGraph:
    """Build the graph. ``models`` are (ModelManifest, dataset) pairs; the dataset may be a hash."""
    g = ProvenanceGraph()
    for s in sources.sources:
        g.add_node(source_node(s.source_id), status=s.status)
        for c in s.chunks:
            g.add_edge(chunk_node(s.source_id, c.chunk_id), source_node(s.source_id))
    for d in datasets:
        dn = dataset_node(d.content_hash)
        g.add_node(dn, dataset_id=d.dataset_id)
        for ex in d.examples:
            en = example_node(d.content_hash, ex.example_id)
            g.add_node(en, split=ex.split)
            for ref in ex.derived_from:
                g.add_edge(en, chunk_node(ref.source_id, ref.chunk_id))
                # keep the source reachable even if chunk records were cleared by a removal
                g.add_edge(chunk_node(ref.source_id, ref.chunk_id), source_node(ref.source_id))
            g.add_edge(dn, en)
    for m, ds in models:
        dh = ds.content_hash if isinstance(ds, DatasetManifest) else ds
        g.add_edge(model_node(m.content_hash), dataset_node(dh))
    for p in packages:
        g.add_edge(package_node(p.content_hash), model_node(p.model_manifest_hash))
    return g


@dataclass
class RebuildPlan:
    removed_sources: list[str]
    affected_examples: dict[str, list[str]]  # dataset_hash -> example ids
    datasets_to_rebuild: list[str]
    models_to_retrain: list[str]  # model manifest hashes whose train/validation data is affected
    models_to_reevaluate: list[str]  # only the test split is affected
    packages_to_withdraw: list[str]
    reference_index_rebuild: bool
    steps: list[str]

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def plan_removal(
    sources: SourceManifest,
    removed_source_ids: list[str],
    datasets: list[DatasetManifest],
    models: list[tuple[ModelManifest, DatasetManifest | str]] = (),
    packages: list[ExportPackage] = (),
) -> RebuildPlan:
    """Compute everything invalidated by removing sources and the ordered rebuild steps."""
    removed = set(removed_source_ids)
    known = {s.source_id for s in sources.sources}
    unknown = removed - known
    if unknown:
        raise KeyError(f"unknown source ids: {sorted(unknown)}")

    affected: dict[str, list[str]] = {}
    train_val_hit: set[str] = set()
    test_only_hit: set[str] = set()
    for d in datasets:
        hit_train = hit_test = False
        ids = []
        for ex in d.examples:
            if any(r.source_id in removed for r in ex.derived_from):
                ids.append(ex.example_id)
                if ex.split == "test":
                    hit_test = True
                else:
                    hit_train = True
        if ids:
            affected[d.content_hash] = sorted(ids)
            if hit_train:
                train_val_hit.add(d.content_hash)
            elif hit_test:
                test_only_hit.add(d.content_hash)

    retrain, reeval = [], []
    model_hashes_by_dataset: dict[str, list[str]] = defaultdict(list)
    for m, ds in models:
        dh = ds.content_hash if isinstance(ds, DatasetManifest) else ds
        model_hashes_by_dataset[dh].append(m.content_hash)
    for dh in sorted(train_val_hit):
        retrain += model_hashes_by_dataset.get(dh, [])
    for dh in sorted(test_only_hit):
        reeval += model_hashes_by_dataset.get(dh, [])

    affected_models = set(retrain) | set(reeval)
    # A reference package embeds chunk text directly, so any package carrying a reference store built from
    # the pre-removal source manifest must be withdrawn even if no model data was affected.
    withdraw = sorted(
        p.content_hash
        for p in packages
        if p.model_manifest_hash in affected_models
        or (p.reference is not None and p.source_manifest_hash == sources.content_hash)
    )
    reference_rebuild = True

    steps = [
        f"mark sources removed and delete raw + derived corpus files: {sorted(removed)}",
        "rebuild source manifest (new manifest_version)",
    ]
    if affected:
        steps.append("rebuild datasets from the new source manifest: " + ", ".join(sorted(affected)))
    if retrain:
        steps.append("retrain models whose train/validation data included removed sources: " + ", ".join(sorted(retrain)))
    if reeval:
        steps.append("re-run evaluation for models whose test split included removed sources: " + ", ".join(sorted(reeval)))
    steps.append("rebuild reference chunk store and index without removed sources")
    if withdraw:
        steps.append("withdraw and rebuild export packages: " + ", ".join(withdraw))
    return RebuildPlan(
        removed_sources=sorted(removed),
        affected_examples=affected,
        datasets_to_rebuild=sorted(affected),
        models_to_retrain=sorted(set(retrain)),
        models_to_reevaluate=sorted(set(reeval)),
        packages_to_withdraw=withdraw,
        reference_index_rebuild=reference_rebuild,
        steps=steps,
    )
