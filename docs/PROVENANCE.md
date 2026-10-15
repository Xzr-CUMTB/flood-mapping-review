# Export provenance

This is a source export of the completed flood-mapping experiment, prepared on
2026-10-15. The original experiment directory and its frozen results were not edited.

`provenance/source_manifest.json` lists the original and exported SHA-256 for every
copied source/configuration file. All model and experiment modules are byte-identical
to their source at export. The only modified copied file is
`scripts/run_core_completion.py`: two final project-dashboard/status refresh calls
were removed. Training, inference, numerical reports and completion checks remain.

New files provide documentation, dependency specifications, ignore rules and a
`release_setup.py` helper for a fresh checkout. The unused initial value-head
prototype, exploratory diagnostics, download monitors, manuscript tools were not exported in the initial code release. Publication figure code is added on this branch.

`original_core_release_s17.json` is the historical release record. It binds data,
weights and source snapshots that are not in this repository. Its file SHA-256 is
`093a72b5ad047aa885d05b382e697192849305165ab63b226f33c1d9160ff242`.
It is not a completed-run marker for a fresh checkout. A rerun has a new release
hash, reflecting its own generated artifacts and runtime provenance.

The two historical data audit JSON files describe reference-label semantics and
retained support for the published benchmark. They contain no model test
predictions. The setup helper reuses them as documented benchmark evidence only
after checking the newly prepared manifest against the protocol's original hash
and matching its retained sample count. It does not claim a new manual label audit.

The original fixed experiment policy includes historical development observations
and storage guards. Those records are retained as protocol context; they are not
promises about a new run. Queues retain their original 76–80 GB working-set guards
and free-space checks. Inspect these guards if adapting this research code to a
different storage layout; do not modify the frozen experimental selection protocol.

`.gitattributes` preserves bytes across clone platforms so source fingerprints do
not change through automatic line-ending conversion. Numerical reproducibility
still depends on dependencies, hardware, preprocessing and training. The selected
observed dependencies are not a complete historical training environment lock.

## Publication candidate additions

The publication branch adds the 102 figure-code files listed in
`provenance/figure_code_manifest.json`; each records original and public hashes.
Five historical derivation programs had unused private task-brief copying lines
removed. The original renderers and all numerical derivations remain unchanged.
The original model/experiment modules are unchanged from the initial source export.

`publication/evaluation_core.py` contains exact AST-extracted original functions;
the extraction hashes are in `publication_evaluator_extraction.json`. Its imports
and output wrapper run on frozen sufficient statistics without requiring GPU
libraries. Actual interval outputs were compared numerically with the originals.
Absolute project paths in public JSON copies become relative paths; the payload
records these changes and separate source/public file hashes. All original
experiment and manuscript files remain untouched. The S7 transcription corrections
are explicit publication-table changes, not modifications of frozen results.
