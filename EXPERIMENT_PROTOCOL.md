# Prespecified experimental protocol

## Research question
Can a hybrid DICOM metadata security pipeline detect simulated metadata tampering in brain CT studies, and how does the metadata Autoencoder compare with structural rules and Isolation Forest?

A separate pixel-integrity experiment asks whether a trusted raw PixelData
SHA-256 reference detects controlled pixel changes and whether a metadata-only
digest remains unchanged under those pixel-only edits. Its findings must not be
pooled with the metadata attack results.

## Hypotheses
- H1: A protected SHA-256 reference detects any metadata-byte representation change relative to its trusted baseline.
- H2: Structural validation detects attacks that violate the encoded DICOM/series rules.
- H3: Isolation Forest and the metadata Autoencoder identify statistical metadata outliers not fully covered by fixed rules.
- H4: The hybrid decision rule has higher or equal attack recall than individual anomaly detectors, potentially at the cost of more false positives.

These are hypotheses, not established findings. Report results whether they support or reject them.

## Unit of analysis and split
The pipeline operates at DICOM-instance (file) level, while partitioning groups by PatientID (StudyInstanceUID fallback). Slices from one group must not cross train, validation and test partitions. Audit grouping against dataset documentation.

## Training and calibration
- Isolation Forest: clean training partition only; its score threshold is calibrated to the 1st percentile of clean validation scores (nominal 1% calibration-tail rate).
- Autoencoder: clean training partition only.
- Autoencoder threshold: percentile of reconstruction errors on clean validation partition.
- Test partition: never used for model fitting or threshold selection. Report file-level and study-level metrics because slices within a study are correlated.
- Trusted baseline: clean held-out test partition, protected independently.

## Test conditions
Six attack variants are generated from copies of held-out test files by selecting studies, not random slices. The UID-inconsistency attack changes one instance per selected series; other attacks modify applicable selected instances. Untouched files in each attack output act as clean controls. The attack log supplies file-level ground truth, and study-level analysis marks a study positive when any instance was modified. Do not describe simulated tampering as proof of malicious intent.

## Comparators
SHA-256 metadata integrity; structural rules; Isolation Forest; Autoencoder; IF + Autoencoder; rules + IF; complete hybrid.

## Outcomes
Primary: precision, recall and F1 by attack type.
Secondary: accuracy, false-positive rate, confusion counts, clean held-out test FPR and runtime.
Report class balance and sample counts alongside metrics. Macro-average across attack types is not the same as pooling all instances.

## Threat model and limitations
The reference manifest is trusted and inaccessible to the attacker. This system does not protect a compromised manifest, authenticate the sender, validate clinical correctness, or guarantee resistance to adaptive attackers. Hash mismatches indicate changes, not maliciousness. Anomaly flags indicate statistical deviation, not proof of attack. This is not a clinical diagnostic or deployment system.

Optional pixel experiment: `scripts/run_pixel_integrity_experiment.py` generates localized-patch, sparse low-amplitude, and single-slice inconsistency copies and compares raw PixelData hashes with metadata hashes. It requires a trusted clean reference and is not an image-anomaly or localization model. Keep its metrics separate from the metadata experiment.

## Added anomaly-model extensions

The kNN metadata baseline is implemented in `src/knn_anomaly.py` and
`scripts/run_knn_metadata_experiment.py`. It reuses the current metadata
feature representation, fits only clean training instances, scores by mean
distance to the k nearest training instances, and calibrates its alarm cutoff
on clean validation scores. It supplements rather than replaces Isolation
Forest and the metadata autoencoder. Its nominal file-level validation-tail
rate is not a study-level false-alarm guarantee.

The optional PatchCore-style pixel baseline is implemented in `src/patchcore.py`
and `scripts/run_patchcore_experiment.py`. It uses frozen ResNet-18 layer2/layer3
features, default ImageNet weights, grayscale channel replication, and a
deterministically random-subsampled memory bank from clean training patches.
This is a PatchCore-style implementation and does not reproduce the paper's
k-center coreset selection. `torchvision` must match the installed PyTorch
build. The transferred ImageNet representation is an assumption to evaluate on
CT, not a validated medical-image feature extractor. Use `--weights none` only
to debug execution, not as the pretrained baseline.

Late fusion is implemented in `scripts/run_late_fusion.py`. Fit and score both
branches on the same official folds and exactly the same paired attack folders.
For `combined_attacks`, pass the existing `combined_predictions.csv` to both
branch runners. The folder-level `attack_log.json` labels pixel-edited files;
the combined CSV additionally supplies metadata-only and union labels. The
branch outputs retain all three targets so each branch can be assessed against
its attack family and the union.

The fusion script joins file keys and labels, uses equal fixed weights,
normalizes scores using clean validation medians and IQRs, and calibrates the
score-fusion cutoff on clean validation scores. It also reports OR and AND
branch decisions. Attack labels do not determine fusion weights or thresholds.
This evaluates complementary anomaly signals; it does not implement
remediation or clinical defense. No outcome from these extensions should be
included in the manuscript until runs complete and per-attack outputs are
inspected.

The pixel experiment also stores packed synthetic edit masks and reports overlap between those masks and the nonzero clean-reference difference map. This is a mask/difference alignment check, not learned localization performance. `--severity-sweep` evaluates prespecified subtle/default/stronger edit settings. `scripts/run_combined_integrity_experiment.py` pairs selected metadata and pixel edits and reports the two reference checks separately and as an OR comparator. Do not describe these known-original hash results as no-reference detection or as a learned multimodal detector. Results from these extensions should be included in the manuscript only after those runs have been executed and their CSV outputs reviewed.
