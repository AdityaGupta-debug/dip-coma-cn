# Brain CT DICOM Metadata and Pixel Integrity Security Research

**Research title:** Detecting DICOM Metadata Tampering in Non-Contrast Brain CT: A Comparative Evaluation of Integrity, Rules, and Anomaly Detection

This version is designed for a small real brain-CT cohort and metadata-only processing. It does not require CQ500's 28.66 GB download.

## Dataset: CPAISD

Use **CPAISD: Core-Penumbra Acute Ischemic Stroke Dataset**.


- 112 non-contrast cranial CT studies.
- 92 training studies (8,376 slices), 10 validation studies (980 slices), 10 test studies (809 slices).
- DICOM slices (`raw.dcm`) plus image/mask arrays and metadata.
- The Zenodo archive is 5.6 GB.
- Dataset DOI: https://doi.org/10.5281/zenodo.10892316
- Download: https://zenodo.org/records/10892316
- The Zenodo record describes the scans as anonymized using Kitware DicomAnonymizer, with selected demographic/scanner attributes retained.
- CPAISD contains hyperacute ischemic stroke cases from one center; it is not a general healthy/abnormal brain CT population.

Dataset facts and split counts are from the Zenodo record. Check the record's current license and terms before use/publication; do not infer reuse rights from anonymization alone. Cite Umerenkov, Kudin, Peksheva & Pavlov, CPAISD, Zenodo, DOI 10.5281/zenodo.10892316.

## Why this is much lighter

Default selection uses all official folds:
- 92 complete training studies
- 10 complete validation studies
- 10 complete test studies

The selector preserves the dataset's official train/validation/test fold boundaries and chooses whole studies, never random slices. It uses hard links when possible, so the subset does not duplicate the DICOM image bytes on the same NTFS volume.

The metadata ML models use DICOM headers only. The metadata SHA-256 reference covers main-dataset and DICOM file-meta elements, excluding PixelData and group-length elements. The separate pixel-integrity experiment decodes and hashes PixelData; it is not part of the metadata-only training run.

## Windows setup

Download and extract the Zenodo `dataset.zip`. You need the extracted folders `train`, `val`, and `test`. Open CMD in the project folder:

```cmd
uv venv
uv pip install -r requirements.txt
```

## Select the small study subset

Assuming the extracted CPAISD folder is `D:\Datasets\CPAISD`:

```cmd
uv run python scripts/select_cpaisd_subset.py --input D:\Datasets\CPAISD --output data/real_ct --seed 42
```

The script creates:
```text
data/real_ct/
  train/
  val/
  test/
  subset_summary.json
```

It uses hard links if source and destination are on the same NTFS volume. If hard links are unavailable, it copies the files. Keep the extracted original until the subset creation succeeds and verify the subset before deleting anything.

## Verify the selected DICOM data

```cmd
uv run python scripts/check_dataset.py --input data/real_ct/train
uv run python scripts/check_dataset.py --input data/real_ct/val
uv run python scripts/check_dataset.py --input data/real_ct/test
```

## Run the entire experiment

```cmd
uv run python scripts/run_research_experiment.py --input data/real_ct --workdir experiment_cpaisd --seed 42 --epochs 40 --attack-fraction 0.20
```

Or run:
```cmd
RUN_REAL_CPAISD.cmd
```

## What the pipeline does

1. Checks the three official folds.
2. Trains Isolation Forest on clean training metadata and calibrates its score threshold on clean validation metadata (1st percentile; nominal 1% calibration-tail rate).
3. Trains the metadata Autoencoder on clean training metadata and calibrates its threshold on validation metadata.
4. Builds a metadata SHA-256 baseline on clean test files.
5. Measures false positives on untouched clean test data.
6. Creates eight controlled metadata attack conditions on copies of test files, selecting studies rather than individual slices. PatientID and StudyDescription are inserted because those attributes are absent in this CPAISD test fold; two added conditions make same-length substitutions in populated ProtocolName and Manufacturer fields.
7. Runs the hybrid detector and ablation.
8. Writes per-attack instance-, study-, and series-level metrics, macro averages, clean false-positive rates, plots, and a report.

## Outputs

```text
experiment_cpaisd/
  models/
  attacks/
    patient_id/
    study_description/
    protocol_name_same_length/
    manufacturer_same_length/
    pixel_spacing/
    study_uid_inconsistency/
    remove_protocol/
    modality/
  reports/
    test_baseline.json
    clean_test_predictions.csv
    clean_test_false_positive_rates.csv
    clean_test_study_false_positive_rates.csv
    per_attack_metrics.csv
    per_attack_study_metrics.csv
    macro_average_metrics.csv
    macro_average_study_metrics.csv
    f1_by_attack.png
    clean_test_fpr.png
    experiment_config.json
    RESEARCH_REPORT.md
```

## Separate pixel-integrity experiment

The main experiment above is metadata-only. To run the additional pixel-level
experiment after preparing the clean CPAISD folds, use a **new, empty output
directory** so its outputs remain separate from the metadata results:

```cmd
uv run python scripts/run_pixel_integrity_experiment.py --input data/cpaisd/test --output experiment_cpaisd_pixel --attack-fraction 0.20 --seed 42
```

It creates three reproducible synthetic pixel-only conditions on copies of
selected test studies: a central localized patch, sparse low-amplitude pixel
perturbations, and one altered slice per selected series. It compares a raw
PixelData SHA-256 reference with metadata SHA-256 as a negative control. The
hash baseline requires a trusted reference and is not an image anomaly model.
Edits are applied to stored integer pixel values without applying the DICOM
rescale slope/intercept; the defaults are +32 on a central 5x5 patch, +1 to a
seeded 1% of pixels, and +8 on one 5x5 patch per selected series, respectively.
The initial editor supports uncompressed single-frame 2-D integer CT images;
compressed or other pixel encodings stop with an explanatory error. Original
dataset files are not modified.

Outputs are written under `experiment_cpaisd_pixel/`: `attacks/`,
`clean_pixel_baseline.json`, `pixel_predictions.csv`, `pixel_metrics.csv`,
`pixel_difference_summary.csv`, representative images under `heatmaps/`,
`pixel_localization_metrics.csv`, packed per-instance masks under each attack's
`ground_truth_masks/`, `experiment_config.json`, and `PIXEL_EXPERIMENT_REPORT.md`.
The localization check thresholds the exact clean-reference difference at zero
and compares it with the synthetic edit mask; it validates mask/difference
alignment, not detector localization performance. The per-file CSV
includes changed-pixel count/fraction, mean absolute difference, RMSE, and
maximum difference; the summary CSV reports means over modified images.
Heatmaps show clean reference, modified image,
and absolute difference; color scaling is per image, so use the CSV for numeric
comparison. Set `--heatmaps-per-attack 0` to disable image generation. Keep this as a
separate experiment from the metadata metrics; pixel hashing detects change
against a clean reference, while it does not identify whether a change is
malicious or clinically meaningful.

## Train a no-reference pixel anomaly baseline

The patch-level Isolation Forest uses clean training images, calibrates
thresholds on clean validation images, and evaluates the saved synthetic pixel
masks. It scores each image independently and does not compare against its clean
original during prediction:

```cmd
uv run python scripts/run_no_reference_pixel_experiment.py --train data/cpaisd/train --validation data/cpaisd/val --test data/cpaisd/test --attacks-root experiment_cpaisd_pixel/attacks --output experiment_cpaisd_no_reference_pixel --seed 42
```

To reuse that trained patch Isolation Forest while adding a median-filter
residual comparator (thresholds are recalibrated on validation data), run it
into a new output folder:

```cmd
uv run python scripts/run_no_reference_pixel_experiment.py --train data/cpaisd/train --validation data/cpaisd/val --test data/cpaisd/test --attacks-root experiment_cpaisd_pixel_masks/attacks --output experiment_cpaisd_no_reference_v2 --seed 42 --pretrained-model experiment_cpaisd_no_reference_pixel/no_reference_pixel_model.joblib
```

## Repeat metadata attacks across seeds

This repeats four representative attacks with different study-selection seeds
and reports per-run metrics plus descriptive study- and series-level bootstrap
intervals. The intervals are based on the ten CPAISD test studies and are not
external-population confidence intervals. Choose a new output folder:

```cmd
uv run python scripts/run_seed_sensitivity.py --input data/cpaisd/test --baseline experiment_cpaisd/research_extensions_v2/test_baseline.json --metadata-model experiment_cpaisd/models/isolation_forest.joblib --autoencoder-model experiment_cpaisd/models/metadata_autoencoder.joblib --output experiment_cpaisd_seed_sensitivity --seeds 42 43 44 --attacks modality pixel_spacing manufacturer_same_length study_uid_inconsistency --attack-fraction 0.20 --bootstrap-iterations 2000
```

After that completes, regenerate stratified bootstrap intervals from the saved
study/series predictions (resampling positive and negative groups separately):

```cmd
uv run python scripts/summarize_seed_sensitivity.py --input experiment_cpaisd_seed_sensitivity --iterations 2000
```

## Live classroom demo

Launch the interactive dashboard:

```cmd
uv run streamlit run dashboard/app.py
```

The app automatically loads `data/cpaisd/test`, the trained metadata models,
the saved no-reference pixel model, and the completed 30-epoch U-Net checkpoint
from `experiment_cpaisd_supervised_localization_512_30ep_complete` when available. No upload or
sidebar path setup is needed. In **Interactive attack lab**, choose a slice and choose **Metadata
only**, **Pixels only**, or **Metadata + pixels**. Pick the edit and click
**Apply change and show everything**. The app displays all implemented checks
side by side plus clean/edited CT images, a reference-difference heatmap, the
U-Net probability heatmap, and its thresholded mask overlay. A sensitivity
slider updates the mask and alert, reports single-slice precision/recall and
clean-slice alarm fraction, and offers a downloadable five-threshold CSV. Those
single-slice measures are for demonstration, not test-set estimates. Pixel
options include a localized patch, sparse edit, and a copy-move patch from the U-Net's
withheld synthetic attack family. The **Experiments completed** tab
shows saved metric tables and U-Net localization examples. It edits only temporary copies; the preloaded
dataset remains unchanged. These are controlled examples, not judgments about
intent or clinical harm. **Experiments completed** shows saved CSV result
tables from prior runs.

## New research extensions

These extensions have already been run in this workspace; their result folders
are `experiment_cpaisd_cross_layer`, `experiment_cpaisd_adaptive_evasion`,
`experiment_cpaisd_benign_vs_attack`, and `experiment_cpaisd_study_calibration`.
The commands below are for reproduction. Each script requires an empty output
folder, so use a different `--output` name when rerunning.

### Supervised pixel-tamper localization with an unseen attack family

This experiment trains a compact U-Net on synthetic localized-patch and
sparse-pixel masks from the clean training fold. It calibrates pixel and image
thresholds on clean validation images, then evaluates on held-out studies. A
copy-move edit family is deliberately withheld from training to measure transfer
to one unseen synthetic edit type. The model sees each image alone at test time;
it does not use a clean-reference image or hash. Results use a resized 256x256
model grid, not native-resolution clinical masks.

For a longer CPU run, limit training to 512 slices and allow up to 30 epochs.
At every epoch the runner scores a fixed synthetic-mask validation subset and
saves the checkpoint with the best macro pixel IoU; the test fold is not used
to choose the epoch:

```cmd
uv run python scripts/run_supervised_pixel_localization.py --train data/cpaisd/train --validation data/cpaisd/val --test data/cpaisd/test --output experiment_cpaisd_supervised_localization_512_30ep --max-train-files 512 --epochs 30 --batch-size 8 --image-size 256 --seed 42
```

For the full official training fold, use `--max-train-files 0` (or omit the
option) and keep in mind that CPU runtime will rise substantially. The
test fold remains the full official test set unless `--max-test-files` is set.
Outputs include the checkpoint, training loss, file/study metrics, pixel Dice
and IoU by epoch on synthetic validation masks, the selected best epoch, file/study
test metrics, and example clean/edit/ground-truth/predicted maps. Do not report
results until the run completes. The withheld copy-move test remains synthetic,
and the test fold contains only ten studies.

### Header-to-pixel layout consistency

This checks whether declared DICOM rows/columns, frames, sample layout, and
stored pixel payload agree. It uses no clean image or hash reference. The
pixel-only attack is a negative control because it changes pixel values while
preserving their layout:

```cmd
uv run python scripts/run_cross_layer_consistency_experiment.py --input data/cpaisd/test --output experiment_cpaisd_cross_layer --pixel-negative-control experiment_cpaisd_pixel_masks/attacks/low_amplitude --attack-fraction 0.20 --seed 42
```

### Feature-aware adaptive evasion

This tests a same-length ProtocolName substitution chosen to collide with the
metadata feature representation. Reference hashing is retained as a control:

```cmd
uv run python scripts/run_adaptive_metadata_evasion.py --input data/cpaisd/test --baseline experiment_cpaisd/research_extensions_v2/test_baseline.json --metadata-model experiment_cpaisd/models/isolation_forest.joblib --autoencoder-model experiment_cpaisd/models/metadata_autoencoder.joblib --output experiment_cpaisd_adaptive_evasion --attack-fraction 0.20 --seed 42
```

### Compare benign-change alarms with attack detection

Run a broader controlled comparison with three benign-like proxies (a file-meta
implementation-version update, limited privacy-field cleanup, and lossless
re-serialization) and the existing scripted attacks. The script copies source
files, checks whether pixels and non-target clinical attributes stay unchanged,
and writes detector alarm rates. These transformations label the experiment's
conditions; they do not tell whether a real-world change was malicious:

```cmd
uv run python -m src.cli baseline --input data/cpaisd/test --manifest experiment_cpaisd_benign_attack_baseline.json
uv run python scripts/run_benign_vs_attack_experiment.py --input data/cpaisd/test --manifest experiment_cpaisd_benign_attack_baseline.json --model experiment_cpaisd/models/isolation_forest.joblib --ae-model experiment_cpaisd/models/metadata_autoencoder.joblib --attacks-root experiment_cpaisd/attacks --output experiment_cpaisd_benign_attack_v3
```

Outputs include `benign_vs_attack_summary.csv`,
`benign_transformation_audit.csv`, `benign_transformation_audit_summary.csv`,
and `BENIGN_VS_ATTACK_REPORT.md`. The runner verifies that the manifest matches
the current clean input and compares benign alarms with the same files' clean
predictions. The
privacy-field cleanup is a limited synthetic proxy, not a complete
standards-conformant de-identification process. Use a fresh `--output` folder
for every run.

### Calibrate alerts at study level

This uses the validation fold to calibrate study-level Isolation Forest and
autoencoder anomaly scores, then reports clean-test and saved attack results.
With only ten validation studies, conformal threshold resolution is coarse:

```cmd
uv run python scripts/run_study_level_calibration.py --validation data/cpaisd/val --test data/cpaisd/test --test-baseline experiment_cpaisd/research_extensions_v2/test_baseline.json --metadata-model experiment_cpaisd/models/isolation_forest.joblib --autoencoder-model experiment_cpaisd/models/metadata_autoencoder.joblib --attack-predictions-root experiment_cpaisd_seed_sensitivity/evaluations --output experiment_cpaisd_study_calibration
```

The standard detector predictions now also include `alert_sources`, a readable
`alert_explanation`, and the autoencoder's largest feature-wise reconstruction
errors. Hash alerts say that a reference digest differs; the hash itself cannot
identify the altered tag.

It saves image alarm metrics, coarse 16x16-block anomaly maps, mask-based
localization scores, and clean-test alarm rates. This handcrafted baseline is
exploratory and can miss subtle changes; use the measured results rather than
assuming that an added image detector improves detection.

To run a predefined subtle/default/stronger perturbation sweep in a separate
fresh directory:

```cmd
uv run python scripts/run_pixel_integrity_experiment.py --input data/cpaisd/test --output experiment_cpaisd_pixel_sweep --attack-fraction 0.20 --seed 42 --severity-sweep
```

The sweep settings are recorded per severity subfolder; its top-level
`severity_sweep_metrics.csv` collects the hash-based results. To evaluate
paired metadata and pixel edits, run:

```cmd
uv run python scripts/run_combined_integrity_experiment.py --input data/cpaisd/test --output experiment_cpaisd_combined --metadata-attacks patient_id modality --pixel-attacks localized_patch low_amplitude --attack-fraction 0.20 --seed 42
```

Combined outputs include per-file predictions and file-/study-level metrics
for metadata hash, pixel hash, and their OR. These remain trusted-reference
comparators and are not a learned multimodal detector. Metadata and pixel
attacks use independent study-selection seeds (`seed` and `seed + 2`) so the
test can contain metadata-only and pixel-only positive instances; inspect the
logged overlap rather than assuming the selected studies are disjoint for every
dataset or seed. Each experiment needs a fresh output directory; the scripts
preserve existing results.

## Evaluate on a second DICOM cohort

To measure transfer of the CPAISD-trained metadata models, place a separate
cohort such as CQ500 in a folder containing readable DICOM files, then run:

```cmd
uv run python scripts/run_external_dataset_evaluation.py --input D:\Datasets\CQ500 --output experiment_cq500_external --metadata-model experiment_cpaisd\models\isolation_forest.joblib --autoencoder-model experiment_cpaisd\models\metadata_autoencoder.joblib --attacks modality pixel_spacing manufacturer_same_length --attack-fraction 0.20 --seed 42
```

This reports alarms on untouched external files and performance on controlled
edits. The script builds a clean external reference for hash comparisons, so
hash performance on those edits is expected; use the rules and learned-model
results to inspect transfer behavior. CQ500 is a hemorrhage head-CT cohort,
not an ischemic-stroke cohort. Follow its non-commercial license and end-user
agreement. No CQ500 results are included until the external data are supplied
and the evaluation is run.

To measure alarms from a benign-like DICOM file-meta update, run the following
after the main experiment has created the baseline and models:

```cmd
uv run python scripts/run_benign_reserialization.py --input data/cpaisd/test --output experiment_cpaisd_reserialization --manifest experiment_cpaisd/reports/test_baseline.json --model experiment_cpaisd/models/isolation_forest.joblib --ae-model experiment_cpaisd/models/metadata_autoencoder.joblib
```

This control changes `ImplementationVersionName` to a synthetic receiver
identifier while keeping the clinical dataset attributes and PixelData fixed.
It measures strict-hash sensitivity to a benign-like header update; it is not a
simulation of a named PACS product. Use a fresh output path for each run.

## Research limitations to state honestly

- The main experiment is metadata manipulation detection; the separate pixel experiment does not train an image anomaly detector.
- CPAISD is single-center and contains hyperacute ischemic stroke cases only.
- For quick development, the selector accepts smaller `--train-studies`, `--validation-studies`, and `--test-studies` values. Such results are exploratory. Full-cohort results still need multi-center validation. Instance-level slice scores are correlated, so study-level metrics are also reported.
- PixelData integrity hashing is outside the metadata-only training run and is evaluated separately by the pixel experiment above.
- SHA-256 comparison assumes the baseline is trusted and protected independently.
- Anomaly scores are not proof of malicious intent.
- This is not a clinical diagnostic or deployment-ready system.
