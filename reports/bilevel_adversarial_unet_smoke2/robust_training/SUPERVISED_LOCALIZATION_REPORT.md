# Supervised DICOM pixel-tamper localization

The unet model is trained to segment synthetic pixel edits. The model receives each image independently; it does not receive a clean counterpart or a hash reference at inference time.
The compound segmentation loss is 0.5050 WBCE + 0.4950 Dice at the selected checkpoint (positive-pixel BCE weight 3). Weighting mode: bilevel.
Training edit families are localized patches and sparse low-amplitude changes. The copy-move family is withheld from training and evaluated as a synthetic out-of-family condition.
The image-level alarm score is the maximum pre-sigmoid logit, avoiding probability saturation. Thresholds are calibrated using clean validation images only. For study-disjoint adaptive weighting, validation studies are split into coefficient-update and checkpoint-selection/calibration groups; the test fold is untouched. Bilevel mode uses a one-inner-step differentiable hypergradient approximation. Adversarial mode balances clean, standard synthetic, and live PGD-evaded samples in each update; PGD inputs are generated with BatchNorm in evaluation mode and are trained against their original tamper masks. The validation-selected checkpoint maximizes mean pixel IoU across unattacked and 10-step PGD validation edits, with clean q99 false-alarm rate as the tie-break. Pixel masks are resized to the model grid before evaluation; they are not native-resolution clinical annotations.

- Train images: 18 (cap=18)
- Validation images: 100
- Test images: 40 across 10 studies
- Device: cpu; epochs completed: 1; input: 256x256
- Architecture: unet; base channels: 8
- Loss-weighting mode: bilevel; selected WBCE/Dice coefficients: 0.5050/0.4950
- Validation studies: 10 total; 5 coefficient-update; 5 checkpoint-selection/calibration
- Best epoch selected by validation macro pixel IoU: 1 (0.0049)
- Synthetic-mask validation images: 16; selection threshold: 0.5
- Clean-test image FPR at calibrated threshold: 0.0000
- Clean-test pixel alarm fraction: 0.004151

## Image-level detection and pixel localization
|   accuracy |   precision |   recall |   f1 |   specificity |   false_positive_rate |   tn |   fp |   fn |   tp | condition            |   n_images |   n_positive_images | loss_weighting   |   bce_weight |   dice_weight |   image_auroc |   image_auprc |   clean_test_image_fpr |   clean_test_pixel_alarm_fraction |   pixel_precision |   pixel_recall |   pixel_f1 |   pixel_iou |   pixel_tp |   pixel_fp |   pixel_fn |   pixel_tn |
|-----------:|------------:|---------:|-----:|--------------:|----------------------:|-----:|-----:|-----:|-----:|:---------------------|-----------:|--------------------:|:-----------------|-------------:|--------------:|--------------:|--------------:|-----------------------:|----------------------------------:|------------------:|---------------:|-----------:|------------:|-----------:|-----------:|-----------:|-----------:|
|      0.75  |           0 |        0 |    0 |             1 |                     0 |   30 |    0 |   10 |    0 | localized_patch      |         40 |                  10 | bilevel          |     0.504988 |      0.495012 |      0.6      |      0.40305  |                      0 |                        0.00415115 |        0.0486922  |     0.0374265  | 0.0423225  |  0.0216187  |        121 |       2364 |       3112 |     649763 |
|      0.775 |           0 |        0 |    0 |             1 |                     0 |   31 |    0 |    9 |    0 | sparse_low_amplitude |         40 |                   9 | bilevel          |     0.504988 |      0.495012 |      0.150538 |      0.148366 |                      0 |                        0.00415115 |        0.00838926 |     0.00170445 | 0.00283326 |  0.00141864 |         10 |       1182 |       5857 |     582775 |
|      0.775 |           0 |        0 |    0 |             1 |                     0 |   31 |    0 |    9 |    0 | copy_move_unseen     |         40 |                   9 | bilevel          |     0.504988 |      0.495012 |      0.379928 |      0.192145 |                      0 |                        0.00415115 |        0.0271845  |     0.00928844 | 0.013846   |  0.00697124 |         56 |       2004 |       5973 |     581791 |

## Best-checkpoint validation-mask metrics
| attack_family        | condition   |   n_images |   pixel_iou |   pixel_dice |   pixel_precision |   pixel_recall |
|:---------------------|:------------|-----------:|------------:|-------------:|------------------:|---------------:|
| localized_patch      | unattacked  |          8 |  0.00509834 |   0.010145   |        0.00509834 |              1 |
| localized_patch      | pgd10       |          8 |  0.00509834 |   0.010145   |        0.00509834 |              1 |
| sparse_low_amplitude | unattacked  |          8 |  0.00473785 |   0.00943103 |        0.00473785 |              1 |
| sparse_low_amplitude | pgd10       |          8 |  0.00473785 |   0.00943103 |        0.00473785 |              1 |

## Study-level detection
| condition            |   n_studies |   accuracy |   precision |   recall |   f1 |   specificity |   false_positive_rate |   tn |   fp |   fn |   tp |
|:---------------------|------------:|-----------:|------------:|---------:|-----:|--------------:|----------------------:|-----:|-----:|-----:|-----:|
| localized_patch      |          10 |        0.8 |           0 |        0 |    0 |             1 |                     0 |    8 |    0 |    2 |    0 |
| sparse_low_amplitude |          10 |        0.8 |           0 |        0 |    0 |             1 |                     0 |    8 |    0 |    2 |    0 |
| copy_move_unseen     |          10 |        0.8 |           0 |        0 |    0 |             1 |                     0 |    8 |    0 |    2 |    0 |

## Interpretation limits
- A withheld copy-move condition tests transfer to one unseen synthetic family, not generalization to natural attacks.
- Scores and masks are evaluated on the held-out CPAISD test fold; the ten-study sample limits population-level inference.
- Localization metrics use generated masks and the resized image grid.
- This experiment does not establish clinical impact, real-world attack prevalence, or deployment readiness.