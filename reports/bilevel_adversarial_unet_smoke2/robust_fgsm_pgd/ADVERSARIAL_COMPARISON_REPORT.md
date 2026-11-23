# U-Net adversarial attack comparison

- Test studies: 10
- Positive studies per edit family: 2
- Maximum evaluated images per edit family: 2
- L-infinity budget in normalized image units: 0.02
- Image score: max pre-sigmoid logit; clean validation calibration quantile: 99.0
- Clean test image FPR: 0.0074; clean test study FPR: 0.3000
- FGSM/BIM/PGD L-infinity budget: 0.02; BIM/PGD steps / step size: 10 / 0.004
- C&W-style L2 radius / optimization steps: 2.0 / 20
- DeepFool-style L2 radius / maximum steps: 2.0 / 10
- Attacks target suppression of the U-Net tamper response; attack success is reported only among synthetic edited images detected by the baseline model.
- The L-infinity and L2 attacks have different constraints and should not be interpreted as equal-budget comparisons.
- These attacks test the U-Net only. Any changed DICOM PixelData would still differ from an exact clean reference hash.

## Results
| family               | method   |   positive_images |   image_recall |   baseline_detected_images |   detected_images |   pixel_precision |   pixel_recall |   pixel_iou |   attack_success_rate |   clean_test_file_fpr |   clean_test_study_fpr |
|:---------------------|:---------|------------------:|---------------:|---------------------------:|------------------:|------------------:|---------------:|------------:|----------------------:|----------------------:|-----------------------:|
| localized_patch      | baseline |                 2 |              0 |                          0 |                 0 |         0.0204461 |     0.0169753  |  0.0093617  |                     0 |            0.00741656 |                    0.3 |
| localized_patch      | fgsm     |                 2 |              0 |                          0 |                 0 |         0.0218069 |     0.0108025  |  0.00727651 |                     0 |            0.00741656 |                    0.3 |
| localized_patch      | pgd      |                 2 |              0 |                          0 |                 0 |         0.0249221 |     0.0123457  |  0.00832466 |                     0 |            0.00741656 |                    0.3 |
| sparse_low_amplitude | baseline |                 2 |              0 |                          0 |                 0 |         0         |     0          |  0          |                     0 |            0.00741656 |                    0.3 |
| sparse_low_amplitude | fgsm     |                 2 |              0 |                          0 |                 0 |         0         |     0          |  0          |                     0 |            0.00741656 |                    0.3 |
| sparse_low_amplitude | pgd      |                 2 |              0 |                          0 |                 0 |         0         |     0          |  0          |                     0 |            0.00741656 |                    0.3 |
| copy_move_unseen     | baseline |                 2 |              0 |                          0 |                 0 |         0.0188324 |     0.00739645 |  0.00533903 |                     0 |            0.00741656 |                    0.3 |
| copy_move_unseen     | fgsm     |                 2 |              0 |                          0 |                 0 |         0.0190476 |     0.00443787 |  0.00361228 |                     0 |            0.00741656 |                    0.3 |
| copy_move_unseen     | pgd      |                 2 |              0 |                          0 |                 0 |         0.0190476 |     0.00443787 |  0.00361228 |                     0 |            0.00741656 |                    0.3 |

All adversarial samples are in-memory normalized arrays. This is a synthetic robustness test, not a real-world attack demonstration or clinical safety assessment.
