# U-Net adversarial attack comparison

- Test studies: 10
- Positive studies per edit family: 2
- Maximum evaluated images per edit family: 2
- L-infinity budget in normalized image units: 0.02
- Image score: max pre-sigmoid logit; clean validation calibration quantile: 99.0
- Clean test image FPR: 0.0222; clean test study FPR: 0.4000
- FGSM/BIM/PGD L-infinity budget: 0.02; BIM/PGD steps / step size: 10 / 0.004
- C&W-style L2 radius / optimization steps: 2.0 / 20
- DeepFool-style L2 radius / maximum steps: 2.0 / 10
- Attacks target suppression of the U-Net tamper response; attack success is reported only among synthetic edited images detected by the baseline model.
- The L-infinity and L2 attacks have different constraints and should not be interpreted as equal-budget comparisons.
- These attacks test the U-Net only. Any changed DICOM PixelData would still differ from an exact clean reference hash.

## Results
| family               | method   |   positive_images |   image_recall |   baseline_detected_images |   detected_images |   pixel_precision |   pixel_recall |   pixel_iou |   attack_success_rate |   clean_test_file_fpr |   clean_test_study_fpr |
|:---------------------|:---------|------------------:|---------------:|---------------------------:|------------------:|------------------:|---------------:|------------:|----------------------:|----------------------:|-----------------------:|
| localized_patch      | baseline |                 2 |            1   |                          2 |                 2 |         0.420233  |       1        |   0.420233  |                   0   |             0.0222497 |                    0.4 |
| localized_patch      | fgsm     |                 2 |            1   |                          2 |                 2 |         0.0251315 |       0.692901 |   0.0248547 |                   0   |             0.0222497 |                    0.4 |
| localized_patch      | pgd      |                 2 |            0   |                          2 |                 0 |         0.0878754 |       0.365741 |   0.0762548 |                   1   |             0.0222497 |                    0.4 |
| sparse_low_amplitude | baseline |                 2 |            1   |                          2 |                 2 |         0.327753  |       0.963105 |   0.323689  |                   0   |             0.0222497 |                    0.4 |
| sparse_low_amplitude | fgsm     |                 2 |            0   |                          2 |                 0 |         0.0191973 |       0.296695 |   0.0183617 |                   1   |             0.0222497 |                    0.4 |
| sparse_low_amplitude | pgd      |                 2 |            0   |                          2 |                 0 |         0.13553   |       0.471176 |   0.117636  |                   1   |             0.0222497 |                    0.4 |
| copy_move_unseen     | baseline |                 2 |            1   |                          2 |                 2 |         0.472874  |       0.99926  |   0.472708  |                   0   |             0.0222497 |                    0.4 |
| copy_move_unseen     | fgsm     |                 2 |            1   |                          2 |                 2 |         0.0592832 |       0.93713  |   0.0590483 |                   0   |             0.0222497 |                    0.4 |
| copy_move_unseen     | pgd      |                 2 |            0.5 |                          2 |                 1 |         0.0968064 |       0.215237 |   0.0715515 |                   0.5 |             0.0222497 |                    0.4 |

All adversarial samples are in-memory normalized arrays. This is a synthetic robustness test, not a real-world attack demonstration or clinical safety assessment.
