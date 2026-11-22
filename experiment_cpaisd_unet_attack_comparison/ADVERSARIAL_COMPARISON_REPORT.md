# U-Net adversarial attack comparison

- Test studies: 10
- Positive studies per edit family: 2
- Maximum evaluated images per edit family: 64
- L-infinity budget in normalized image units: 0.02
- FGSM/BIM/PGD L-infinity budget: 0.02; BIM/PGD steps / step size: 10 / 0.004
- C&W-style L2 radius / optimization steps: 2.0 / 20
- DeepFool-style L2 radius / maximum steps: 2.0 / 10
- Attacks target suppression of the U-Net tamper response; attack success is reported only among synthetic edited images detected by the baseline model.
- The L-infinity and L2 attacks have different constraints and should not be interpreted as equal-budget comparisons.
- These attacks test the U-Net only. Any changed DICOM PixelData would still differ from an exact clean reference hash.

## Results
| family               | method      |   positive_images |   image_recall |   baseline_detected_images |   detected_images |   pixel_precision |   pixel_recall |   pixel_iou |   attack_success_rate |
|:---------------------|:------------|------------------:|---------------:|---------------------------:|------------------:|------------------:|---------------:|------------:|----------------------:|
| localized_patch      | baseline    |                64 |       0.984375 |                         63 |                63 |         0.390531  |       0.99676  |   0.390036  |              0        |
| localized_patch      | fgsm        |                64 |       0.84375  |                         63 |                54 |         0.0318425 |       0.87413  |   0.0316971 |              0.142857 |
| localized_patch      | bim         |                64 |       0.09375  |                         63 |                 6 |         0.0932688 |       0.388346 |   0.0813225 |              0.904762 |
| localized_patch      | pgd         |                64 |       0.046875 |                         63 |                 3 |         0.0810732 |       0.361944 |   0.0709351 |              0.952381 |
| localized_patch      | cw_l2       |                64 |       0        |                         63 |                 0 |         0.0374717 |       0.295358 |   0.0343967 |              1        |
| localized_patch      | deepfool_l2 |                64 |       0.671875 |                         63 |                43 |         0.242394  |       0.941925 |   0.238824  |              0.31746  |
| sparse_low_amplitude | baseline    |                64 |       1        |                         64 |                64 |         0.337386  |       0.964015 |   0.33319   |              0        |
| sparse_low_amplitude | fgsm        |                64 |       0.234375 |                         64 |                15 |         0.0197464 |       0.304518 |   0.0188943 |              0.765625 |
| sparse_low_amplitude | bim         |                64 |       0        |                         64 |                 0 |         0.132766  |       0.36859  |   0.108166  |              1        |
| sparse_low_amplitude | pgd         |                64 |       0        |                         64 |                 0 |         0.13477   |       0.473929 |   0.117232  |              1        |
| sparse_low_amplitude | cw_l2       |                64 |       1        |                         64 |                64 |         0.16976   |       0.952252 |   0.168327  |              0        |
| sparse_low_amplitude | deepfool_l2 |                64 |       1        |                         64 |                64 |         0.336713  |       0.960798 |   0.332149  |              0        |
| copy_move_unseen     | baseline    |                64 |       0.890625 |                         57 |                57 |         0.41927   |       0.788837 |   0.376962  |              0        |
| copy_move_unseen     | fgsm        |                64 |       0.59375  |                         57 |                38 |         0.0276958 |       0.430296 |   0.0267161 |              0.350877 |
| copy_move_unseen     | bim         |                64 |       0.09375  |                         57 |                 6 |         0.0753624 |       0.135399 |   0.0508782 |              0.894737 |
| copy_move_unseen     | pgd         |                64 |       0.140625 |                         57 |                 9 |         0.0728839 |       0.142385 |   0.0506492 |              0.842105 |
| copy_move_unseen     | cw_l2       |                64 |       0.046875 |                         57 |                 3 |         0.0660163 |       0.232542 |   0.0542062 |              0.947368 |
| copy_move_unseen     | deepfool_l2 |                64 |       0.390625 |                         57 |                25 |         0.323416  |       0.600084 |   0.266069  |              0.561404 |

All adversarial samples are in-memory normalized arrays. This is a synthetic robustness test, not a real-world attack demonstration or clinical safety assessment.
