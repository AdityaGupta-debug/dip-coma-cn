# DICOM Cryptographic Security Experiment

- Input files: 10
- Scenario evaluations: 130
- Mean manifest creation/signing time: 10.265 ms
- Mean end-to-end verification time: 7.865 ms
- Mean signed manifest envelope: 868.5 bytes
- Mean manifest-to-DICOM storage overhead: 0.1648%

Per-operation hash, HMAC, Ed25519 and end-to-end timing is in `security_performance.csv`; scenario outcomes are in `security_scenarios.csv`. These local measurements are not deployment benchmarks.
