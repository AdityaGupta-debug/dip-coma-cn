# DICOM Cryptographic Security Experiment

- Held-out DICOM files: 10
- Scenario rows: 130
- Mean signed envelope: 868.5 bytes
- Mean manifest storage overhead: 0.1648% of DICOM file size
- Keys were generated temporarily for this run and deleted at exit; these results do not identify a hospital signer.

## Scenario outcomes

| Scenario | Outcome | Files |
|---|---|---:|
| altered_unsigned_digest | INVALID_SIGNATURE | 10 |
| authorized_deidentify_and_resign | VERIFIED | 10 |
| forged_signature | INVALID_SIGNATURE | 10 |
| forged_unsigned_reference | UNSIGNED_REFERENCE_ACCEPTS | 10 |
| incorrect_hmac_key | INVALID | 10 |
| metadata_change | INTEGRITY_MISMATCH | 10 |
| missing_signature | MISSING_REFERENCE | 10 |
| pixel_byte_flip | INTEGRITY_MISMATCH | 10 |
| replayed_manifest_with_local_cache | POLICY_REVIEW_REQUIRED | 10 |
| reserialization | INTEGRITY_MISMATCH | 10 |
| tiny_single_pixel_change | INTEGRITY_MISMATCH | 10 |
| unchanged_signed_file | VERIFIED | 10 |
| untrusted_signer | UNTRUSTED_SIGNER | 10 |

An unsigned forged reference matches its attacker-modified file by construction; the signed verifier returns `MISSING_REFERENCE` because no signature is present. The single-pixel scenario asserts exactly one changed decoded value for every supported test object.

## Mean operation timings

| Operation | Mean time per file (ms) |
|---|---:|
| manifest create and sign ms | 17.0229 |
| end to end verify ms | 11.3306 |
| raw file sha256 ms | 3.2564 |
| metadata sha256 ms | 4.6831 |
| pixel data sha256 ms | 0.7056 |
| hmac generation ms | 0.0374 |
| hmac verification ms | 0.0141 |
| ed25519 signing ms | 0.1388 |
| ed25519 verification ms | 0.2634 |

![Operation timing chart](security_operation_timings.png)

These values are local prototype timings, not deployment benchmarks. The replay cache is a local stateful demo and must be protected; it is not distributed replay prevention. Pixel editing was tested only for the supported uncompressed single-frame CPAISD objects.
