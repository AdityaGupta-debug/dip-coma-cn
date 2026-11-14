import argparse
from pathlib import Path
from .integrity import create_baseline
from .anomaly import train_isolation_forest
from .deep_anomaly import train_autoencoder
from .attacks import create_attack, ATTACKS
from .detection import detect_directory
from .evaluation import evaluate_attack, run_ablation

def main():
    parser = argparse.ArgumentParser(description="Brain CT DICOM metadata security research pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("baseline"); p.add_argument("--input", required=True); p.add_argument("--manifest", default="reports/baseline.json"); p.add_argument("--include-pixel-hash", action="store_true")
    p = sub.add_parser("train"); p.add_argument("--input", required=True); p.add_argument("--validation", default=None); p.add_argument("--model", default="models/isolation_forest.joblib"); p.add_argument("--threshold-percentile", type=float, default=1.0); p.add_argument("--max-files", type=int, default=30000); p.add_argument("--max-validation-files", type=int, default=10000)
    p = sub.add_parser("train-ae"); p.add_argument("--input", required=True); p.add_argument("--validation", default=None); p.add_argument("--model", default="models/metadata_autoencoder.joblib"); p.add_argument("--epochs", type=int, default=100); p.add_argument("--batch-size", type=int, default=32); p.add_argument("--threshold-percentile", type=float, default=99.0); p.add_argument("--max-train-files", type=int, default=30000); p.add_argument("--max-validation-files", type=int, default=10000)
    p = sub.add_parser("attack"); p.add_argument("--input", required=True); p.add_argument("--output", required=True); p.add_argument("--attack", choices=ATTACKS, required=True); p.add_argument("--fraction", type=float, default=0.20); p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("detect"); p.add_argument("--input", required=True); p.add_argument("--manifest", default="reports/baseline.json"); p.add_argument("--model", default="models/isolation_forest.joblib"); p.add_argument("--ae-model", default="models/metadata_autoencoder.joblib"); p.add_argument("--output", default="reports/detection.csv")
    p = sub.add_parser("evaluate"); p.add_argument("--input", required=True); p.add_argument("--attack-log", required=True); p.add_argument("--manifest", default="reports/baseline.json"); p.add_argument("--model", default="models/isolation_forest.joblib"); p.add_argument("--ae-model", default="models/metadata_autoencoder.joblib"); p.add_argument("--output-dir", required=True)
    p = sub.add_parser("ablation"); p.add_argument("--attacks-root", default="data/attacked"); p.add_argument("--attack-names", nargs="+", default=list(ATTACKS)); p.add_argument("--manifest", default="reports/baseline.json"); p.add_argument("--model", default="models/isolation_forest.joblib"); p.add_argument("--ae-model", default="models/metadata_autoencoder.joblib"); p.add_argument("--output-dir", default="reports/ablation")
    args = parser.parse_args()
    if args.command == "baseline":
        result = create_baseline(args.input, args.manifest, include_pixel_hash=args.include_pixel_hash); print(f"Trusted baseline created: {result['file_count']} DICOM files -> {args.manifest}")
    elif args.command == "train":
        result = train_isolation_forest(args.input, args.model, max_files=args.max_files, validation_dir=args.validation, threshold_percentile=args.threshold_percentile, max_validation_files=args.max_validation_files); print(f"Isolation Forest trained on {result['training_files']} files; threshold={result['threshold']:.6f} ({result['threshold_source']}) -> {args.model}")
    elif args.command == "train-ae":
        result = train_autoencoder(args.input, args.model, validation_dir=args.validation, epochs=args.epochs, batch_size=args.batch_size, threshold_percentile=args.threshold_percentile, max_train_files=args.max_train_files, max_validation_files=args.max_validation_files)
        print(f"Autoencoder trained on {result['training_files']} files; threshold={result['threshold']:.6f}; source={result['threshold_source']} -> {args.model}")
    elif args.command == "attack":
        result = create_attack(args.input, args.output, args.attack, args.fraction, args.seed); print(f"Created {args.attack}: {result['modified_file_count']} modified files -> {args.output}")
    elif args.command == "detect":
        frame = detect_directory(args.input, args.manifest, args.model, args.output, args.ae_model); print(frame.to_string(index=False)); print(f"Saved: {args.output}")
    elif args.command == "evaluate":
        summary, _ = evaluate_attack(args.input, args.manifest, args.model, args.attack_log, args.output_dir, args.ae_model); print(summary.to_string(index=False))
    elif args.command == "ablation":
        dirs = {name: str(Path(args.attacks_root) / name) for name in args.attack_names}
        summary = run_ablation(dirs, args.manifest, args.model, args.output_dir, args.ae_model); print(summary.to_string(index=False)); print(f"Saved: {Path(args.output_dir) / 'ablation_summary.csv'}")

if __name__ == "__main__":
    main()
