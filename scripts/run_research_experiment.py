"""Run the lightweight CPAISD metadata-security experiment."""
from pathlib import Path
import argparse, json, subprocess, sys, time, os
import pandas as pd
import matplotlib.pyplot as plt

ATTACKS = ["patient_id", "study_description", "protocol_name_same_length",
           "manufacturer_same_length", "pixel_spacing",
           "study_uid_inconsistency", "remove_protocol", "modality"]

def run(cmd, root):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([str(x) for x in cmd], cwd=root, check=True, env=env)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/real_ct")
    ap.add_argument("--workdir", default="experiment_cpaisd")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--attack-fraction", type=float, default=0.20)
    ap.add_argument("--max-train-files", type=int, default=0, help="Maximum sampled files per fold; 0 uses all files.")
    ap.add_argument("--max-validation-files", type=int, default=0, help="Maximum sampled validation files; 0 uses all files.")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    data, work = (root/a.input).resolve(), (root/a.workdir).resolve()
    if work.exists() and any(work.iterdir()):
        raise FileExistsError(f"Experiment work directory is not empty; choose a fresh --workdir: {work}")
    reports, models, attacks = work/"reports", work/"models", work/"attacks"
    for d in (reports, models, attacks): d.mkdir(parents=True, exist_ok=True)
    folds = {k:data/k for k in ("train","val","test")}
    for name, folder in folds.items():
        if not folder.exists() or not any(x.is_file() for x in folder.rglob("*")):
            raise FileNotFoundError(f"No files under {folder}; run scripts/select_cpaisd_subset.py first.")
    subset_file = data/"subset_summary.json"
    subset = json.loads(subset_file.read_text()) if subset_file.exists() else {}
    start = time.perf_counter()
    for name, folder in folds.items():
        run([sys.executable,"scripts/check_dataset.py","--input",folder],root)
    run([sys.executable,"-m","src.cli","train","--input",folds["train"],"--validation",folds["val"],"--model",models/"isolation_forest.joblib","--max-files",a.max_train_files,"--max-validation-files",a.max_validation_files],root)
    run([sys.executable,"-m","src.cli","train-ae","--input",folds["train"],"--validation",folds["val"],"--model",models/"metadata_autoencoder.joblib","--epochs",a.epochs,"--max-train-files",a.max_train_files,"--max-validation-files",a.max_validation_files],root)
    baseline = reports/"test_baseline.json"
    run([sys.executable,"-m","src.cli","baseline","--input",folds["test"],"--manifest",baseline],root)
    clean_csv = reports/"clean_test_predictions.csv"
    run([sys.executable,"-m","src.cli","detect","--input",folds["test"],"--manifest",baseline,"--model",models/"isolation_forest.joblib","--ae-model",models/"metadata_autoencoder.joblib","--output",clean_csv],root)
    for attack in ATTACKS:
        run([sys.executable,"-m","src.cli","attack","--input",folds["test"],"--output",attacks/attack,"--attack",attack,"--fraction",a.attack_fraction,"--seed",a.seed],root)
    ablation = reports/"ablation"
    run([sys.executable,"-m","src.cli","ablation","--attacks-root",attacks,"--attack-names",*ATTACKS,"--manifest",baseline,"--model",models/"isolation_forest.joblib","--ae-model",models/"metadata_autoencoder.joblib","--output-dir",ablation],root)
    frames=[]
    for attack in ATTACKS:
        df=pd.read_csv(ablation/attack/"metrics.csv")
        df.insert(0,"attack",attack)
        frames.append(df)
    per=pd.concat(frames,ignore_index=True)
    per.to_csv(reports/"per_attack_metrics.csv",index=False)
    study_frames=[]
    for attack in ATTACKS:
        df=pd.read_csv(ablation/attack/"study_level_metrics.csv")
        df.insert(0,"attack",attack)
        study_frames.append(df)
    per_study=pd.concat(study_frames,ignore_index=True)
    per_study.to_csv(reports/"per_attack_study_metrics.csv",index=False)
    clean=pd.read_csv(clean_csv)
    cols=[("Metadata SHA-256","sha256_flag"),("Structural rules","rule_flag"),("Isolation Forest","isolation_forest_flag"),("Autoencoder","autoencoder_flag"),("IF + AE","if_ae_flag"),("Rules + IF","rules_if_flag"),("Hybrid","hybrid_or_flag")]
    fpr=pd.DataFrame([{"method":n,"clean_test_false_positive_rate":float(clean[c].mean())} for n,c in cols])
    fpr.to_csv(reports/"clean_test_false_positive_rates.csv",index=False)
    clean_study=clean.groupby("study_group",as_index=False).agg(**{c:(c,"max") for _,c in cols})
    study_fpr=pd.DataFrame([{"method":n,"clean_test_study_false_positive_rate":float(clean_study[c].mean()),"clean_test_studies":int(len(clean_study))} for n,c in cols])
    study_fpr.to_csv(reports/"clean_test_study_false_positive_rates.csv",index=False)
    agg=per.groupby("method")[["accuracy","precision","recall","f1","false_positive_rate","specificity"]].mean().reset_index()
    agg.to_csv(reports/"macro_average_metrics.csv",index=False)
    study_agg=per_study.groupby("method")[["accuracy","precision","recall","f1","false_positive_rate","specificity"]].mean().reset_index()
    study_agg.to_csv(reports/"macro_average_study_metrics.csv",index=False)
    per.pivot(index="attack",columns="method",values="f1").plot(kind="bar",figsize=(13,6))
    plt.ylabel("F1-score"); plt.xlabel("Attack"); plt.ylim(0,1.05); plt.tight_layout()
    plt.savefig(reports/"f1_by_attack.png",dpi=180); plt.close()
    fpr.set_index("method").plot(kind="bar",legend=False,figsize=(10,5))
    plt.ylabel("False-positive rate"); plt.tight_layout()
    plt.savefig(reports/"clean_test_fpr.png",dpi=180); plt.close()
    config={"dataset":"CPAISD","subset":subset,"seed":a.seed,"epochs":a.epochs,"attack_fraction":a.attack_fraction,"max_train_files":a.max_train_files,"max_validation_files":a.max_validation_files,"isolation_forest_threshold_percentile":1.0,"autoencoder_threshold_percentile":99.0,"pixel_data_hashing":False,"elapsed_seconds":round(time.perf_counter()-start,2),"notes":["Official CPAISD folds preserved; complete studies selected.","Training uses clean training studies only; anomaly thresholds use clean validation studies only.","Training, baseline, and detection read metadata headers only.","PixelData is not decoded or hashed in this lightweight run.","CPAISD is a single-center hyperacute ischemic stroke cohort; wider generalization is untested."]}
    (reports/"experiment_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    report=["# CPAISD metadata-security experiment","","Metrics are generated from the local run; none are fabricated.","","## Macro-average instance-level metrics",agg.to_markdown(index=False),"","## Macro-average study-level metrics",study_agg.to_markdown(index=False),"","## Clean test file false-positive rates",fpr.to_markdown(index=False),"","## Clean test study false-positive rates",study_fpr.to_markdown(index=False),"","## Per-attack instance-level metrics",per.to_markdown(index=False),"","## Per-attack study-level metrics",per_study.to_markdown(index=False),"","## Interpretation notes","- The metadata hash baseline is created from each clean held-out test file, so its clean false-positive rate is zero by construction; the hash result measures exact-change detection against that trusted reference.","- The complete OR hybrid necessarily inherits this hash detector's sensitivity when the trusted baseline is available. Interpret anomaly-model comparisons separately.","","## Limitations","- CPAISD is a single-center hyperacute ischemic stroke cohort.","- The cohort is exploratory and not clinical validation.","- Attack sampling is study-level; instance-level results remain correlated within studies.","- Anomaly flags are not proof of malicious intent.","- PixelData integrity checking is disabled in this run.","- SHA-256 results assume the reference manifest is trusted and protected independently."]
    (reports/"RESEARCH_REPORT.md").write_text("\n".join(report),encoding="utf-8")
    print(f"\nEXPERIMENT COMPLETE: {reports/'RESEARCH_REPORT.md'}")

if __name__=="__main__": main()
