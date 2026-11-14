import tempfile
import unittest
from pathlib import Path
from src.integrity import create_baseline, load_baseline
from src.anomaly import train_isolation_forest
from src.attacks import create_attack
from src.detection import detect_directory
from scripts.make_demo_dataset import main as unused

class PipelineSmokeTest(unittest.TestCase):
    def test_baseline_training_attack_detection(self):
        from scripts.make_demo_dataset import create_dataset
        from pydicom.uid import generate_uid
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean = root / "clean"
            study = generate_uid()
            for s in range(2):
                series = generate_uid()
                for i in range(6):
                    create_dataset(clean / f"s{s}" / f"{i}.dcm", study, series, i+1, f"P{s}")
            manifest = root / "baseline.json"
            model = root / "if.joblib"
            create_baseline(clean, manifest)
            train_isolation_forest(clean, model, contamination=0.1)
            attacked = root / "attacked"
            create_attack(clean, attacked, "patient_id", 1.0)
            result = detect_directory(attacked, manifest, model)
            self.assertEqual(len(result), 12)
            self.assertTrue((result["sha256_flag"] == 1).all())

if __name__ == "__main__":
    unittest.main()
