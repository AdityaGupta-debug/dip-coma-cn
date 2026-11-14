"""Create a tiny synthetic CT-like DICOM dataset for smoke testing only."""
from pathlib import Path
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage, generate_uid
from pydicom import dcmwrite
from datetime import datetime
import argparse

def create_dataset(path, study_uid, series_uid, instance, patient_id):
    path.parent.mkdir(parents=True, exist_ok=True)
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.Modality = "CT"
    ds.PatientID = patient_id
    ds.PatientName = "SYNTHETIC^PHANTOM"
    ds.StudyDescription = "Synthetic CT Phantom"
    ds.SeriesDescription = "Axial brain phantom"
    ds.ProtocolName = "DEMO_BRAIN"
    ds.BodyPartExamined = "HEAD"
    ds.Manufacturer = "SYNTHETIC"
    ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
    ds.Rows = 64
    ds.Columns = 64
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.PixelSpacing = [0.5, 0.5]
    ds.SliceThickness = 1.0
    ds.InstanceNumber = instance
    ds.ImagePositionPatient = [0.0, 0.0, float(instance)]
    ds.RescaleIntercept = -1024
    ds.RescaleSlope = 1
    ds.KVP = 120
    ds.XRayTubeCurrent = 200
    ds.StudyDate = datetime.now().strftime("%Y%m%d")
    ds.StudyTime = datetime.now().strftime("%H%M%S")
    rng = np.random.default_rng(instance)
    y, x = np.ogrid[:64, :64]
    phantom = ((x - 32) ** 2 + (y - 32) ** 2 < 24 ** 2).astype(np.int16) * 700 - 1000
    phantom += rng.normal(0, 15, (64, 64)).astype(np.int16)
    ds.PixelData = phantom.astype("<i2").tobytes()
    dcmwrite(str(path), ds, enforce_file_format=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/demo_ct")
    parser.add_argument("--series", type=int, default=3)
    parser.add_argument("--slices", type=int, default=12)
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists():
        import shutil
        shutil.rmtree(root)
    study = generate_uid()
    for s in range(args.series):
        series_uid = generate_uid()
        for i in range(args.slices):
            create_dataset(root / f"series_{s+1}" / f"slice_{i+1:03d}.dcm",
                           study, series_uid, i + 1, f"PHANTOM{s+1:02d}")
    print(f"Generated {args.series * args.slices} SYNTHETIC DICOM files in {root}")
    print("Synthetic data is only for software testing, not clinical or research-performance claims.")

if __name__ == "__main__":
    main()
