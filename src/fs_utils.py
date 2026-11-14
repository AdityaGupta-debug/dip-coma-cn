from pathlib import Path
import shutil
import os

def link_or_copy(source, destination):
    """Create a hard link when possible; otherwise fall back to a normal copy."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"
