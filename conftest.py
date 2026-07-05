"""Konfigurasi pytest: atur path import dan database sementara untuk test."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))          # untuk `import api.server`
sys.path.insert(0, str(ROOT / "app"))  # untuk `import model_utils`, `geo_utils`, `db`

# Pakai database sementara agar test tidak menyentuh data asli.
os.environ.setdefault("POTHOLE_DB", str(Path(tempfile.gettempdir()) / "pothole_test.db"))
