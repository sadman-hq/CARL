"""Project directory layout. Every script works relative to a project root (default: the repository root)."""
from pathlib import Path


class Paths:
    def __init__(self, root="."):
        self.root = Path(root).resolve()
        self.data = self.root / "data"
        self.carbon = self.data / "carbon"
        self.runs = self.root / "runs"
        self.live = self.runs / "live"
        self.des = self.runs / "des"
        self.analysis = self.root / "analysis"
        self.plots = self.analysis / "plots"
        self.tables = self.analysis / "tables"
        self.logs = self.root / "logs"

    def ensure(self):
        for p in [self.carbon, self.live, self.des, self.plots, self.tables, self.logs]:
            p.mkdir(parents=True, exist_ok=True)
        return self
