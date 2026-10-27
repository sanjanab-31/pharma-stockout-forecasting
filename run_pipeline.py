"""run_pipeline.py - runs the complete project end to end.

    python run_pipeline.py
Steps: generate data -> pre-process -> features -> models -> simulation -> statistics -> ingestion demo.
Results are written to outputs/ and charts to figures/.
"""
import runpy, sys, time
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))
STEPS = ["gen_data", "preprocess", "features", "models", "simulate", "stats_analysis", "ingest"]

if __name__ == "__main__":
    t0 = time.time()
    for step in STEPS:
        print(f"\n=== {step}.py ===")
        sys.argv = [f"{step}.py"]
        runpy.run_path(str(SRC / f"{step}.py"), run_name="__main__")
    print(f"\nPipeline finished in {(time.time() - t0) / 60:.1f} min. See outputs/ and figures/.")
