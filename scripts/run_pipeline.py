#!/usr/bin/env python
"""X-TUBIT Execution Runner Script.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --raw-data data/raw/pks13_compounds.csv --output data/processed
"""

import argparse
import sys
from pathlib import Path

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xtubit.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Execute the X-TUBIT End-to-End Computational Pipeline")
    parser.add_argument(
        "--raw-data",
        type=str,
        default="data/raw/pks13_compounds.csv",
        help="Path to raw compound CSV containing SMILES and bioassay labels."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Directory to save processed artifacts, Parquet files, and metrics."
    )
    args = parser.parse_args()

    print("================================================================================")
    print("                    X-TUBIT COMPUTATIONAL PIPELINE EXECUTION                   ")
    print("   Digital Annealing and Bayesian Screening for Mycobacterium tuberculosis      ")
    print("================================================================================")
    print(f"[*] Input Data  : {args.raw_data}")
    print(f"[*] Output Dir  : {args.output}")
    print("--------------------------------------------------------------------------------")

    results = run_pipeline(raw_csv=args.raw_data, out_dir=args.output)

    print("--------------------------------------------------------------------------------")
    print("Pipeline finished successfully!")
    print(f"[*] Status               : {results['status']}")
    print(f"[*] Target Reference PDB : {results['reference_pdb']}")
    print(f"[*] Lead Molecule        : {results['lead_compound']}")
    print(f"[*] Heavy-atom RMSD      : {results['heavy_atom_rmsd_A']} A")
    print(f"[*] RMSD < 2.0 A Pass    : {results['rmsd_under_2A_success']}")
    print(f"[*] Total Wall Time      : {results['pipeline_wall_time_s']:.2f} s")
    print("================================================================================")


if __name__ == "__main__":
    main()
