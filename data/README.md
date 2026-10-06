# Data acquisition checklist

## Pks13 activity

1. Download/parse Aggarwal 2017 supplementary tables.
2. Download/parse Krieger 2024 supplementary information.
3. Query BindingDB and ChEMBL for Pks13/Pks13-TE entries.
4. Deduplicate by canonical isomeric SMILES plus assay context.
5. Keep only the chosen enzymatic endpoint for the main regression.

## Structures

RCSB entries to obtain and verify:

- 5V3Y
- 5V40
- 5V41
- 5V42
- 8TQG
- 8TQV
- 8TR4
- 8TRY (verification gate)

Example RCSB download endpoint pattern:

`https://files.rcsb.org/download/5V3Y.cif`

## Candidate pool

Start at a few 10^4 drug-like compounds from ZINC20/PubChem. Keep the pool small enough that B3-B5 can be profiled end-to-end before scaling.

## Decoys

Create Pks13-specific property-matched decoys from ZINC or use a DUD-E-style construction as the methodological reference.
