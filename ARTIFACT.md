# FDP-Rocks Artifact Source

This branch is the source snapshot used by the ATC 2026 artifact.

- Base FDP-Rocks revision: `33016b94e4cfb73b5a56126b47c203a25c70d46a`
- Evaluated mode: p25 thresholds with `ROCKSDB_ML_PREDICT=0`
- Runtime ML models: not required by the evaluated mode
- RocksDB plugins: disabled during the artifact build

The branch preserves executable behavior while removing development-only
scripts, generated Python bytecode, non-English design notes, and redundant
comments. User-visible diagnostic messages are in English.

Build instructions and the benchmark driver are maintained in:

`https://github.com/strivesnail/FDP-rocks`
