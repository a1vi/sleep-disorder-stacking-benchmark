#!/bin/sh
# Full pipeline (about 1 h on 2 cores). Results land in ../results, tables/macros in ../paper.
set -e; cd "$(dirname "$0")/code"
python3 audit.py
for p in grouped dedup naive; do python3 run_cv.py $p 10; done
python3 orig_holdout.py; python3 train_gap.py
python3 nested_cv.py dedup 5; python3 nested_cv.py grouped 5; python3 nested_stats.py
python3 feat_ablation.py; python3 explain.py; python3 stats_calib.py
python3 build_tables.py; python3 make_figures.py
cd ../paper && latexmk -pdf main.tex && latexmk -pdf response.tex
