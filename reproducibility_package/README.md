# NEURI-D-26-00299 revision: code, results, manuscript
- `code/` analysis scripts (`common.py` loads data with `keep_default_na=False`, asserts the three literal labels, checks SHA-256).
- `results/` split ids (`split_ids_*.csv`: repeat, fold, row), observation-level predictions (`oof_predictions_*.csv`, `nested_oof_predictions_*.csv`, `original_split_observation_level.csv`; columns y/pred/p_insomnia/p_none/p_apnea; class codes 0=Insomnia,1=None,2=Sleep Apnea), fold metrics, summaries, SHAP/LIME outputs.
- `paper/` LaTeX source, generated tables (`tab_*.tex`) and number macros (`numbers.tex`), figures, response letter.
- `row` indexes the cleaned data frame in file order (Person ID = row+1); for the unique-rows protocol it indexes the deduplicated frame.
Run `./run_all.sh` after placing the dataset in `data/sleep.csv`.
