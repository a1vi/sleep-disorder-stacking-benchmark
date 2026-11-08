import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import shutil

HERE = r"c:\Users\alvia\Downloads\Sleep_final_package\reproducibility_package\code"
RESDIR = os.path.join(HERE, "..", "results") + os.sep
FIGSDIR = r"c:\Users\alvia\Downloads\Sleep_final_package\figs" + os.sep

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
BLUE, ORG, GRN, GRY, RED = "#1b6ca8", "#d17a22", "#2e8b57", "#8a8a8a", "#b23a48"

S = pd.read_csv(RESDIR + "summary_all_protocols.csv")
P = pd.read_csv(RESDIR + "paired_vs_stack.csv")
A = json.load(open(RESDIR + "audit.json"))
E = json.load(open(RESDIR + "explain_results.json"))

# ----------------- Fig 1: Protocol Schematic -----------------
fig, ax = plt.subplots(figsize=(8.5, 4.0), dpi=300)
ax.axis("off")
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)

def box(x, y, w, h, t, fc="#eef3f8", ec=BLUE, fs=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=fs, linespacing=1.25)

def arr(x1, y1, x2, y2, c="k"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, lw=1.2, color=c))

W, G = 17.6, 3.0
xs = [1 + i * (W + G) for i in range(5)]
r1 = [
    "374 records\n(literal 'None'\nkept as class)",
    "Group IDs:\nidentical predictor\nvectors share group",
    "Outer split: 5-fold,\nstratified + grouped,\n10 repeats",
    "Outer training\npartition only:\ntuning & fitting",
    "Outer test fold:\nscored once"
]
fcs = ["#fff6e8", "#eef3f8", "#eef3f8", "#eef3f8", "#e9f5ee"]
ecs = [ORG, BLUE, BLUE, BLUE, GRN]
for x, t, fc, ec in zip(xs, r1, fcs, ecs):
    box(x, 34, W, 12, t, fc, ec)
for i in range(4):
    arr(xs[i] + W + 0.3, 40, xs[i + 1] - 0.3, 40)

W2, G2 = 22.5, 2.6
xs2 = [1 + i * (W2 + G2) for i in range(4)]
r2 = [
    "Inner 3-fold grouped CV:\nselect hyper-parameters\nby log-loss",
    "Inner 5-fold grouped OOF\n(scaler/encoder refit per\nfold) gives p_RF & p_XGB",
    "Meta-learner: multinomial\nlogistic regression on\n[encoded X, p_RF, p_XGB]",
    "Refit on full outer-train\ndata; predict the outer\ntest fold"
]
for i, (x, t) in enumerate(zip(xs2, r2)):
    box(x, 14, W2, 12, t, "#e9f5ee" if i == 3 else "#eef3f8", GRN if i == 3 else BLUE)
for i in range(3):
    arr(xs2[i] + W2 + 0.3, 20, xs2[i + 1] - 0.3, 20)
arr(xs[3] + W / 2, 33.6, xs2[0] + W2 / 2, 26.4, c=GRY)
ax.text(40, 30.2, "training partition passed on", fontsize=7.2, color=GRY)
box(1, 1.5, 98, 7.5,
    "Every model (majority class, logistic regression, depth-3 tree, RF, XGBoost, soft voting, stacking, six further learners) sees identical outer splits;\nobservation-level predictions are saved for paired tests",
    "#f4f4f4", GRY, 7.0)
plt.tight_layout()
plt.savefig(FIGSDIR + "fig1_protocol.png", bbox_inches="tight", dpi=300)
plt.close()

# ----------------- Fig 2: Audit & Duplicate Inflation -----------------
fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.5), gridspec_kw=dict(width_ratios=[1, 1.25]), dpi=300)
a = axs[0]
vals = [A["n_rows"], A["unique_rows"], A["unique_predictor_vectors"]]
labs = ["Records", "Unique rows\n(predictors + label)", "Unique predictor\nvectors"]
a.bar(range(3), vals, color=[GRY, BLUE, ORG], width=0.62)
for i, v in enumerate(vals):
    a.text(i, v + 6, str(v), ha="center", fontsize=8.5)
a.set_xticks(range(3))
a.set_xticklabels(labs, fontsize=7.5)
a.set_ylabel("Count")
a.set_ylim(0, 430)
a.set_title("(a) Duplicate structure", fontsize=9.5, loc="left")
ceiling_str = f"{100*A['bayes_ceiling_accuracy_on_374']:.1f}%"
a.text(1.45, 300,
       f"{A['predictor_vectors_with_conflicting_labels']} predictor vectors carry\nconflicting labels → best possible\naccuracy on 374 records: {ceiling_str}",
       fontsize=7.5, ha="left", va="center", color=RED)

b = axs[1]
models = ["Majority", "LogReg", "Stack(RF+XGB)"]
protos = [
    ("naive", "Plain stratified CV\n(duplicates split)", GRY),
    ("grouped", "Grouped CV\n(duplicates kept together)", BLUE),
    ("dedup", "Unique rows only\n(plain stratified CV)", ORG)
]
w = 0.26
for j, (p, lab_txt, c) in enumerate(protos):
    d = S[(S.protocol == p)].set_index("model").loc[models]
    b.bar(np.arange(3) + (j - 1) * w, 100 * d["acc"], w,
          yerr=[100 * (d["acc"] - d["acc_bs_lo"]), 100 * (d["acc_bs_hi"] - d["acc"])],
          color=c, label=lab_txt, capsize=2, error_kw=dict(lw=0.8))
b.set_xticks(range(3))
b.set_xticklabels(["Majority\nclass", "Logistic\nregression", "RF–XGB\nstack"])
b.set_ylim(40, 100)
b.set_ylabel("Accuracy (%); 95% cluster-bootstrap CI")
b.legend(fontsize=6.8, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.25), columnspacing=1.0, handlelength=1.2)
b.set_title("(b) Accuracy depends on how duplicates are handled", fontsize=9.5, loc="left")
plt.tight_layout()
plt.savefig(FIGSDIR + "fig2_audit.png", bbox_inches="tight", dpi=300)
plt.close()

# ----------------- Fig 3: All Models Benchmark -----------------
order = ["Majority", "Tree(d=3)", "LogReg", "SVM(rbf)", "MLP", "RF", "ExtraTrees", "HistGB", "LightGBM", "CatBoost", "XGB", "SoftVote(RF+XGB)", "Stack(RF+XGB)"]
lab_map = {
    "Tree(d=3)": "Decision tree (depth 3)", "LogReg": "Logistic regression", "SVM(rbf)": "SVM (RBF)",
    "MLP": "MLP (32,16)", "RF": "Random forest", "ExtraTrees": "Extra trees", "HistGB": "HistGradientBoosting",
    "LightGBM": "LightGBM", "CatBoost": "CatBoost", "XGB": "XGBoost", "SoftVote(RF+XGB)": "Soft voting (RF+XGB)",
    "Stack(RF+XGB)": "Stacking (RF+XGB)", "Majority": "Majority class"
}
fig, axs = plt.subplots(1, 2, figsize=(8.8, 4.2), sharey=True, dpi=300)
for ax, p, t in zip(axs, ["grouped", "dedup"], ["(a) Grouped CV, 374 records", "(b) Unique rows only, 132 records"]):
    d = S[S.protocol == p].set_index("model").loc[order]
    y = np.arange(len(order))[::-1]
    cols = [RED if m.startswith("Stack") else (GRY if m == "Majority" else BLUE) for m in order]
    ax.errorbar(100 * d["acc"], y, xerr=[100 * (d["acc"] - d["acc_bs_lo"]), 100 * (d["acc_bs_hi"] - d["acc"])], fmt="none", ecolor="#555", lw=1, capsize=2)
    ax.scatter(100 * d["acc"], y, c=cols, s=28, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([lab_map[m] for m in order], fontsize=8)
    ax.set_xlabel("Accuracy (%), mean and 95% CI", fontsize=8.5)
    ax.set_title(t, fontsize=9.5, loc="left")
    ax.axvline(100 * d.loc["Majority", "acc"], color=GRY, ls=":", lw=0.8)
plt.tight_layout()
plt.savefig(FIGSDIR + "fig3_models.png", bbox_inches="tight", dpi=300)
plt.close()

# ----------------- Fig 4: Paired Differences vs Stack -----------------
fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.8), sharey=True, dpi=300)
comps = [m for m in order if m not in ("Stack(RF+XGB)", "Majority")]
for ax, p, t in zip(axs, ["grouped", "dedup"], ["(a) Grouped CV", "(b) Unique rows only"]):
    d = P[P.protocol == p].set_index("comparator").loc[comps]
    y = np.arange(len(comps))[::-1]
    ax.errorbar(d["mean_acc_diff_pp"], y,
                xerr=[d["mean_acc_diff_pp"] - d["boot_lo_pp"], d["boot_hi_pp"] - d["mean_acc_diff_pp"]],
                fmt="o", color=BLUE, ecolor="#555", ms=5, lw=1, capsize=2)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([lab_map[m] for m in comps], fontsize=8)
    ax.set_xlabel("Stack minus comparator, accuracy (pp)\n95% cluster-bootstrap CI", fontsize=8)
    ax.set_title(t, fontsize=9.5, loc="left")
plt.tight_layout()
plt.savefig(FIGSDIR + "fig4_paired.png", bbox_inches="tight", dpi=300)
plt.close()

# ----------------- Fig 5: Calibration Curve -----------------
shutil.copy(RESDIR + "fig_calibration.png", FIGSDIR + "fig5_calibration.png")

# ----------------- Fig 6: SHAP Stability -----------------
runs = pd.read_csv(RESDIR + "shap_stability_runs.csv")
feats = [c for c in runs.columns if c not in ("run", "model")]
fig, axs = plt.subplots(1, 2, figsize=(8.8, 4.0), dpi=300)
for ax, m, t in zip(axs, ["Stack", "SoftVote"], ["(a) Stacking", "(b) Soft voting"]):
    M = runs[runs.model == m][feats]
    mu = M.mean()
    order_ = mu.sort_values().index.tolist()
    y = np.arange(len(order_))
    ax.barh(y, mu[order_], color=BLUE, alpha=0.85)
    ax.hlines(y, M.min()[order_], M.max()[order_], color="k", lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([f.replace("_", " ") for f in order_], fontsize=7.5)
    ax.set_xlabel("Mean |SHAP| (bar); range over 10 runs (line)", fontsize=8)
    ax.set_title(t + f"  — rank rho across runs = {E[m]['mean_pairwise_spearman']:.2f}", fontsize=8.5, loc="left")
plt.tight_layout()
plt.savefig(FIGSDIR + "fig6_shap.png", bbox_inches="tight", dpi=300)
plt.close()

print("All PNG figures successfully generated in figs/!")
