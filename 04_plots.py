"""STEP 4. Turns the ablation CSV into figures, a table, and picks the best front end.
Usage: python 04_plots.py --dataset gtzan
Reads : results/ablation_<ds>.csv
Writes: results/figs/<ds>_ablation.png (window / N / mel-band sweeps, 95% CI bars)
        results/figs/<ds>_heatmap.png (N x hop grid)
        results/ablation_table_<ds>.md (table for the report)
        results/best_config_<ds>.json  <- INPUT to step 5 (CNN). Chosen by VALIDATION accuracy, not test."""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True, choices=["gtzan", "fma"])
    ds = ap.parse_args().dataset; R = config.RESULTS_DIR
    df = pd.read_csv(f"{R}/ablation_{ds}.csv"); os.makedirs(f"{R}/figs", exist_ok=True)
    def sel(extra=None, **kw):
        m = np.ones(len(df), bool)
        for k, v in kw.items(): m &= (df[k] == v).values
        if extra is not None: m &= extra
        return df[m]
    panels = [("window", sel(N=1024, hop=512, n_mels=40), ["rect", "hann", "hamming", "blackman"]),
              ("N (hop = N/2)", sel(extra=(df.hop == df.N//2).values, window="hann", n_mels=40), None),
              ("n_mels", sel(window="hann", N=1024, hop=512), None)]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (name, sub, order) in zip(axs, panels):
        col = {"window": "window", "N (hop = N/2)": "N", "n_mels": "n_mels"}[name]
        sub = sub.set_index(col).loc[order] if order else sub.sort_values(col).set_index(col)
        x = np.arange(len(sub))
        ax.errorbar(x, sub.acc, yerr=[sub.acc - sub.acc_lo, sub.acc_hi - sub.acc], fmt="o-", capsize=4)
        ax.set_xticks(x); ax.set_xticklabels([str(i) for i in sub.index]); ax.set_xlabel(name); ax.grid(alpha=.3)
    axs[0].set_ylabel("test accuracy (95% bootstrap CI)"); plt.tight_layout()
    plt.savefig(f"{R}/figs/{ds}_ablation.png", dpi=150); plt.close()

    g = sel(window="hann", n_mels=40).copy(); g["ratio"] = (g.N/g.hop).astype(int)
    pv = g.pivot(index="N", columns="ratio", values="acc").sort_index().reindex(columns=[4, 2, 1])
    plt.figure(figsize=(4.5, 3.6)); plt.imshow(pv.values, cmap="viridis"); plt.colorbar(label="accuracy")
    plt.xticks(range(3), ["N/4", "N/2", "N"]); plt.yticks(range(3), pv.index); plt.xlabel("hop"); plt.ylabel("N")
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]): plt.text(j, i, f"{pv.values[i, j]:.3f}", ha="center", va="center", color="w")
    plt.tight_layout(); plt.savefig(f"{R}/figs/{ds}_heatmap.png", dpi=150); plt.close()

    cols = ["window", "N", "hop", "n_mels", "val_acc", "acc", "acc_lo", "acc_hi", "macro_f1", "empty_filters", "infer_ms_per_clip"]
    t = df[cols].sort_values("val_acc", ascending=False).round(3)
    try: txt = t.to_markdown(index=False)            # needs: pip install tabulate
    except ImportError: txt = t.to_string(index=False)
    open(f"{R}/ablation_table_{ds}.md", "w").write(txt)
    best = df.sort_values("val_acc", ascending=False).iloc[0]
    json.dump({**{k: (int(best[k]) if k != "window" else best[k]) for k in ["window", "N", "hop", "n_mels"]},
               "val_acc": float(best.val_acc), "test_acc": float(best.acc)}, open(f"{R}/best_config_{ds}.json", "w"))
    print(t.to_string(index=False)); print("\nbest by validation accuracy ->", f"{R}/best_config_{ds}.json")
    print(f"test-set CI half-width is about +/-{(df.acc_hi - df.acc_lo).mean()/2:.3f}: gaps smaller than that are NOT significant")
