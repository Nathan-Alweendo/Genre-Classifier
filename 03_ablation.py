"""STEP 3. The DSP ablation: 15 front-end configs, classifier (RBF-SVM) held fixed.
Usage: python 03_ablation.py --dataset gtzan   [--limit 20 for a quick test run]
Reads : cache/<ds>_audio.npy, cache/<ds>_table.csv
Writes: results/ablation_<ds>.csv (one row per config: val_acc, acc, 95% CI, macro-F1, timings, confusion matrix)
        results/classes_<ds>.json
Resumable: re-running skips configs already in the CSV."""
import argparse, json, os, time
import pandas as pd
import genre_dsp as g, config

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True, choices=["gtzan", "fma"])
    ap.add_argument("--limit", type=int, default=None, help="clips per genre (quick test)")
    a = ap.parse_args(); ds = a.dataset
    df, apath = g.load_cached(ds, config.CACHE_DIR, a.limit)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    classes = sorted(df.genre.unique())
    json.dump(classes, open(f"{config.RESULTS_DIR}/classes_{ds}.json", "w"))
    out = f"{config.RESULTS_DIR}/ablation_{ds}.csv"
    rows = pd.read_csv(out).to_dict("records") if os.path.exists(out) else []
    done = {tuple(r[k] for k in g.KEYS) for r in rows}
    cfgs = g.configs()
    print(f"{len(df)} clips, {len(cfgs)} configs, {len(done)} already done")
    for i, cfg in enumerate(cfgs, 1):
        if tuple(cfg[k] for k in g.KEYS) in done: continue
        t0 = time.time(); X = g.extract_features(apath, df.row.values, cfg); wall = time.time() - t0
        lat = g.measure_ms(g._feat_one, df.row.values, apath, cfg)
        r = g.evaluate_svm(df, X, cfg, classes)
        r.update(feat_wall_s=wall, feat_ms_single=lat, infer_ms_per_clip=lat + r["predict_ms"], cm=json.dumps(r["cm"]))
        rows.append(r); pd.DataFrame(rows).to_csv(out, index=False)          # saved after every config
        print(f"[{i}/{len(cfgs)}] {cfg['window']:8s} N={cfg['N']:4d} hop={cfg['hop']:4d} mels={cfg['n_mels']:3d} "
              f"val={r['val_acc']:.3f} test={r['acc']:.3f} [{r['acc_lo']:.3f},{r['acc_hi']:.3f}] "
              f"F1={r['macro_f1']:.3f} ({time.time()-t0:.0f}s)")
    print("done ->", out)
