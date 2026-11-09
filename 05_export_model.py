"""STEP 5. Trains the final FMA model with the best front end and saves it for the API.
Usage: python 05_export_model.py
Reads : cache/fma_audio.npy, cache/fma_table.csv, results/ablation_fma.csv,
        results/best_config_fma.json, results/classes_fma.json
Writes: genre_model_fma.joblib  (dict: model, cfg, classes)
Does not modify any existing script. Same recipe as evaluate_svm(): the final model is fit on the
TRAINING split only, with C/gamma taken from the grid search stored in the ablation CSV."""
import ast, json
import joblib
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import genre_dsp as g, config

DS = "fma"
OUT = f"genre_model_{DS}.joblib"

if __name__ == "__main__":
    R = config.RESULTS_DIR
    best = json.load(open(f"{R}/best_config_{DS}.json"))
    classes = json.load(open(f"{R}/classes_{DS}.json"))
    cfg = {k: best[k] for k in g.KEYS}
    print("front end:", cfg)

    # best C / gamma for exactly this front end, from the ablation CSV
    ab = pd.read_csv(f"{R}/ablation_{DS}.csv")
    m = (ab[list(g.KEYS)] == pd.Series(cfg)).all(axis=1)
    bp = ast.literal_eval(ab[m].iloc[0]["best_params"])
    print("SVM params:", bp)

    df, apath = g.load_cached(DS, config.CACHE_DIR)
    X = g.extract_features(apath, df.row.values, cfg)
    y = df["genre"].values
    tr, te = (df.split == "training").values, (df.split == "test").values

    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=bp["svc__C"], gamma=bp["svc__gamma"]))
    model.fit(X[tr], y[tr])
    print(f"test accuracy (sanity check) = {accuracy_score(y[te], model.predict(X[te])):.3f}")

    joblib.dump({"model": model, "cfg": cfg, "classes": classes}, OUT)
    print("saved ->", OUT)
