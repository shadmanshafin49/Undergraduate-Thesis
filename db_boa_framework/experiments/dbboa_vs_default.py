"""
experiments/dbboa_vs_default.py
===============================
Reproduces and PERSISTS the DB-BOA-tuned vs hand-set-default ADTCN comparison
that main.py computes at runtime but never saved to JSON (Chapter 6, Table 6.4).

Trains both configurations on the exact same data path as main.py
(loader.load() -> X_full = train+val ; evaluate on the held-out X_test),
with the deterministic seed (random_state=42, torch.manual_seed in ADTCN.fit),
so the tuned row should reproduce db_boa_results.json (MCC 0.677) as a
faithfulness check, and the default row gives the previously-unbacked number.

Run:  python3 experiments/dbboa_vs_default.py
"""
import sys, os, json, datetime, time
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))
import numpy as np
import torch
torch.set_num_threads(os.cpu_count() or 4)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config        import ADTCN_CONFIG, DATA_CONFIG
from data.data_loader import FinancialDataLoader
from models.adtcn  import ADTCN
from utils.metrics import compute_all_metrics

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def train_eval(params, tag, X_full, y_full, X_test, y_test):
    print(f"\n[RUN] {tag}: {params}", flush=True)
    t0 = time.time()
    m = ADTCN(cfg=ADTCN_CONFIG)
    m.optimal_params = dict(params)        # bypass the search; fix the config
    m.fit(X_full, y_full, verbose=True)
    met = compute_all_metrics(y_test, m.predict(X_test))
    print(f"[RUN] {tag}: Accuracy={met['Accuracy']:.4f}%  MCC={met['MCC']:.4f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return {k: float(v) for k, v in met.items()
            if isinstance(v, (int, float, np.floating))}


def main():
    loader = FinancialDataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load(verbose=True)
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])

    print(f"[DATA] train+val={len(y_full):,}  test={len(y_test):,}  "
          f"test fraud={int(y_test.sum())}", flush=True)

    out = {
        "task": "DB-BOA-tuned vs hand-set-default ADTCN (single full run, corrected bounded objective)",
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "data_split": {"test_size": DATA_CONFIG["test_size"], "val_size": DATA_CONFIG["val_size"],
                       "stratified": True, "random_state": ADTCN_CONFIG["random_state"]},
        "test_set": {"n": int(len(y_test)), "n_fraud": int(y_test.sum())},
        "architecture": ADTCN_CONFIG.get("architecture", "cnn"),
        "epoch_count": ADTCN_CONFIG["epoch_count"],
        "note": "Reproduces the comparison main.py prints (Phase 4) but does not persist. "
                "Tuned row cross-checks db_boa_results.json (MCC 0.677).",
    }
    path = os.path.join(RESULTS_DIR, "dbboa_vs_default.json")

    def save():
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[SAVE] {path}", flush=True)

    # Default config FIRST (the critical, previously-unbacked number), saving the
    # JSON incrementally after each model so partial progress survives interruption.
    default = train_eval(
        {"hidden_neurons": 128, "epoch_count": ADTCN_CONFIG["epoch_count"], "steps_per_epoch": 150},
        "hand-set default (128/30/150)", X_full, y_full, X_test, y_test)
    out["hand_set_default"] = {"hidden_neurons": 128, "steps_per_epoch": 150, "metrics": default}
    save()

    tuned = train_eval(
        {"hidden_neurons": 142, "epoch_count": ADTCN_CONFIG["epoch_count"], "steps_per_epoch": 76},
        "DB-BOA-tuned (142/30/76)", X_full, y_full, X_test, y_test)
    out["dbboa_tuned"] = {"hidden_neurons": 142, "steps_per_epoch": 76, "metrics": tuned}
    out["mcc_gap_default_minus_tuned"] = round(default["MCC"] - tuned["MCC"], 4)
    save()

    print(f"[DONE] default MCC={default['MCC']:.4f}  tuned MCC={tuned['MCC']:.4f}  "
          f"gap(default-tuned)={out['mcc_gap_default_minus_tuned']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
