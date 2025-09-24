
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

DESCRIPTION = """
Auto-detect, no-prompt NN trainer for sector sequences.
- Tries defaults (Ligier JS P320 @ Charlotte Roval 2025).
- If not found, scans telemetria/*/*/*.csv and picks the (car, track) with MOST CSVs.
- No interactive input; runs end-to-end or exits with a clear diagnostic listing available options.
"""

import os, re, glob, sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd

# --------------------- Torch optional import ---------------------
TORCH_OK = True
try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
except Exception as e:
    TORCH_OK = False
    TORCH_ERR = e

# --------------------- Utils ---------------------

def parse_meta_from_filename(path: str):
    name = Path(path).name
    parts = name.split(" - ")
    driver = parts[1] if len(parts) > 1 else "Unknown"
    m = re.search(r'(\d{1,2})[.:](\d{2})[.:](\d{3})', name)
    lap_time_sec = None
    if m:
        minutes, seconds, ms = map(int, m.groups())
        lap_time_sec = minutes*60 + seconds + ms/1000.0
    return driver, lap_time_sec

def read_csv_auto(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=';')

def detect_column(cols, candidates):
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        for low, orig in lower_map.items():
            if cand in low:
                return orig
    return None

def load_file_cols(path: str):
    df = read_csv_auto(path)
    col_dist = 'LapDistPct' if 'LapDistPct' in df.columns else detect_column(df.columns, ['lapdistpct','dist_pct','distancepercentage','splinepos','spline_pos'])
    col_speed = 'Speed' if 'Speed' in df.columns else detect_column(df.columns, ['speed_kph','speed','velocity'])
    col_brake = 'Brake' if 'Brake' in df.columns else detect_column(df.columns, ['brake'])
    col_throttle = 'Throttle' if 'Throttle' in df.columns else detect_column(df.columns, ['throttle'])
    return df, col_dist, col_speed, col_brake, col_throttle

# sectors loader (same format as sectors.yaml used in pipeline)
def load_sectors_from_file(sectors_path: Path, track_name: str):
    data = None
    if sectors_path and sectors_path.exists():
        try:
            if sectors_path.suffix.lower() in (".yaml", ".yml"):
                import yaml
                with open(sectors_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            elif sectors_path.suffix.lower() == ".json":
                import json
                data = json.loads(sectors_path.read_text(encoding="utf-8"))
        except Exception as e:
            print("Не удалось прочитать файл с секторами:", e)
            data = None
    if not isinstance(data, dict): return None
    tracks = data.get("tracks", {}) or {}
    default = data.get("default", None)
    if track_name in tracks: return tracks[track_name]
    key_lower = (track_name or "").lower()
    for k in tracks.keys():
        if k.lower() in key_lower or key_lower in k.lower():
            return tracks[k]
    return default

def pick_sectors(track_name: str):
    # fallback when no file
    return [("S1",0.0,1/3),("S2",1/3,2/3),("S3",2/3,1.0)]

# integrate sector time using dx/v scaled to total lap time
def integrate_sector_time(dist, speed_kph, lap_time_total, a, b):
    m = np.isfinite(dist) & np.isfinite(speed_kph)
    x = dist[m]; v = speed_kph[m] * (1000.0/3600.0)
    if len(x) < 5: return np.nan
    idx = np.argsort(x); x = x[idx]; v = v[idx]
    dx = np.diff(x, prepend=x[0]); dx[0]=0.0
    v = np.clip(v, 0.1, None)
    contrib = dx / v
    total = contrib.sum()
    if total <= 0: return np.nan
    mask = (x>=a) & (x<=b)
    sec_contrib = contrib[mask].sum()
    return lap_time_total * (sec_contrib / total)

def resample_sector_sequence(dist, arr, a, b, seq_len=256):
    # arr is the series to resample (speed/throttle/brake)
    m = np.isfinite(dist) & np.isfinite(arr)
    d = dist[m]; y = arr[m]
    if len(d) < 3: 
        return np.full((seq_len,), np.nan, dtype=float)
    idx = np.argsort(d); d = d[idx]; y = y[idx]
    # clip to sector range
    mask = (d >= a) & (d <= b)
    if not mask.any():
        return np.full((seq_len,), np.nan, dtype=float)
    d = d[mask]; y = y[mask]
    # normalize d to [0,1] within sector
    d0, d1 = d[0], d[-1]
    if d1 - d0 < 1e-6: 
        return np.full((seq_len,), np.nan, dtype=float)
    dn = (d - d0)/(d1 - d0)
    grid = np.linspace(0, 1, seq_len)
    yi = np.interp(grid, dn, y)
    return yi

# --------------------- Dataset building ---------------------

def build_nn_dataset(csvs, sectors, seq_len=256, use_features=("speed","throttle","brake")):
    rows = []
    for p in csvs:
        driver, lap_time = parse_meta_from_filename(p)
        df, col_dist, col_speed, col_brake, col_throttle = load_file_cols(p)
        if not col_dist or not col_speed: 
            continue
        dist = pd.to_numeric(df[col_dist], errors='coerce').values
        speed = pd.to_numeric(df[col_speed], errors='coerce').values
        brake = pd.to_numeric(df[col_brake], errors='coerce').values if col_brake else np.full_like(speed, np.nan)
        thr = pd.to_numeric(df[col_throttle], errors='coerce').values if col_throttle else np.full_like(speed, np.nan)

        # basic cleaning
        m = np.isfinite(dist) & np.isfinite(speed)
        if m.sum() < 10: 
            continue
        dist = np.clip(dist[m], 0, 1)
        speed = speed[m]
        brake = brake[m] if np.isfinite(brake).sum()>0 else np.full_like(speed, np.nan)
        thr = thr[m] if np.isfinite(thr).sum()>0 else np.full_like(speed, np.nan)

        for (name,a,b) in sectors:
            seqs = []
            if "speed" in use_features:
                seqs.append(resample_sector_sequence(dist, speed, a, b, seq_len=seq_len))
            if "throttle" in use_features:
                seqs.append(resample_sector_sequence(dist, thr, a, b, seq_len=seq_len))
            if "brake" in use_features:
                seqs.append(resample_sector_sequence(dist, brake, a, b, seq_len=seq_len))

            X = np.vstack(seqs).T  # (seq_len, n_features)
            if np.isnan(X).any():
                continue

            # target: sector_time (dx/v scaled)
            t_sec = integrate_sector_time(dist, speed, lap_time or np.nan, a, b)
            if not np.isfinite(t_sec):
                continue

            rows.append({
                "driver": driver,
                "file": Path(p).name,
                "sector": name,
                "X": X.astype(np.float32),   # seq_len x nfeat
                "y": float(t_sec)
            })
    if not rows:
        raise RuntimeError("Нет валидных последовательностей для обучения. Проверьте CSV и сектора.")
    df = pd.DataFrame(rows)
    # compute delta to best per sector for optional comparison
    best = df.groupby("sector")["y"].min().rename("y_best").reset_index()
    df = df.merge(best, on="sector", how="left")
    df["delta"] = df["y"] - df["y_best"]
    return df

# --------------------- PyTorch Dataset ---------------------

class SectorSequenceDataset(Dataset):
    def __init__(self, df, feature_key="X", target_key="y", standardize=True):
        self.df = df.reset_index(drop=True)
        self.feature_key = feature_key
        self.target_key = target_key
        self.standardize = standardize
        # compute global mean/std over all sequences (per feature)
        X_stack = np.concatenate([np.asarray(x)[None, ...] for x in self.df[self.feature_key].tolist()], axis=0)  # N x T x F
        if standardize:
            self.mean = np.nanmean(X_stack, axis=(0,1), keepdims=True)  # 1x1xF
            self.std = np.nanstd(X_stack, axis=(0,1), keepdims=True) + 1e-8
        else:
            self.mean = 0.0; self.std = 1.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        X = self.df.at[idx, self.feature_key].astype(np.float32)  # T x F
        y = np.float32(self.df.at[idx, self.target_key])
        # standardize per dataset
        Xn = (X - self.mean.squeeze()) / self.std.squeeze()
        return torch.from_numpy(Xn), torch.tensor([y], dtype=torch.float32)

# --------------------- Models ---------------------

class LSTMRegressor(nn.Module):
    def __init__(self, n_feat, hidden=64, num_layers=2, bidir=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_feat, hidden_size=hidden, num_layers=num_layers,
                            batch_first=True, bidirectional=bidir, dropout=0.2)
        out_dim = hidden*(2 if bidir else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(out_dim, 1)
        )

    def forward(self, x):
        # x: B x T x F
        out, _ = self.lstm(x)
        # take last time step
        last = out[:, -1, :]
        y = self.head(last)
        return y

class CNN1DRegressor(nn.Module):
    def __init__(self, n_feat, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_feat, base, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(base, base*2, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(base*2, base*4, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Sequential(nn.Linear(base*4, base*4), nn.ReLU(), nn.Dropout(0.2), nn.Linear(base*4, 1))

    def forward(self, x):
        # x: B x T x F -> B x F x T
        z = x.transpose(1,2)
        z = self.net(z).squeeze(-1)
        y = self.head(z)
        return y

# --------------------- Train/Eval ---------------------

def train_one_model(model, train_loader, val_loader, epochs=30, lr=1e-3, device="cpu"):
    crit = nn.L1Loss()  # optimize MAE
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = 1e9; best_state = None
    hist = {"epoch": [], "train_mae": [], "val_mae": []}
    for ep in range(1, epochs+1):
        model.train(); tr_loss=0.0; n=0
        for X,y in train_loader:
            X=X.to(device); y=y.to(device)
            opt.zero_grad()
            pred = model(X)
            loss = crit(pred, y)
            loss.backward(); opt.step()
            tr_loss += loss.item()*X.size(0); n += X.size(0)
        train_mae = tr_loss/max(1,n)

        # val
        model.eval(); va_loss=0.0; m=0
        with torch.no_grad():
            for X,y in val_loader:
                X=X.to(device); y=y.to(device)
                pred = model(X)
                loss = crit(pred, y)
                va_loss += loss.item()*X.size(0); m += X.size(0)
        val_mae = va_loss/max(1,m)

        hist["epoch"].append(ep); hist["train_mae"].append(train_mae); hist["val_mae"].append(val_mae)
        if val_mae < best_val:
            best_val = val_mae
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        print(f"Epoch {ep:3d}/{epochs}  train MAE={train_mae:.4f}  val MAE={val_mae:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(hist)

# --------------------- Split by drivers ---------------------

def group_train_val_split(df, val_ratio=0.25, seed=42):
    drivers = sorted(df["driver"].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(drivers)
    n_val = max(1, int(len(drivers)*val_ratio))
    val_drivers = set(drivers[:n_val])
    train_idx = df.index[~df["driver"].isin(val_drivers)].tolist()
    val_idx = df.index[df["driver"].isin(val_drivers)].tolist()
    return train_idx, val_idx, list(val_drivers)

# --------------------- Auto-detect helpers ---------------------

def pick_best_pair(telemetry_root: Path):
    pairs = {}
    for car_dir in telemetry_root.glob("*"):
        if not car_dir.is_dir(): continue
        for track_dir in car_dir.glob("*"):
            if not track_dir.is_dir(): continue
            n = len(list(track_dir.glob("*.csv")))
            if n > 0:
                pairs[(car_dir.name, track_dir.name)] = n
    if not pairs:
        return None, None
    # choose largest count
    (car, track), _ = max(pairs.items(), key=lambda kv: kv[1])
    return car, track

# --------------------- Main ---------------------

def main():
    ap = argparse.ArgumentParser(description="Auto-detect, no-prompt NN training on sector sequences (LSTM/CNN).")
    ap.add_argument('--telemetry-root', type=str, default='telemetria')
    ap.add_argument('--result-root', type=str, default='result')
    ap.add_argument('--sectors-file', type=str, default='sectors.yaml')
    ap.add_argument('--model', choices=['lstm','cnn1d'], default='lstm')
    ap.add_argument('--seq-len', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--car', type=str, default=None)
    ap.add_argument('--track', type=str, default=None)
    args = ap.parse_args()

    if not TORCH_OK:
        print("PyTorch недоступен в окружении:", TORCH_ERR)
        print("Установите torch, например: pip install torch --index-url https://download.pytorch.org/whl/cu121 (или cpu)")
        sys.exit(1)

    tel_root = Path(args.telemetry_root)

    # Try provided or defaults
    car = args.car or "Ligier JS P320"
    track = args.track or "Charlotte Motor Speedway (Roval 2025)"
    track_dir = tel_root / car / track

    if not track_dir.exists() or not list(track_dir.glob("*.csv")):
        # Auto-detect best available pair
        auto_car, auto_track = pick_best_pair(tel_root)
        if auto_car is None:
            # Build a diagnostic of what's inside telemetry_root
            if not tel_root.exists():
                print("Папка telemetria не найдена:", tel_root.resolve())
            else:
                print("Не найдено ни одной пары <машина>/<трасса> с CSV в:", tel_root.resolve())
                print("Структура должна быть telemetria/<машина>/<трасса>/*.csv")
                cars = [p.name for p in tel_root.glob("*") if p.is_dir()]
                if cars:
                    print("Найдены машины:", ", ".join(cars))
            sys.exit(1)
        car, track = auto_car, auto_track
        track_dir = tel_root / car / track
        print(f"[Auto-detect] Использую пару: {car} / {track}  (CSV: {len(list(track_dir.glob('*.csv')))} шт.)")

    csvs = sorted(glob.glob(str(track_dir / "*.csv")))
    if not csvs:
        print("Нет CSV файлов в:", track_dir.resolve())
        sys.exit(1)

    # sectors
    sectors = load_sectors_from_file(Path(args.sectors_file), track) or pick_sectors(track)

    # dataset
    df = build_nn_dataset(csvs, sectors, seq_len=args.seq_len)
    out_dir = Path(args.result_root) / car / track / "nn"
    out_dir.mkdir(parents=True, exist_ok=True)

    # save a sample row to inspect shapes
    np.save(out_dir/"sample_X.npy", df.iloc[0]["X"])
    df[["driver","file","sector","y","delta"]].to_csv(out_dir/"nn_targets.csv", index=False)

    # split by drivers
    train_idx, val_idx, val_drivers = group_train_val_split(df, val_ratio=0.25, seed=42)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    print("Валидация по пилотам:", val_drivers)

    # dataset objects
    train_ds = SectorSequenceDataset(train_df)
    val_ds = SectorSequenceDataset(val_df)

    # loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, drop_last=False)

    # model
    n_feat = train_df.iloc[0]["X"].shape[1]
    if args.model == "lstm":
        model = LSTMRegressor(n_feat=n_feat, hidden=64, num_layers=2, bidir=True)
    else:
        model = CNN1DRegressor(n_feat=n_feat, base=32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model, hist = train_one_model(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device)

    # save model & history
    torch.save(model.state_dict(), out_dir / f"nn_{args.model}.pt")
    hist.to_csv(out_dir / f"nn_{args.model}_history.csv", index=False)

    # plot curves
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,4))
        plt.plot(hist["epoch"], hist["train_mae"], label="train MAE")
        plt.plot(hist["epoch"], hist["val_mae"], label="val MAE")
        plt.xlabel("Epoch"); plt.ylabel("MAE (sec)"); plt.title(f"NN ({args.model}) learning curves")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / f"nn_{args.model}_learning_curves.png", dpi=160)
        plt.close()
    except Exception as e:
        print("Не удалось нарисовать график обучения:", e)

    # compare with classical ML (if available)
    comp_rows = []
    ml_metrics_path = Path(args.result_root)/car/track/"ml_sector_metrics.csv"
    if ml_metrics_path.exists():
        mlm = pd.read_csv(ml_metrics_path)
        row = mlm[mlm["target"]=="delta_time_sec"].head(1)
        if row.empty and not mlm.empty: row = mlm.head(1)
        if not row.empty:
            comp_rows.append({"Method":"Classical ML", "MAE": float(row["MAE_mean"].values[0])})
    if len(hist):
        best_nn = float(np.min(hist["val_mae"].values))
        comp_rows.append({"Method": f"NN-{args.model}", "MAE": best_nn})
    if comp_rows:
        comp = pd.DataFrame(comp_rows)
        comp.to_csv(out_dir / "nn_vs_ml_comparison.csv", index=False)

    with open(out_dir / "nn_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Neural network training done for {car} — {track}\n")
        f.write(f"Validation drivers: {', '.join(val_drivers)}\n")
        if comp_rows:
            f.write("\nComparison (lower MAE is better):\n")
            for r in comp_rows:
                f.write(f" - {r['Method']}: MAE={r['MAE']:.4f} sec\n")

    print("Готово. Файлы сохранены в:", out_dir.resolve())
    print(" - model:", out_dir / f"nn_{args.model}.pt")
    print(" - history:", out_dir / f"nn_{args.model}_history.csv")
    print(" - learning curves:", out_dir / f"nn_{args.model}_learning_curves.png")
    if comp_rows:
        print(" - comparison:", out_dir / "nn_vs_ml_comparison.csv")
    print(" - targets:", out_dir / "nn_targets.csv")

if __name__ == "__main__":
    main()
