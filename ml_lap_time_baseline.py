
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob, argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import joblib

# ---------- helpers ----------
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

def load_file(path: str):
    df = read_csv_auto(path)
    col_speed = 'Speed' if 'Speed' in df.columns else detect_column(df.columns, ['speed_kph','speed','velocity'])
    col_brake = 'Brake' if 'Brake' in df.columns else detect_column(df.columns, ['brake'])
    col_throttle = 'Throttle' if 'Throttle' in df.columns else detect_column(df.columns, ['throttle'])
    col_latacc = 'LatAccel' if 'LatAccel' in df.columns else detect_column(df.columns, ['latacc','lateral'])
    col_longacc = 'LongAccel' if 'LongAccel' in df.columns else detect_column(df.columns, ['longacc','longitudinal'])
    col_rpm = 'RPM' if 'RPM' in df.columns else detect_column(df.columns, ['rpm'])
    col_steer = 'SteeringWheelAngle' if 'SteeringWheelAngle' in df.columns else detect_column(df.columns, ['steer','steering'])
    col_gear = 'Gear' if 'Gear' in df.columns else detect_column(df.columns, ['gear'])
    col_distpct = 'LapDistPct' if 'LapDistPct' in df.columns else detect_column(df.columns, ['lapdistpct','dist_pct','distancepercentage','splinepos','spline_pos'])
    return {'df': df, 'speed': col_speed, 'brake': col_brake, 'throttle': col_throttle, 'latacc': col_latacc, 'longacc': col_longacc, 'rpm': col_rpm, 'steer': col_steer, 'gear': col_gear, 'dist': col_distpct}

def ensure_numeric(series):
    return pd.to_numeric(series, errors='coerce')

# ---------- feature engineering ----------
def compute_lap_features(path):
    driver, lap_time = parse_meta_from_filename(path)
    d = load_file(path)
    df = d['df']
    feats = {'driver': driver, 'file': Path(path).name, 'lap_time_sec': lap_time}
    # helpers
    def col(c): 
        return ensure_numeric(df[c]) if (c and c in df.columns) else pd.Series(dtype=float)
    speed = col(d['speed']); throttle = col(d['throttle']); brake = col(d['brake'])
    latacc = col(d['latacc']); longacc = col(d['longacc']); rpm = col(d['rpm'])
    steer = col(d['steer']); gear = col(d['gear']); dist = col(d['dist'])

    def safe_pct(s, q): return s.quantile(q) if s.size else np.nan
    def ratio(mask): 
        s = mask.astype(float) if mask.size else pd.Series(dtype=float)
        return s.mean() if s.size else np.nan

    feats.update({
        'speed_mean': speed.mean(), 'speed_p95': safe_pct(speed, 0.95), 'speed_var': speed.var(),
        'throttle_mean': throttle.mean(), 'full_throttle_ratio': ratio(throttle>0.9),
        'brake_ratio': ratio(brake>0.1),
        'latacc_max': latacc.max(), 'longacc_max': longacc.max(),
        'rpm_mean': rpm.mean(), 'rpm_max': rpm.max(),
        'steer_abs_mean': steer.abs().mean() if not steer.empty else np.nan,
        'gear_shift_count': (gear.shift()!=gear).sum() if not gear.empty else np.nan,
        'samples': len(df)
    })
    return feats

def main():
    ap = argparse.ArgumentParser(description='Train ML baseline to predict lap time from iRacing CSVs.')
    ap.add_argument('--telemetry-root', type=str, default='telemetria', help='telemetria/<car>/<track>/*.csv')
    ap.add_argument('--result-root', type=str, default='result', help='where to save features/model: result/<car>/<track>/')
    ap.add_argument('--car', type=str, default=None)
    ap.add_argument('--track', type=str, default=None)
    args = ap.parse_args()

    tel_root = Path(args.telemetry_root); res_root = Path(args.result_root)

    # interactive selection if not provided
    def choose_from(items, prompt):
        if not items:
            print("Нет вариантов для выбора."); sys.exit(1)
        for i, name in enumerate(items, 1): print(f"{i}) {name}")
        while True:
            sel = input(f"{prompt} (номер): ").strip()
            if sel.isdigit() and 1 <= int(sel) <= len(items): return items[int(sel)-1]
            print("Неверный ввод.")

    cars = sorted([p.name for p in tel_root.iterdir() if p.is_dir()])
    car = args.car if args.car in cars else None
    if car is None:
        print("Выберите машину:"); car = choose_from(cars, "Машина")
    tracks = sorted([p.name for p in (tel_root/car).iterdir() if p.is_dir()])
    track = args.track if args.track in tracks else None
    if track is None:
        print(f"Выберите трассу для {car}:"); track = choose_from(tracks, "Трасса")

    in_dir = tel_root / car / track
    out_dir = res_root / car / track; out_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(glob.glob(str(in_dir / "*.csv")))
    if not csvs:
        print("Нет CSV в", in_dir); sys.exit(1)

    # features
    rows = [compute_lap_features(p) for p in csvs]
    feats = pd.DataFrame(rows)
    feats_path = out_dir / "features.csv"
    feats.to_csv(feats_path, index=False)

    # basic cleaning
    df = feats.dropna(subset=['lap_time_sec']).copy()
    y = df['lap_time_sec'].values
    feature_cols = [c for c in df.columns if c not in ['driver','file','lap_time_sec']]
    X = df[feature_cols].values

    # CV by driver (to avoid leakage)
    groups = df['driver'].values
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    model = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=42)
    scores_mae = []; scores_rmse = []; scores_r2 = []
    for train, test in cv.split(X, y, groups):
        model.fit(X[train], y[train])
        pred = model.predict(X[test])
        scores_mae.append(mean_absolute_error(y[test], pred))
        scores_rmse.append(np.sqrt(mean_squared_error(y[test], pred)))
        scores_r2.append(r2_score(y[test], pred))

    report = pd.DataFrame({
        'MAE_sec': [np.mean(scores_mae), np.std(scores_mae)],
        'RMSE_sec': [np.mean(scores_rmse), np.std(scores_rmse)],
        'R2': [np.mean(scores_r2), np.std(scores_r2)],
    }, index=['mean','std'])
    report_path = out_dir / "ml_cv_report.csv"
    report.to_csv(report_path)

    # Train on all and export importances
    model.fit(X, y)
    importances = permutation_importance(model, X, y, n_repeats=20, random_state=42, n_jobs=-1)
    imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importances.importances_mean}).sort_values('importance', ascending=False)
    imp_path = out_dir / "ml_feature_importance.csv"
    imp_df.to_csv(imp_path, index=False)
    joblib.dump(model, out_dir / "ml_random_forest.joblib")

    print("Готово. Файлы сохранены в:", out_dir.resolve())
    print(" - features:", feats_path)
    print(" - CV report:", report_path)
    print(" - feature importance:", imp_path)
    print(" - model:", out_dir / "ml_random_forest.joblib")

if __name__ == '__main__':
    main()
