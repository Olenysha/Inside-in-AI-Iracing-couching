
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob, argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ============================ Common utils ============================

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

def choose_from(items, prompt):
    if not items:
        print("Нет вариантов для выбора."); sys.exit(1)
    for i, name in enumerate(items, 1):
        print(f"{i}) {name}")
    while True:
        sel = input(f"{prompt} (введите номер): ").strip()
        if sel.isdigit():
            idx = int(sel)
            if 1 <= idx <= len(items):
                return items[idx-1]
        print("Неверный ввод.")

def ensure_numeric(series):
    return pd.to_numeric(series, errors='coerce')

def moving_average(x, w):
    if len(x)==0: return x
    w = max(1, int(w))
    return pd.Series(x).rolling(w, min_periods=1, center=True).mean().values

# ============================ Sector defs ============================

def load_sectors_from_file(sectors_path: Path, track_name: str):
    """
    Читает sectors.yaml / sectors.json и подбирает сектора для указанного трека.
    Правила подбора:
      1) Точное совпадение ключа.
      2) Поиск по подстроке (без регистра).
      3) Фолбэк "default".
    Формат элементов: [name, start, end], где start/end в [0,1].
    """
    data = None
    if sectors_path and sectors_path.exists():
        try:
            if sectors_path.suffix.lower() in (".yaml", ".yml"):
                try:
                    import yaml  # optional
                except Exception:
                    # Простейший YAML-парсер: поддерживаем только наш формат с минимальной структурой
                    # Если PyYAML не установлен, попробуем загрузить как JSON-подобный с заменой:
                    txt = sectors_path.read_text(encoding="utf-8")
                    # Плохая, но рабочая эвристика: преобразуем к JSON через замену двоеточий по ключам.
                    # Лучше просто установить PyYAML: pip install pyyaml
                    raise RuntimeError("Для .yaml рекомендуется установить pyyaml (pip install pyyaml).")
                else:
                    with open(sectors_path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
            elif sectors_path.suffix.lower() == ".json":
                import json
                data = json.loads(sectors_path.read_text(encoding="utf-8"))
        except Exception as e:
            print("Не удалось прочитать файл с секторами:", e)
            data = None

    # если не прочитали — возврат None
    if not isinstance(data, dict):
        return None

    tracks = data.get("tracks", {}) or {}
    default = data.get("default", None)

    # 1) точное совпадение
    if track_name in tracks:
        return tracks[track_name]

    # 2) подстрока
    key_lower = track_name.lower()
    for k in tracks.keys():
        if k.lower() in key_lower or key_lower in k.lower():
            return tracks[k]

    # 3) фолбэк
    return default

# --- built-ins ---

def monza_named_sectors():
    return [
        ("T1-T2 (Rettifilo)", 0.00, 0.18),
        ("Curva Grande",      0.18, 0.30),
        ("Var. della Roggia", 0.30, 0.39),
        ("Lesmo 1-2",         0.39, 0.56),
        ("Serraglio",         0.56, 0.67),
        ("Ascari",            0.67, 0.83),
        ("Parabolica",        0.83, 1.00),
    ]

def default_three_sectors():
    return [("S1",0.0,1/3),("S2",1/3,2/3),("S3",2/3,1.0)]

def pick_sectors(track_name: str):
    name = (track_name or "").lower()
    if "monza" in name:
        return monza_named_sectors()
    return default_three_sectors()

# scale-free integration of dt ~ dx/v, scaled to lap_time_total
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

# ============================ Loading helpers ============================

def load_file(path: str):
    df = read_csv_auto(path)
    col_lon = 'Lon' if 'Lon' in df.columns else detect_column(df.columns, ['lon','longitude','worldpositionx'])
    col_lat = 'Lat' if 'Lat' in df.columns else detect_column(df.columns, ['lat','latitude','worldpositiony'])
    col_speed = 'Speed' if 'Speed' in df.columns else detect_column(df.columns, ['speed_kph','speed','velocity'])
    col_brake = 'Brake' if 'Brake' in df.columns else detect_column(df.columns, ['brake'])
    col_throttle = 'Throttle' if 'Throttle' in df.columns else detect_column(df.columns, ['throttle'])
    col_distpct = 'LapDistPct' if 'LapDistPct' in df.columns else detect_column(df.columns, ['lapdistpct','dist_pct','distancepercentage','splinepos','spline_pos'])
    return {'df': df, 'lon': col_lon, 'lat': col_lat, 'speed': col_speed, 'brake': col_brake, 'throttle': col_throttle, 'dist': col_distpct}

def load_lap(path: str):
    d = load_file(path)
    df = d["df"]
    def col(c):
        s = pd.to_numeric(df[c], errors='coerce') if (c and c in df.columns) else pd.Series(dtype=float)
        return s.values
    return {
        "dist": col(d["dist"]),
        "speed": col(d["speed"]),
        "brake": col(d["brake"]),
        "throttle": col(d["throttle"])
    }

# ============================ Plotting ============================

def plot_trajectories(files_info, outdir):
    plt.figure(figsize=(7,7))
    for info in files_info:
        df = info['data']['df']
        lon = info['data']['lon']; lat = info['data']['lat']
        if lon and lat and lon in df.columns and lat in df.columns:
            x = ensure_numeric(df[lon]); y = ensure_numeric(df[lat])
            m = x.notna() & y.notna()
            plt.plot(x[m][::5], y[m][::5], label=info['driver'], alpha=0.9)
    ax = plt.gca(); ax.set_aspect('equal', adjustable='box')
    plt.title('Trajectories (Lon/Lat) — all drivers')
    plt.xlabel('Longitude'); plt.ylabel('Latitude')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    p = Path(outdir) / 'trajectories_overlay.png'
    plt.savefig(p, dpi=180); plt.close(); return str(p)

def plot_speed_heatmap_fastest(files_info, outdir):
    candidates = [fi for fi in files_info if fi['lap_time_sec'] is not None]
    if not candidates: return None
    fastest = min(candidates, key=lambda fi: fi['lap_time_sec'])
    df = fastest['data']['df']; lon = fastest['data']['lon']; lat = fastest['data']['lat']; spd = fastest['data']['speed']
    if not (lon and lat and spd and lon in df.columns and lat in df.columns and spd in df.columns):
        return None
    x = ensure_numeric(df[lon]); y = ensure_numeric(df[lat]); c = ensure_numeric(df[spd])
    m = x.notna() & y.notna() & c.notna()
    plt.figure(figsize=(7,7))
    plt.scatter(x[m][::3], y[m][::3], c=c[m][::3], s=6)
    ax = plt.gca(); ax.set_aspect('equal', adjustable='box')
    plt.title(f"Speed map (color=speed) — {fastest['driver']}")
    plt.xlabel('Longitude'); plt.ylabel('Latitude')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    p = Path(outdir) / 'speed_heatmap_fastest.png'
    plt.savefig(p, dpi=180); plt.close(); return str(p)

def plot_braking_points(files_info, outdir, brake_threshold: float):
    plt.figure(figsize=(7,7)); any_points = False
    for info in files_info:
        df = info['data']['df']
        lon = info['data']['lon']; lat = info['data']['lat']; brk = info['data']['brake']
        if lon and lat and brk and all(c in df.columns for c in [lon,lat,brk]):
            x = ensure_numeric(df[lon]); y = ensure_numeric(df[lat]); b = ensure_numeric(df[brk])
            m = x.notna() & y.notna() & b.notna() & (b >= brake_threshold)
            if m.any():
                plt.scatter(x[m][::3], y[m][::3], s=4, label=info['driver'], alpha=0.7); any_points = True
    if not any_points: plt.close(); return None
    ax = plt.gca(); ax.set_aspect('equal', adjustable='box')
    plt.title(f'Braking points (Brake ≥ {brake_threshold}) — all drivers')
    plt.xlabel('Longitude'); plt.ylabel('Latitude')
    plt.legend(markerscale=3); plt.grid(True, alpha=0.3); plt.tight_layout()
    p = Path(outdir) / 'braking_points.png'
    plt.savefig(p, dpi=180); plt.close(); return str(p)

def plot_speed_profile(files_info, outdir, smooth_window=25, sectors=None):
    plt.figure(figsize=(10,4)); any_plots = False
    for info in files_info:
        df = info['data']['df']; dist = info['data']['dist']; spd = info['data']['speed']
        if dist and spd and all(c in df.columns for c in [dist, spd]):
            x = ensure_numeric(df[dist]); y = ensure_numeric(df[spd])
            m = x.notna() & y.notna(); x = x[m].values; y = y[m].values
            if len(x) < 5: continue
            idx = np.argsort(x); x = x[idx]; y = y[idx]
            y_smooth = pd.Series(y).rolling(smooth_window, min_periods=1, center=True).mean().values
            plt.plot(x, y_smooth, label=info['driver'], alpha=0.9); any_plots = True
    if not any_plots: plt.close(); return None
    if sectors:
        for name,a,b in sectors:
            plt.axvspan(a, b, alpha=0.08)
            plt.text((a+b)/2, plt.ylim()[1]*0.95, name, ha='center', va='top', fontsize=8)
    plt.title('Speed profile vs lap distance (0–1)'); plt.xlabel('LapDistPct'); plt.ylabel('Speed')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    p = Path(outdir) / 'speed_profile.png'; plt.savefig(p, dpi=160); plt.close(); return str(p)

def plot_delta_speed(files_info, outdir, smooth_window=25, sectors=None):
    candidates = [fi for fi in files_info if fi['lap_time_sec'] is not None]
    if not candidates: return None
    fastest = min(candidates, key=lambda fi: fi['lap_time_sec'])
    df_fast = fastest['data']['df']; dist_f = fastest['data']['dist']; spd_f = fastest['data']['speed']
    if not (dist_f and spd_f and dist_f in df_fast.columns and spd_f in df_fast.columns):
        return None
    xf = ensure_numeric(df_fast[dist_f]).values; yf = ensure_numeric(df_fast[spd_f]).values
    mf = np.isfinite(xf) & np.isfinite(yf); xf = xf[mf]; yf = yf[mf]
    if len(xf) < 5: return None
    idx = np.argsort(xf); xf = xf[idx]; yf = yf[idx]
    yf = pd.Series(yf).rolling(smooth_window, min_periods=1, center=True).mean().values
    xg = np.linspace(0,1,2000); yf_interp = np.interp(xg, xf, yf, left=np.nan, right=np.nan)

    plt.figure(figsize=(10,4))
    for info in files_info:
        df = info['data']['df']; dist = info['data']['dist']; spd = info['data']['speed']
        if dist and spd and all(c in df.columns for c in [dist, spd]):
            x = ensure_numeric(df[dist]).values; y = ensure_numeric(df[spd]).values
            m = np.isfinite(x) & np.isfinite(y); x = x[m]; y = y[m]
            if len(x) < 5: continue
            idx = np.argsort(x); x = x[idx]; y = y[idx]
            y = pd.Series(y).rolling(smooth_window, min_periods=1, center=True).mean().values
            y_interp = np.interp(xg, x, y, left=np.nan, right=np.nan)
            delta = y_interp - yf_interp
            plt.plot(xg, delta, label=info['driver'], alpha=0.95)
    if sectors:
        for name,a,b in sectors:
            plt.axvspan(a, b, alpha=0.08)
            plt.text((a+b)/2, 0.95*plt.ylim()[1], name, ha='center', va='top', fontsize=8)
    plt.axhline(0, linewidth=1); plt.title('Delta speed vs fastest (km/h)')
    plt.xlabel('LapDistPct'); plt.ylabel('Δ speed (km/h)')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    p = Path(outdir) / 'delta_speed.png'; plt.savefig(p, dpi=160); plt.close(); return str(p)

# ============================ Coaching ============================

def first_cross_up(series, thresh, a_idx, b_idx):
    s = series[a_idx:b_idx+1]
    if len(s) < 2: return None
    prev = s[:-1] < thresh; now = s[1:] >= thresh
    idx = np.where(prev & now)[0]
    if idx.size>0: return a_idx + idx[0] + 1
    return None

def last_cross_up_after_min(speed, throttle, thresh, a_idx, b_idx):
    s = speed[a_idx:b_idx+1]; t = throttle[a_idx:b_idx+1]
    if len(s)==0: return None
    min_idx_local = int(np.nanargmin(s)); start = a_idx + min_idx_local
    for i in range(start, b_idx+1):
        if np.isfinite(t[i-a_idx]) and t[i-a_idx] >= thresh: return i
    return None

def coaching_from_csvs(csvs, track_name, sectors_file, track_length_m=5793.0, brake_threshold=0.1, throttle_threshold=0.5, smooth_w=25):
    laps = []
    for p in csvs:
        driver, laptime = parse_meta_from_filename(p)
        lap = load_lap(p)
        dist = lap['dist']; spd = lap['speed']; brk = lap['brake']; thr = lap['throttle']
        if dist.size==0 or spd.size==0: continue
        dist = np.clip(dist, 0, 1)
        spd_s = moving_average(spd, smooth_w)
        thr_s = moving_average(thr, max(3, smooth_w//3)) if thr.size else thr
        brk_s = moving_average(brk, max(3, smooth_w//3)) if brk.size else brk
        laps.append({"path": p, "driver": driver, "lap_time": laptime, "dist": dist, "speed": spd_s, "brake": brk_s, "throttle": thr_s})
    if not laps: raise RuntimeError("No usable laps.")
    laps_with_times = [L for L in laps if L["lap_time"] is not None]
    fastest = min(laps_with_times, key=lambda L: L["lap_time"]) if laps_with_times else laps[0]
    sectors = load_sectors_from_file(Path(sectors_file), track_name) or pick_sectors(track_name)

    ref = {}
    for name, a, b in sectors:
        dist = fastest["dist"]
        idx_a = int(np.searchsorted(dist, a, side="left"))
        idx_b = int(np.searchsorted(dist, b, side="right")) - 1
        idx_a = max(0, min(idx_a, len(dist)-1)); idx_b = max(0, min(idx_b, len(dist)-1))
        if idx_b <= idx_a: idx_a, idx_b = 0, len(dist)-1
        t_sec = integrate_sector_time(fastest["dist"], fastest["speed"], fastest["lap_time"] or np.nan, a, b)
        bp = first_cross_up(fastest["brake"], brake_threshold, idx_a, idx_b)
        tp = last_cross_up_after_min(fastest["speed"], fastest["throttle"], throttle_threshold, idx_a, idx_b)
        vmin = np.nanmin(fastest["speed"][idx_a:idx_b+1]) if idx_b>idx_a else np.nan
        ref[name] = {"t": t_sec, "bp": bp, "tp": tp, "vmin": vmin}

    rows, suggestions = [], []
    for L in laps:
        if L is fastest: continue
        for name, a, b in sectors:
            dist = L["dist"]
            idx_a = int(np.searchsorted(dist, a, side="left"))
            idx_b = int(np.searchsorted(dist, b, side="right")) - 1
            idx_a = max(0, min(idx_a, len(dist)-1)); idx_b = max(0, min(idx_b, len(dist)-1))
            if idx_b <= idx_a: idx_a, idx_b = 0, len(dist)-1
            t_sec = integrate_sector_time(L["dist"], L["speed"], L["lap_time"] or np.nan, a, b)
            t_ref = ref[name]["t"]
            if not np.isfinite(t_sec) or not np.isfinite(t_ref): continue
            delta_t = t_sec - t_ref
            bp = first_cross_up(L["brake"], brake_threshold, idx_a, idx_b)
            tp = last_cross_up_after_min(L["speed"], L["throttle"], throttle_threshold, idx_a, idx_b)
            vmin = np.nanmin(L["speed"][idx_a:idx_b+1]) if idx_b>idx_a else np.nan

            def idx_to_m(idx):
                if idx is None or idx<0 or idx>=len(L["dist"]): return np.nan
                return float(L["dist"][idx]) * track_length_m
            bp_m = idx_to_m(bp); tp_m = idx_to_m(tp)
            bp_ref_m = idx_to_m(ref[name]["bp"]); tp_ref_m = idx_to_m(ref[name]["tp"]); vmin_ref = ref[name]["vmin"]

            d_bp = bp_m - bp_ref_m if np.isfinite(bp_m) and np.isfinite(bp_ref_m) else np.nan
            d_tp = tp_m - tp_ref_m if np.isfinite(tp_m) and np.isfinite(tp_ref_m) else np.nan
            d_vmin = (vmin - vmin_ref) if np.isfinite(vmin) and np.isfinite(vmin_ref) else np.nan

            rows.append({"driver": L["driver"], "sector": name, "delta_time_sec": delta_t,
                         "delta_brake_m": d_bp, "delta_throttle_m": d_tp, "delta_min_speed_kph": d_vmin})

            reasons = []
            if np.isfinite(d_bp) and d_bp < -5: reasons.append("рано начал тормозить")
            elif np.isfinite(d_bp) and d_bp > 5: reasons.append("поздно начал тормозить")
            if np.isfinite(d_tp) and d_tp > 5: reasons.append("позно открыл газ")
            elif np.isfinite(d_tp) and d_tp < -5: reasons.append("слишком рано открыл газ (возможна пробуксовка)")
            if np.isfinite(d_vmin) and d_vmin < -3: reasons.append("слишком низкая минимальная скорость в повороте")
            elif np.isfinite(d_vmin) and d_vmin > 3: reasons.append("высокая минимальная скорость (риск выноса)")
            if delta_t > 0.05:
                reason_txt = "; ".join(reasons) if reasons else "менее эффективная траектория/фазы торможения и газа"
                suggestions.append(f"Пилот {L['driver']} проигрывает {delta_t:.2f}с в секции {name}: {reason_txt}.")
    return ref, pd.DataFrame(rows), suggestions

# ============================ ML per-sector ============================

def sector_features_for_lap(lap, lap_time, sectors, track_length_m=5793.0, brake_thr=0.1, throttle_thr=0.5, smooth_w=25):
    dist = np.clip(lap["dist"], 0, 1)
    speed = moving_average(lap["speed"], smooth_w)
    throttle = moving_average(lap["throttle"], max(3, smooth_w//3)) if lap["throttle"].size else lap["throttle"]
    brake = moving_average(lap["brake"], max(3, smooth_w//3)) if lap["brake"].size else lap["brake"]
    feats = []
    for name, a, b in sectors:
        idx_a = int(np.searchsorted(dist, a, side="left"))
        idx_b = int(np.searchsorted(dist, b, side="right")) - 1
        idx_a = max(0, min(idx_a, len(dist)-1)); idx_b = max(0, min(idx_b, len(dist)-1))
        if idx_b <= idx_a: idx_a, idx_b = 0, len(dist)-1
        t_sec = integrate_sector_time(dist, speed, lap_time or np.nan, a, b)
        # points
        def first_cross_up(series, thresh, a_idx, b_idx):
            s = series[a_idx:b_idx+1]
            if len(s) < 2: return None
            prev = s[:-1] < thresh; now = s[1:] >= thresh
            idx = np.where(prev & now)[0]
            return (a_idx + idx[0] + 1) if idx.size>0 else None
        def last_cross_up_after_min(speed, throttle, thresh, a_idx, b_idx):
            s = speed[a_idx:b_idx+1]; t = throttle[a_idx:b_idx+1]
            if len(s)==0: return None
            min_idx_local = int(np.nanargmin(s)); start = a_idx + min_idx_local
            for i in range(start, b_idx+1):
                if np.isfinite(t[i-a_idx]) and t[i-a_idx] >= thresh: return i
            return None
        bp = first_cross_up(brake, 0.1, idx_a, idx_b)
        tp = last_cross_up_after_min(speed, throttle, 0.5, idx_a, idx_b)
        def idx_to_m(idx): return float(dist[idx])*track_length_m if (idx is not None and 0 <= idx < len(dist)) else np.nan
        feats.append({
            "sector": name,
            "sector_time_sec": t_sec,
            "brake_start_m": idx_to_m(bp),
            "throttle_on_m": idx_to_m(tp),
            "entry_speed_kph": speed[idx_a] if idx_a < len(speed) else np.nan,
            "exit_speed_kph": speed[idx_b] if idx_b < len(speed) else np.nan,
            "min_speed_kph": float(np.nanmin(speed[idx_a:idx_b+1])) if idx_b>idx_a else np.nan,
            "max_speed_kph": float(np.nanmax(speed[idx_a:idx_b+1])) if idx_b>idx_a else np.nan,
            "mean_speed_kph": float(np.nanmean(speed[idx_a:idx_b+1])) if idx_b>idx_a else np.nan,
            "var_speed": float(np.nanvar(speed[idx_a:idx_b+1])) if idx_b>idx_a else np.nan,
            "full_throttle_ratio": float(np.nanmean(throttle[idx_a:idx_b+1]>=0.9)) if throttle.size else np.nan,
            "brake_ratio": float(np.nanmean(brake[idx_a:idx_b+1]>=0.1)) if brake.size else np.nan
        })
    return feats

def train_ml_sector(data, out_dir):
    try:
        from sklearn.model_selection import GroupKFold
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.inspection import permutation_importance
        SK_AVAIL = True
    except Exception as e:
        print("Внимание: scikit-learn недоступен, пропускаю обучение ML. Ошибка:", e)
        SK_AVAIL = False

    metrics = None; imp_time = None; imp_delta = None; model_time = None; model_delta = None
    if SK_AVAIL and not data.empty:
        feature_cols = [
            "brake_start_m","throttle_on_m","entry_speed_kph","exit_speed_kph",
            "min_speed_kph","max_speed_kph","mean_speed_kph","var_speed",
            "full_throttle_ratio","brake_ratio"
        ]

        results = []
        for target in ["sector_time_sec","delta_time_sec"]:
            df = data.dropna(subset=feature_cols+[target, "driver"]).copy()
            if df.empty:
                continue
            X = df[feature_cols].values; y = df[target].values; groups = df["driver"].values
            cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
            from math import sqrt
            maes, rmses, r2s = [], [], []
            model = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=42)
            for tr, te in cv.split(X, y, groups):
                model.fit(X[tr], y[tr])
                pred = model.predict(X[te])
                maes.append(mean_absolute_error(y[te], pred))
                rmses.append(np.sqrt(mean_squared_error(y[te], pred)))
                r2s.append(r2_score(y[te], pred))
            results.append({"target": target, "MAE_mean": float(np.mean(maes)), "RMSE_mean": float(np.mean(rmses)), "R2_mean": float(np.mean(r2s))})
            # Fit on all for importances
            model.fit(X, y)
            perm = permutation_importance(model, X, y, n_repeats=15, random_state=42, n_jobs=-1)
            imp_df = pd.DataFrame({"feature": feature_cols, "importance": perm.importances_mean}).sort_values("importance", ascending=False)
            if target == "sector_time_sec":
                imp_time = imp_df
                model_time = model
            else:
                imp_delta = imp_df
                model_delta = model
        metrics = pd.DataFrame(results)
    return metrics, imp_time, imp_delta, locals().get('model_time', None), locals().get('model_delta', None), data


# ============================ Extra visuals ============================

def save_importance_plot(imp_df, title, out_path):
    try:
        import matplotlib.pyplot as plt
        if imp_df is None or imp_df.empty: return None
        top = imp_df.sort_values("importance", ascending=True)
        plt.figure(figsize=(8,5))
        plt.barh(top["feature"], top["importance"])
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
        return str(out_path)
    except Exception as e:
        print("Failed to plot importances:", e)
        return None

def save_pdp_plot(model, X_df, feature_name, title, out_path):
    try:
        # Try sklearn's partial_dependence (older) or PartialDependenceDisplay (newer)
        try:
            from sklearn.inspection import partial_dependence
            import numpy as np
            import matplotlib.pyplot as plt
            if model is None or X_df is None or feature_name not in X_df.columns: return None
            fi = list(X_df.columns).index(feature_name)
            pdp = partial_dependence(model, X_df.values, [fi])
            xs = pdp["values"][0]; ys = pdp["average"][0]
            plt.figure(figsize=(6,4))
            plt.plot(xs, ys)
            plt.title(title)
            plt.xlabel(feature_name); plt.ylabel("Prediction")
            plt.tight_layout()
            plt.savefig(out_path, dpi=160); plt.close()
            return str(out_path)
        except Exception:
            # Newer API
            from sklearn.inspection import PartialDependenceDisplay
            import matplotlib.pyplot as plt
            if model is None or X_df is None or feature_name not in X_df.columns: return None
            fig = plt.figure(figsize=(6,4))
            ax = plt.gca()
            PartialDependenceDisplay.from_estimator(model, X_df, [feature_name], ax=ax)
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(out_path, dpi=160); plt.close(fig)
            return str(out_path)
    except Exception as e:
        print("Failed to plot PDP:", e)
        return None

def build_top_losses_table(deltas_df):
    if deltas_df is None or deltas_df.empty: 
        return pd.DataFrame()
    # Keep only positive losses and aggregate by driver/sector (mean loss)
    df = deltas_df.copy()
    df = df[df["delta_time_sec"] > 0]
    grp = df.groupby(["driver","sector"])["delta_time_sec"].mean().reset_index()
    return grp.sort_values("delta_time_sec", ascending=False).head(15)

# ============================ Report builders ============================

def build_unified_html(out_dir: Path, title: str, images: list, coaching_lines: list, metrics: pd.DataFrame, imp_time: pd.DataFrame, imp_delta: pd.DataFrame, sample_feats: pd.DataFrame, imp_time_png=None, imp_delta_png=None, pdp_pngs=None, top_losses_tbl: pd.DataFrame=None):
    html = out_dir / "unified_report.html"
    parts = [f"<html><head><meta charset='utf-8'><title>{title}</title></head><body>",
             f"<h1>{title}</h1>"]

    # Plots
    for cap, img in images:
        if img and Path(img).exists():
            parts.append(f"<h2>{cap}</h2><img src='{Path(img).name}' style='max-width:100%;'>")

    # Coaching
    if coaching_lines:
        parts.append("<h2>Coaching</h2><ul>")
        for line in coaching_lines:
            parts.append(f"<li>{line}</li>")
        parts.append("</ul>")

    # ML Metrics & Importances
    if metrics is not None and not metrics.empty:
        parts.append("<h2>ML Metrics</h2>")
        parts.append(metrics.to_html(index=False))
    if imp_time is not None:
        parts.append("<h3>Feature importance — sector_time_sec</h3>")
        if imp_time_png and Path(imp_time_png).exists():
            parts.append(f"<img src='{Path(imp_time_png).name}' style='max-width:100%;'>")
        parts.append(imp_time.to_html(index=False))
    if imp_delta is not None:
        parts.append("<h3>Feature importance — delta_time_sec</h3>")
        if imp_delta_png and Path(imp_delta_png).exists():
            parts.append(f"<img src='{Path(imp_delta_png).name}' style='max-width:100%;'>")
        parts.append(imp_delta.to_html(index=False))

    # PDP
    if pdp_pngs:
        parts.append("<h2>Partial Dependence (PDP) — top features</h2>")
        for cap, img in pdp_pngs:
            if img and Path(img).exists():
                parts.append(f"<h3>{cap}</h3><img src='{Path(img).name}' style='max-width:100%;'>")

    # Top losses mini-summary
    if top_losses_tbl is not None and not top_losses_tbl.empty:
        parts.append("<h2>Top time losses by Driver & Sector</h2>")
        parts.append(top_losses_tbl.to_html(index=False))

    # Sample features
    if sample_feats is not None and not sample_feats.empty:
        parts.append("<h2>Sample sector features</h2>")
        parts.append(sample_feats.head(30).to_html(index=False))

    parts.append("</body></html>")
    with open(html, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return str(html)

def build_unified_pdf(out_dir: Path, title: str, images: list, coaching_lines: list, metrics: pd.DataFrame, imp_time: pd.DataFrame, imp_delta: pd.DataFrame):
    pdf_path = out_dir / "unified_report.pdf"
    with PdfPages(pdf_path) as pdf:
        # Cover
        fig = plt.figure(figsize=(8.27, 11.69)); plt.axis('off')
        plt.text(0.5, 0.9, title, ha='center', va='center', fontsize=20)
        pdf.savefig(fig); plt.close(fig)

        # Images
        for cap, img in images:
            if img and Path(img).exists():
                fig = plt.figure(figsize=(8.27, 11.69)); plt.title(cap); plt.axis('off')
                arr = plt.imread(img); plt.imshow(arr)
                pdf.savefig(fig); plt.close(fig)

        # Coaching
        if coaching_lines:
            fig = plt.figure(figsize=(8.27, 11.69)); plt.title("Coaching"); plt.axis('off')
            text = "\n".join(["• " + s for s in coaching_lines[:40]])
            plt.text(0.05, 0.95, text, ha='left', va='top', fontsize=9, wrap=True)
            pdf.savefig(fig); plt.close(fig)

        # Metrics/Importances
        if metrics is not None and not metrics.empty:
            fig = plt.figure(figsize=(8.27, 11.69)); plt.title("ML Metrics"); plt.axis('off')
            tbl = plt.table(cellText=metrics.values, colLabels=list(metrics.columns), loc='center')
            tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1,1.5)
            pdf.savefig(fig); plt.close(fig)
        for title2, imp in [("Feature importance — sector_time_sec", imp_time), ("Feature importance — delta_time_sec", imp_delta)]:
            if imp is not None:
                fig = plt.figure(figsize=(8.27, 11.69)); plt.title(title2); plt.axis('off')
                tbl = plt.table(cellText=imp.values, colLabels=list(imp.columns), loc='center')
                tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1,1.5)
                pdf.savefig(fig); plt.close(fig)
    return str(pdf_path)

# ============================ Main pipeline ============================

def main():
    ap = argparse.ArgumentParser(description="All-in-one iRacing pipeline: plots + coaching + ML + unified report.")
    ap.add_argument('--telemetry-root', type=str, default='telemetria', help='telemetria/<car>/<track>/*.csv')
    ap.add_argument('--result-root', type=str, default='result', help='result/<car>/<track>/')
    ap.add_argument('--car', type=str, default=None)
    ap.add_argument('--track', type=str, default=None)
    ap.add_argument('--report', choices=['html','pdf','both'], default='both')
    ap.add_argument('--brake-threshold', type=float, default=0.1)
    ap.add_argument('--smooth-window', type=int, default=25)
    ap.add_argument('--track-length-m', type=float, default=5793.0)
    ap.add_argument('--sectors-file', type=str, default='sectors.yaml', help='Путь к файлу с секторами (yaml/json)')
    args = ap.parse_args()

    tel_root = Path(args.telemetry_root); res_root = Path(args.result_root)
    if not tel_root.exists():
        print(f"Папка с телеметрией не найдена: {tel_root.resolve()}"); sys.exit(1)

    cars = sorted([p.name for p in tel_root.iterdir() if p.is_dir()])
    car = args.car if args.car in cars else None
    if car is None:
        print("Выберите машину из telemetria/:"); car = choose_from(cars, "Машина")
    car_dir = tel_root / car

    tracks = sorted([p.name for p in car_dir.iterdir() if p.is_dir()])
    track = args.track if args.track in tracks else None
    if track is None:
        print(f"Выберите трассу из telemetria/{car}/:"); track = choose_from(tracks, "Трасса")
    track_dir = car_dir / track

    csvs = sorted(glob.glob(str(track_dir / "*.csv")))
    if not csvs:
        print("Нет CSV в", track_dir); sys.exit(1)

    # Output roots
    out_base = res_root / car / track
    out_plot = out_base  # keep plots here
    out_coach = out_base / "sector_coaching"; out_coach.mkdir(parents=True, exist_ok=True)
    out_ml = out_base  # will save feats & metrics here
    out_unified = out_base / "combined_report"; out_unified.mkdir(parents=True, exist_ok=True)

    # ---------------- Plots ----------------
    # Load and meta
    files_info = []
    for p in csvs:
        driver, lap_time_sec = parse_meta_from_filename(p)
        data = load_file(p)
        files_info.append({'path': p, 'driver': driver, 'lap_time_sec': lap_time_sec, 'data': data})
    summary = pd.DataFrame([{'file': Path(fi['path']).name, 'driver': fi['driver'], 'lap_time_sec': fi['lap_time_sec']} for fi in files_info])
    summary.to_csv(out_plot / 'summary.csv', index=False)

    sectors = load_sectors_from_file(Path(args.sectors_file), track) or pick_sectors(track)

    img_traj = plot_trajectories(files_info, out_plot)
    img_heat = plot_speed_heatmap_fastest(files_info, out_plot)
    img_brake = plot_braking_points(files_info, out_plot, args.brake_threshold)
    img_spd = plot_speed_profile(files_info, out_plot, args.smooth_window, sectors=sectors)
    img_delta = plot_delta_speed(files_info, out_plot, args.smooth_window, sectors=sectors)

    # ---------------- Coaching ----------------
    ref, deltas_df, coaching_lines = coaching_from_csvs(csvs, track, args.sectors_file, track_length_m=args.track_length_m, brake_threshold=args.brake_threshold)
    deltas_df.to_csv(out_coach / "sector_deltas.csv", index=False, float_format="%.4f")
    with open(out_coach / "coaching_report.txt", "w", encoding="utf-8") as f:
        f.write("Быстрейший круг — ориентир вычислен автоматически по времени в имени файла.\n\n")
        for line in coaching_lines: f.write(line + "\n")

    # ---------------- ML per-sector ----------------
    # Build sector features
    rows = []
    for p in csvs:
        driver, lap_time = parse_meta_from_filename(p)
        lap = load_lap(p)
        feats = sector_features_for_lap(lap, lap_time, sectors, track_length_m=args.track_length_m)
        for f in feats:
            row = {"driver": driver, "file": Path(p).name, **f}
            rows.append(row)
    feat_df = pd.DataFrame(rows)
    # compute delta target
    ref_best = feat_df.groupby("sector")["sector_time_sec"].min().rename("sector_time_best").reset_index()
    feat_df = feat_df.merge(ref_best, on="sector", how="left")
    feat_df["delta_time_sec"] = feat_df["sector_time_sec"] - feat_df["sector_time_best"]
    feat_df.to_csv(out_ml / "sector_features.csv", index=False)

    metrics, imp_time, imp_delta, model_time, model_delta, _ = train_ml_sector(feat_df, out_ml)
    if metrics is not None:
        metrics.to_csv(out_ml / "ml_sector_metrics.csv", index=False)
    if imp_time is not None:
        imp_time.to_csv(out_ml / "ml_feature_importance_sector_time_sec.csv", index=False)
    if imp_delta is not None:
        imp_delta.to_csv(out_ml / "ml_feature_importance_delta_time_sec.csv", index=False)

    # Create importance plots
    imp_time_png = save_importance_plot(imp_time, "Feature importance — sector_time_sec", out_unified/"imp_sector_time.png") if imp_time is not None else None
    imp_delta_png = save_importance_plot(imp_delta, "Feature importance — delta_time_sec", out_unified/"imp_delta_time.png") if imp_delta is not None else None

    # PDP for top 2 features of each target (if available)
    pdp_pngs = []
    try:
        import pandas as _pd
        # Use the same feature set as in training
        feats_cols = ["brake_start_m","throttle_on_m","entry_speed_kph","exit_speed_kph","min_speed_kph","max_speed_kph","mean_speed_kph","var_speed","full_throttle_ratio","brake_ratio"]
        X_df = feat_df.dropna(subset=feats_cols).reset_index(drop=True)
        X_df = X_df[feats_cols]
        if imp_time is not None and not imp_time.empty and 'model_time' in locals() and model_time is not None and not X_df.empty:
            for feat in imp_time.head(2)['feature'].tolist():
                png = save_pdp_plot(model_time, X_df, feat, f"PDP — sector_time_sec vs {feat}", out_unified/f"pdp_time_{feat}.png")
                if png: pdp_pngs.append((f"PDP (sector_time_sec) — {feat}", png))
        if imp_delta is not None and not imp_delta.empty and 'model_delta' in locals() and model_delta is not None and not X_df.empty:
            for feat in imp_delta.head(2)['feature'].tolist():
                png = save_pdp_plot(model_delta, X_df, feat, f"PDP — delta_time_sec vs {feat}", out_unified/f"pdp_delta_{feat}.png")
                if png: pdp_pngs.append((f"PDP (delta_time_sec) — {feat}", png))
    except Exception as e:
        print("PDP skipped:", e)

    # Top losses mini-summary
    top_losses_tbl = build_top_losses_table(deltas_df)

    # ---------------- Unified report ----------------
    images = [("Trajectories", img_traj),
              ("Speed (fastest)", img_heat),
              ("Braking points", img_brake),
              ("Speed profile", img_spd),
              ("Delta speed vs fastest", img_delta)]
    images = [im for im in images if im[1] is not None]

    html = build_unified_html(out_unified, f"iRacing — {car} — {track}",
                              images, coaching_lines, metrics, imp_time, imp_delta, feat_df,
                              imp_time_png=imp_time_png, imp_delta_png=imp_delta_png, pdp_pngs=pdp_pngs, top_losses_tbl=top_losses_tbl)
    if args.report in ("pdf","both"):
        pdf = build_unified_pdf(out_unified, f"iRacing — {car} — {track}",
                                images, coaching_lines, metrics, imp_time, imp_delta)
        print("PDF report:", pdf)
    print("HTML report:", html)
    print("Готово. База результатов:", out_base.resolve())

if __name__ == '__main__':
    main()
