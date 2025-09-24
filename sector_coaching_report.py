
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob, argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

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

def load_lap(path: str):
    df = read_csv_auto(path)
    col_dist = 'LapDistPct' if 'LapDistPct' in df.columns else detect_column(df.columns, ['lapdistpct','dist_pct','distancepercentage','splinepos','spline_pos'])
    col_speed = 'Speed' if 'Speed' in df.columns else detect_column(df.columns, ['speed_kph','speed','velocity'])
    col_brake = 'Brake' if 'Brake' in df.columns else detect_column(df.columns, ['brake'])
    col_throttle = 'Throttle' if 'Throttle' in df.columns else detect_column(df.columns, ['throttle'])
    d = {}
    def col(c):
        s = pd.to_numeric(df[c], errors='coerce') if (c and c in df.columns) else pd.Series(dtype=float)
        return s.values
    d['dist'] = col(col_dist)
    d['speed'] = col(col_speed)
    d['brake'] = col(col_brake)
    d['throttle'] = col(col_throttle)
    return d


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

def moving_average(x, w):
    if len(x)==0: return x
    w = max(1, int(w))
    return pd.Series(x).rolling(w, min_periods=1, center=True).mean().values

def first_cross_up(series, thresh, a_idx, b_idx):
    s = series[a_idx:b_idx+1]
    if len(s) < 2: return None
    prev = s[:-1] < thresh
    now = s[1:] >= thresh
    idx = np.where(prev & now)[0]
    if idx.size>0:
        return a_idx + idx[0] + 1
    return None

def last_cross_up_after_min(speed, throttle, thresh, a_idx, b_idx):
    s = speed[a_idx:b_idx+1]
    t = throttle[a_idx:b_idx+1]
    if len(s)==0: return None
    min_idx_local = int(np.nanargmin(s))
    start = a_idx + min_idx_local
    for i in range(start, b_idx+1):
        if np.isfinite(t[i-a_idx]) and t[i-a_idx] >= thresh:
            return i
    return None

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

def analyze(files, track_length_m=5793.0, brake_threshold=0.1, throttle_threshold=0.5, smooth_w=25):
    laps = []
    for p in files:
        driver, laptime = parse_meta_from_filename(p)
        d = load_lap(p)
        dist = d['dist']; spd = d['speed']; brk = d['brake']; thr = d['throttle']
        if dist.size==0 or spd.size==0:
            continue
        dist = np.clip(dist, 0, 1)
        spd_s = moving_average(spd, smooth_w)
        thr_s = moving_average(thr, max(3, smooth_w//3)) if thr.size else thr
        brk_s = moving_average(brk, max(3, smooth_w//3)) if brk.size else brk
        laps.append({"path": p, "driver": driver, "lap_time": laptime, "dist": dist, "speed": spd_s, "brake": brk_s, "throttle": thr_s})
    if not laps:
        raise RuntimeError("No usable laps.")
    laps_with_times = [L for L in laps if L["lap_time"] is not None]
    fastest = min(laps_with_times, key=lambda L: L["lap_time"]) if laps_with_times else laps[0]

    sectors = load_sectors_from_file(Path(args.sectors_file), track) or monza_named_sectors()
    ref = {}
    for name, a, b in sectors:
        dist = fastest["dist"]
        idx_a = int(np.searchsorted(dist, a, side="left"))
        idx_b = int(np.searchsorted(dist, b, side="right")) - 1
        idx_a = max(0, min(idx_a, len(dist)-1))
        idx_b = max(0, min(idx_b, len(dist)-1))
        if idx_b <= idx_a: idx_a, idx_b = 0, len(dist)-1
        t_sec = integrate_sector_time(fastest["dist"], fastest["speed"], fastest["lap_time"] or np.nan, a, b)
        bp = first_cross_up(fastest["brake"], 0.1, idx_a, idx_b)
        tp = last_cross_up_after_min(fastest["speed"], fastest["throttle"], 0.5, idx_a, idx_b)
        vmin = np.nanmin(fastest["speed"][idx_a:idx_b+1]) if idx_b>idx_a else np.nan
        ref[name] = {"t": t_sec, "bp": bp, "tp": tp, "vmin": vmin}

    rows = []
    suggestions = []
    for L in laps:
        if L is fastest:
            continue
        for name, a, b in sectors:
            dist = L["dist"]
            idx_a = int(np.searchsorted(dist, a, side="left"))
            idx_b = int(np.searchsorted(dist, b, side="right")) - 1
            idx_a = max(0, min(idx_a, len(dist)-1))
            idx_b = max(0, min(idx_b, len(dist)-1))
            if idx_b <= idx_a: idx_a, idx_b = 0, len(dist)-1

            t_sec = integrate_sector_time(L["dist"], L["speed"], L["lap_time"] or np.nan, a, b)
            t_ref = ref[name]["t"]
            if not np.isfinite(t_sec) or not np.isfinite(t_ref):
                continue
            delta_t = t_sec - t_ref

            bp = first_cross_up(L["brake"], 0.1, idx_a, idx_b)
            tp = last_cross_up_after_min(L["speed"], L["throttle"], 0.5, idx_a, idx_b)
            vmin = np.nanmin(L["speed"][idx_a:idx_b+1]) if idx_b>idx_a else np.nan

            def idx_to_m(idx):
                if idx is None or idx<0 or idx>=len(L["dist"]): return np.nan
                return float(L["dist"][idx]) * track_length_m

            bp_m = idx_to_m(bp); tp_m = idx_to_m(tp)
            bp_ref_m = idx_to_m(ref[name]["bp"]); tp_ref_m = idx_to_m(ref[name]["tp"])
            vmin_ref = ref[name]["vmin"]

            d_bp = bp_m - bp_ref_m if np.isfinite(bp_m) and np.isfinite(bp_ref_m) else np.nan
            d_tp = tp_m - tp_ref_m if np.isfinite(tp_m) and np.isfinite(tp_ref_m) else np.nan
            d_vmin = (vmin - vmin_ref) if np.isfinite(vmin) and np.isfinite(vmin_ref) else np.nan

            rows.append({
                "driver": L["driver"], "sector": name, "delta_time_sec": delta_t,
                "delta_brake_m": d_bp, "delta_throttle_m": d_tp, "delta_min_speed_kph": d_vmin
            })

            reasons = []
            if np.isfinite(d_bp) and d_bp < -5:
                reasons.append("рано начал тормозить")
            elif np.isfinite(d_bp) and d_bp > 5:
                reasons.append("поздно начал тормозить")
            if np.isfinite(d_tp) and d_tp > 5:
                reasons.append("поздно открыл газ")
            elif np.isfinite(d_tp) and d_tp < -5:
                reasons.append("слишком рано открыл газ (возможна пробуксовка)")
            if np.isfinite(d_vmin) and d_vmin < -3:
                reasons.append("слишком низкая минимальная скорость в повороте")
            elif np.isfinite(d_vmin) and d_vmin > 3:
                reasons.append("высокая минимальная скорость (риск выноса)")

            if delta_t > 0.05:
                reason_txt = "; ".join(reasons) if reasons else "менее эффективная траектория/фазы торможения и газа"
                suggestions.append(f"Пилот {L['driver']} проигрывает {delta_t:.2f}с в секции {name}: {reason_txt}.")

    details_df = pd.DataFrame(rows)
    return fastest["driver"], details_df, suggestions

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

def main():
    ap = argparse.ArgumentParser(description="Sector coaching vs fastest lap (interactive car/track selection).")
    ap.add_argument("--telemetry-root", type=str, default="telemetria", help="telemetria/<car>/<track>/*.csv")
    ap.add_argument("--result-root", type=str, default="result", help="result/<car>/<track>/sector_coaching")
    ap.add_argument("--car", type=str, default=None)
    ap.add_argument("--track", type=str, default=None)
    ap.add_argument("--track-length-m", type=float, default=5793.0)
    ap.add_argument('--sectors-file', type=str, default='sectors.yaml', help='Путь к файлу с секторами (yaml/json)')
    args = ap.parse_args()

    tel_root = Path(args.telemetry_root)
    res_root = Path(args.result_root)
    if not tel_root.exists():
        print(f"Папка с телеметрией не найдена: {tel_root.resolve()}"); sys.exit(1)

    cars = sorted([p.name for p in tel_root.iterdir() if p.is_dir()])
    car = args.car if args.car in cars else None
    if car is None:
        print("Выберите машину из telemetria/:")
        car = choose_from(cars, "Машина")
    car_dir = tel_root / car

    tracks = sorted([p.name for p in car_dir.iterdir() if p.is_dir()])
    track = args.track if args.track in tracks else None
    if track is None:
        print(f"Выберите трассу из telemetria/{car}/:")
        track = choose_from(tracks, "Трасса")
    track_dir = car_dir / track

    csvs = sorted(glob.glob(str(track_dir / "*.csv")))
    if not csvs:
        print("Нет CSV в", track_dir); sys.exit(1)

    out_dir = res_root / car / track / "sector_coaching"
    out_dir.mkdir(parents=True, exist_ok=True)

    fastest_driver, details, suggestions = analyze(csvs, track_length_m=args.track_length_m)

    details_path = out_dir / "sector_deltas.csv"
    details.to_csv(details_path, index=False, float_format="%.4f")

    report_path = out_dir / "coaching_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Быстрейший круг: {fastest_driver}\n\n")
        for line in suggestions:
            f.write(line + "\n")

    print("Готово. Файлы сохранены в:", out_dir.resolve())
    print(" - deltas:", details_path)
    print(" - report:", report_path)

if __name__ == "__main__":
    main()
