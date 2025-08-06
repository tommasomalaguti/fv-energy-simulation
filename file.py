# ============================================================
#  COMBINED CONSUMPTION + PV   (Feb 2024 → Jan 2025)
# ============================================================

import pandas as pd, numpy as np, re
from pathlib import Path

# ---------- CONFIG ---------- #
TZ = "Europe/Rome"

CONSUMPTION_FILE = "C:/Users/TommasoMalaguti/Desktop/fv_sim_project/combined_2024_consumption_pv.csv"
PV_2023_FILE     = "C:/Users/TommasoMalaguti/Desktop/fv_sim_project/Timeseries_44.852_11.207_SA3_6kWp_crystSi_14_39deg_0deg_2023_2023.csv"
OUTPUT_FILE      = "C:/Users/TommasoMalaguti/Desktop/fv_sim_project/combined_Feb2024_Jan2025.csv"

START = pd.Timestamp("2024-02-01 00:00", tz=TZ)
END   = pd.Timestamp("2025-01-31 23:00", tz=TZ)
# ----------------------------- #

# === 1) CONSUMI =============================================================
cons = pd.read_csv(CONSUMPTION_FILE, index_col=0, parse_dates=True)
cons.index = pd.to_datetime(cons.index, errors="coerce")
cons = cons[cons.index.notna()]
cons.index = cons.index.tz_localize(TZ) if cons.index.tz is None else cons.index.tz_convert(TZ)
cons = cons.loc[START:END].copy()
cons = cons.drop(columns=['PV_kWh', 'PV_KWh'], errors='ignore')  # elimina vecchia produzione

# === 2) FUNZIONE LETTURA PVGIS (generica) ====================================
def load_pvgis_generic(path: str | Path, tz="Europe/Rome") -> pd.DataFrame:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if re.match(r"\s*time", line, flags=re.I):
                header_line, sample = i, line
                break
        else:
            raise ValueError("Header con 'time' non trovato")

    sep = next((s for s in ("\t", ";", ",") if s in sample), ",")
    df = pd.read_csv(path, sep=sep, header=header_line, comment="#",
                     engine="python", skip_blank_lines=True)

    ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).set_index(ts_col)
    df.index = df.index.tz_localize(tz) if df.index.tz is None else df.index.tz_convert(tz)

    num_cols = df.select_dtypes("number").columns
    pv_col = next((c for c in num_cols if re.search(r"kwh", c, re.I)), num_cols[0])
    pv = df[[pv_col]].rename(columns={pv_col: "PV_kWh"})

    if pv.index.to_series().diff().dt.total_seconds().median() < 3600:
        pv = pv.resample("h").sum()

    return pv

pv_hour = load_pvgis_generic(PV_2023_FILE, tz=TZ)

# === 3) CLONA PV 2024 + GEN 2025 (riempi 29 feb) =============================
pv_2024  = pv_hour.shift(1, freq="Y").loc["2024-02-01":"2024-12-31 23:00"]
pv_jan25 = pv_hour.shift(2, freq="Y").loc["2025-01-01":"2025-01-31 23:00"]

full_idx = pd.date_range("2024-02-01 00:00", "2024-12-31 23:00", freq="h", tz=TZ)
pv_2024 = pv_2024.reindex(full_idx)
pv_2024["PV_kWh"] = pv_2024["PV_kWh"].fillna(pv_2024["PV_kWh"].shift(24))

pv_all = pd.concat([pv_2024, pv_jan25])

# === 4) MERGE & GAP FILL =====================================================
df = cons.join(pv_all, how="inner")

def interp_small(s): return s.interpolate(limit=2, limit_area="inside")
df["Total"]  = interp_small(df["Total"])
df["PV_kWh"] = interp_small(df["PV_kWh"])
df = df.dropna(subset=["Total", "PV_kWh"])   # <<-- nome corretto

# === 5) KPI ORARI ============================================================
df["Autocons_kWh"] = np.minimum(df["PV_kWh"], df["Total"])
df["Export_kWh"]   = np.maximum(df["PV_kWh"] - df["Total"], 0)
df["NetGrid_kWh"]  = df["Total"] - df["Autocons_kWh"]

def assign_band(ts):
    wd, h = ts.weekday(), ts.hour
    if wd < 5 and 8 <= h < 19:
        return "F1"
    if (wd < 5 and ((7 <= h < 8) or (19 <= h < 23))) or (wd == 5 and 7 <= h < 23):
        return "F2"
    return "F3"

df["Band"] = [assign_band(t) for t in df.index]

# === 6) SAVE OUTPUT ==========================================================
Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE)
print(f"\n✅  File creato: {OUTPUT_FILE}")

# === 7) RIEPILOGHI ===========================================================
summary_band = df.groupby("Band")[["Total","PV_kWh","Autocons_kWh",
                                   "Export_kWh","NetGrid_kWh"]].sum().round(1)

monthly_total = df.resample("M")[["Total","PV_kWh","Autocons_kWh",
                                  "Export_kWh","NetGrid_kWh"]].sum().round(1)
monthly_total.index = monthly_total.index.strftime("%Y-%m")

print("\n=== Totali per Fascia (kWh) ===")
print(summary_band)

print("\n=== Totali Mensili (kWh) ===")
print(monthly_total.head(13))
