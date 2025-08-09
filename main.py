"""
Streamlit dashboard: Consumi & Produzione FV – 12 mesi (Feb 2024 → Gen 2025)
============================================================================

Prerequisiti:
    pip install streamlit pandas plotly numpy

Esecuzione locale (Windows):
    streamlit run main.py
"""

from pathlib import Path
from datetime import time, timedelta
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# Config Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Energia 12 mesi",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Utility: ARERA bands (F1/F2/F3) e festività italiane
# -----------------------------------------------------------------------------
def _easter_sunday(year: int) -> pd.Timestamp:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19*a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2*e + 2*i - h - k) % 7
    m = (a + 11*h + 22*l) // 451
    month = (h + l - 7*m + 114) // 31
    day = ((h + l - 7*m + 114) % 31) + 1
    return pd.Timestamp(year=int(year), month=int(month), day=int(day))

def _italy_holidays(year: int) -> set:
    fixed = [
        pd.Timestamp(year, 1, 1),   # Capodanno
        pd.Timestamp(year, 1, 6),   # Epifania
        pd.Timestamp(year, 4, 25),  # Liberazione
        pd.Timestamp(year, 5, 1),   # Lavoro
        pd.Timestamp(year, 6, 2),   # Repubblica
        pd.Timestamp(year, 8, 15),  # Ferragosto
        pd.Timestamp(year, 11, 1),  # Tutti i Santi
        pd.Timestamp(year, 12, 8),  # Immacolata
        pd.Timestamp(year, 12, 25), # Natale
        pd.Timestamp(year, 12, 26), # Santo Stefano
    ]
    easter = _easter_sunday(year)
    easter_monday = easter + pd.Timedelta(days=1)
    return set(d.date() for d in fixed + [easter_monday])

def compute_arera_band(index_tzaware: pd.DatetimeIndex) -> pd.Series:
    """
    Restituisce Serie 'Band' con F1/F2/F3 per indice Europe/Rome.
      - Domenica e festivi: F3 tutto il giorno.
      - Sabato: F2 07–23 (ore 7..22).
      - Lun–Ven (non festivi): F1 08–19 (ore 8..18); F2 07 e 19–23 (ore 7 e 19..22); F3 resto.
    """
    if index_tzaware.tz is None:
        raise ValueError("L'indice deve essere timezone-aware (Europe/Rome).")
    years = sorted(set(index_tzaware.year))
    hol = set()
    for y in years:
        hol |= _italy_holidays(int(y))

    dow = pd.Series(index_tzaware.weekday, index=index_tzaware)  # 0=Mon..6=Sun
    hour = pd.Series(index_tzaware.hour, index=index_tzaware)
    is_holiday = pd.Series([ts.date() in hol for ts in index_tzaware], index=index_tzaware, dtype=bool)

    bands = pd.Series('F3', index=index_tzaware, dtype=object)  # default F3
    # Sabato
    sat_mask = (dow == 5) & (~is_holiday)
    bands.loc[sat_mask & hour.between(7, 22)] = 'F2'
    # Lun–Ven (non festivi)
    mf_mask = dow.between(0, 4) & (~is_holiday)
    bands.loc[mf_mask & hour.between(8, 18)] = 'F1'
    bands.loc[mf_mask & ((hour == 7) | hour.between(19, 22))] = 'F2'
    return bands

# -----------------------------------------------------------------------------
# Caricamento CSV: upload → file in repo → GitHub raw
# -----------------------------------------------------------------------------
DEFAULT_NAME = "combined_Feb2024_to_Jan2025_hourly_F123_withNetGrid_lower.csv"
REPO_DIR = Path(__file__).resolve().parent
DEFAULT_PATH = REPO_DIR / DEFAULT_NAME

# URL raw corretto al CSV nella repo GitHub (modifica se il file non è in root)
RAW_URL = (
    "https://raw.githubusercontent.com/tommasomalaguti/fv-energy-simulation/main/"
    + DEFAULT_NAME
)

@st.cache_data(show_spinner=True)
def load_data_auto(uploaded_file=None) -> pd.DataFrame:
    # 1) File caricato dall’utente
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # 2) File locale nella repo
        try:
            df = pd.read_csv(DEFAULT_PATH)
            st.caption(f"Fonte dati: file nella repo → {DEFAULT_NAME}")
        except Exception:
            # 3) Fallback: URL raw GitHub
            st.caption("Fonte dati: GitHub raw (fallback)")
            df = pd.read_csv(RAW_URL)

    # Normalizzazione
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce").dt.tz_convert("Europe/Rome")
    df = df.dropna(subset=["Datetime"]).set_index("Datetime").sort_index()

    if "Export_KWh" not in df.columns and "Export_kWh" in df.columns:
        df = df.rename(columns={"Export_kWh": "Export_KWh"})
    if "NetGrid_KWh" not in df.columns and "NetGrid_kWh" in df.columns:
        df = df.rename(columns={"NetGrid_kWh": "NetGrid_KWh"})
    if "NetGrid_KWh" not in df.columns and {"Total", "Autocons_kWh"}.issubset(df.columns):
        df["NetGrid_KWh"] = (df["Total"] - df["Autocons_kWh"]).clip(lower=0)
    if "Band" not in df.columns:
        df["Band"] = compute_arera_band(df.index)

    return df

# -----------------------------------------------------------------------------
# Simulazione batteria (carica solo da surplus FV)
# -----------------------------------------------------------------------------
def simulate_battery(df,
                     capacity_kwh=5.0,
                     p_charge_kw=3.0,
                     p_discharge_kw=3.0,
                     eta_ch=0.95,
                     eta_dis=0.95,
                     soc_min_frac=0.05,
                     soc_init_frac=0.50):
    """
    Modello semplice a passi orari.
    df deve avere: 'Total', 'PV_kWh'
    Ritorna df con colonne *_batt (post-batteria).
    """
    out = df.copy()
    load = out["Total"].astype(float).values
    pv   = out["PV_kWh"].astype(float).values

    n = len(out)
    soc = np.zeros(n, dtype=float)
    soc_max = capacity_kwh
    soc_min = soc_min_frac * capacity_kwh
    soc0    = soc_init_frac * capacity_kwh

    pv_direct = np.zeros(n)          # FV usata direttamente
    pv_surplus = np.zeros(n)         # FV eccedente dopo uso diretto
    batt_charge_in = np.zeros(n)     # energia che FINISCE in batteria (dopo efficienza di carica)
    batt_discharge_out = np.zeros(n) # energia dalla batteria ai carichi (lato AC, già con efficienza di scarica)
    import_grid = np.zeros(n)
    export_grid = np.zeros(n)

    soc_prev = np.clip(soc0, soc_min, soc_max)

    for t in range(n):
        L = max(load[t], 0.0)
        E = max(pv[t], 0.0)

        # 1) FV → carichi
        pv_direct[t] = min(L, E)
        rem_load = L - pv_direct[t]
        pv_surplus[t] = E - pv_direct[t]

        # 2) Carica batteria (solo da surplus FV)
        headroom = max(soc_max - soc_prev, 0.0)
        charge_batt_side_max = min(headroom, p_charge_kw)  # kWh in 1h
        charge_batt_side = min(charge_batt_side_max, pv_surplus[t] * eta_ch)
        batt_charge_in[t] = charge_batt_side

        # FV lato AC consumata per caricare
        pv_used_for_charge_ac = charge_batt_side / eta_ch if eta_ch > 0 else 0.0
        pv_surplus[t] -= min(pv_surplus[t], pv_used_for_charge_ac)

        soc_tmp = soc_prev + batt_charge_in[t]

        # 3) Scarica per coprire carico residuo
        available_batt_side = max(soc_tmp - soc_min, 0.0)
        deliverable_ac = min(rem_load, available_batt_side * eta_dis, p_discharge_kw * eta_dis)
        discharge_batt_side = deliverable_ac / eta_dis if eta_dis > 0 else 0.0
        batt_discharge_out[t] = deliverable_ac

        soc_new = soc_tmp - discharge_batt_side

        # 4) Rete ed export
        rem_after_batt = rem_load - batt_discharge_out[t]
        import_grid[t] = max(rem_after_batt, 0.0)
        export_grid[t] = max(pv_surplus[t], 0.0)

        soc[t] = np.clip(soc_new, soc_min, soc_max)
        soc_prev = soc[t]

    # Alias coerenti
    out["Batt_SOC_kWh"] = soc
    out["Batt_Charge_kWh"] = batt_charge_in
    out["Batt_Discharge_kWh"] = batt_discharge_out
    out["Batt_Discharge_KWh"] = out["Batt_Discharge_kWh"]  # compatibilità

    out["PV_Direct_kWh"] = pv_direct
    out["PV_Surplus_KWh"] = pv_surplus
    out["NetGrid_KWh_batt"] = import_grid
    out["Export_KWh_batt"] = export_grid
    out["Autocons_kWh_batt"] = out["PV_Direct_kWh"] + out["Batt_Discharge_KWh"]
    return out

# -----------------------------------------------------------------------------
# EV charging helpers (modalità per sessione)
# -----------------------------------------------------------------------------
DAYS: List[str] = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]
DAY_IDX: Dict[str, int] = {name: i for i, name in enumerate(DAYS)}

def build_ev_profile(index: pd.DatetimeIndex,
                     mode: str = "Orario fisso",
                     power_kw: float = 3.0,
                     energy_session_kwh: float = 10.0,
                     efficiency_ac_to_batt: float = 0.92,
                     start_time: time = time(22, 0),
                     window_h: int = 8,
                     days_sel: Optional[List[str]] = None,
                     pv_series: Optional[pd.Series] = None) -> pd.Series:
    """
    Serie oraria (kWh/h lato AC) di ricarica EV.
    - 'Orario fisso': riempie la finestra in ordine.
    - 'PV-priority': riempie prima le ore con PV più alto nella finestra.
    """
    if days_sel is None:
        days_sel = ["Lun", "Mer", "Ven"]
    tz = index.tz
    ev = pd.Series(0.0, index=index)

    ac_energy_needed = energy_session_kwh / max(efficiency_ac_to_batt, 1e-9)
    if ac_energy_needed <= 0 or power_kw <= 0 or window_h <= 0:
        return ev

    all_days = pd.to_datetime(index.tz_convert(tz).date).unique()
    sel_idx = {DAY_IDX[d] for d in days_sel if d in DAY_IDX}

    for d in all_days:
        d = pd.Timestamp(d).tz_localize(tz)
        if d.weekday() not in sel_idx:
            continue

        start = d + timedelta(hours=start_time.hour, minutes=start_time.minute)
        end = start + timedelta(hours=window_h)
        mask = (index >= start) & (index < end)
        if not mask.any():
            continue

        remaining = ac_energy_needed

        if mode == "PV-priority" and pv_series is not None:
            order = pv_series.loc[mask].sort_values(ascending=False).index
        else:
            order = index[mask]

        for ts in order:
            if remaining <= 0:
                break
            put = min(power_kw, remaining)
            ev.loc[ts] += put
            remaining -= put

    return ev

def split_ev_coverage(df_base: pd.DataFrame,
                      df_with_batt: Optional[pd.DataFrame],
                      use_batt: bool) -> Dict[str, float]:
    """
    Ripartisce EV_kWh tra FV diretto, Batteria e Rete.
    """
    pv = df_base["PV_kWh"]
    total_base = df_base["Total_base"]
    total_with_ev = df_base["Total"]
    ev = df_base["EV_kWh"]

    pv_used_with_ev  = np.minimum(pv, total_with_ev)
    pv_used_baseonly = np.minimum(pv, total_base)
    ev_from_pv = (pv_used_with_ev - pv_used_baseonly).clip(lower=0)

    ev_rem_after_pv = (ev - ev_from_pv).clip(lower=0)

    if use_batt and df_with_batt is not None and "Batt_Discharge_kWh" in df_with_batt.columns:
        batt_dis = df_with_batt["Batt_Discharge_kWh"]
        base_rem_after_pv = (total_base - pv_used_baseonly).clip(lower=0)
        denom = base_rem_after_pv + ev_rem_after_pv
        share = np.divide(ev_rem_after_pv, denom, out=np.zeros_like(ev_rem_after_pv), where=denom > 0)
        ev_from_batt = batt_dis * share
    else:
        ev_from_batt = pd.Series(0.0, index=df_base.index)

    ev_from_grid = (ev - ev_from_pv - ev_from_batt).clip(lower=0)

    return {
        "pv": float(ev_from_pv.sum()),
        "batt": float(ev_from_batt.sum()),
        "grid": float(ev_from_grid.sum()),
        "total": float(ev.sum())
    }

# -----------------------------------------------------------------------------
# EV planner basato su km/giorno (modalità intelligente)
# -----------------------------------------------------------------------------
def _km_dict_to_array(km_per_day_dict: dict) -> np.ndarray:
    order = ["Lun","Mar","Mer","Gio","Ven","Sab","Dom"]
    return np.array([float(km_per_day_dict.get(d, 0.0)) for d in order], dtype=float)

def _is_plugged(ts: pd.Timestamp,
                km_w: np.ndarray,
                overnight_start: int,
                overnight_end: int) -> bool:
    """
    Regole: 
      - Giorno senza viaggio (km==0): sempre collegata (tutto il giorno).
      - Giorno con viaggio (km>0): collegata dalle 00:00 fino a overnight_end (prima della partenza).
      - Sera del giorno precedente a un viaggio: collegata da overnight_start a 24:00.
    """
    wd = ts.weekday()
    wd_next = (ts + pd.Timedelta(days=1)).weekday()

    if km_w[wd] == 0.0:
        return True
    if km_w[wd] > 0.0 and ts.hour < overnight_end:
        return True
    if km_w[wd_next] > 0.0 and ts.hour >= overnight_start:
        return True
    return False

def _departures(index: pd.DatetimeIndex,
                km_w: np.ndarray,
                depart_hour: int,
                depart_min: int) -> List[pd.Timestamp]:
    tz = index.tz
    days = pd.to_datetime(index.tz_convert(tz).date).unique()
    deps: List[pd.Timestamp] = []
    for d in days:
        d = pd.Timestamp(d).tz_localize(tz)
        if km_w[d.weekday()] > 0.0:
            t = d + pd.Timedelta(hours=depart_hour, minutes=depart_min)
            if index[0] < t <= index[-1]:
                deps.append(t)
    return deps

def plan_ev_from_km(df_hour: pd.DataFrame,
                    km_per_day: dict,
                    veh_eff_kwh_per_100km: float = 16.0,
                    batt_capacity_kwh: float = 60.0,
                    soc_init_frac: float = 0.6,
                    soc_min_frac: float = 0.1,
                    charger_power_kw: float = 7.4,
                    depart_time: time = time(7,0),
                    overnight_start_h: int = 20,
                    overnight_end_h: int = 7) -> pd.Series:
    """
    Genera Serie oraria 'EV_kWh' pianificata dai km/giorno.
    Obiettivo: prima di ogni partenza avere SOC >= (kWh_viaggio + riserva),
    caricando nelle ore con PV più alto dove l'auto è collegata.
    """
    idx = df_hour.index
    tz = idx.tz
    km_w = _km_dict_to_array(km_per_day)
    eff_kwh_km = veh_eff_kwh_per_100km / 100.0  # kWh per km

    # Lista partenze
    deps = _departures(idx, km_w, depart_time.hour, depart_time.minute)
    if not deps:
        return pd.Series(0.0, index=idx)

    ev = pd.Series(0.0, index=idx, dtype=float)

    # Stato di carica EV (kWh) all'avvio
    soc = soc_init_frac * batt_capacity_kwh
    soc_min = soc_min_frac * batt_capacity_kwh

    prev_dep = idx[0]

    for dep in deps:
        need_kwh = km_w[dep.weekday()] * eff_kwh_km
        target_before = min(need_kwh + soc_min, batt_capacity_kwh)
        deficit = max(target_before - soc, 0.0)

        if deficit > 0:
            # Segmento precedente la partenza, solo ore "collegate"
            mask_seg = (idx > prev_dep) & (idx < dep)
            if mask_seg.any():
                cand = idx[mask_seg & idx.map(lambda t: _is_plugged(t, km_w, overnight_start_h, overnight_end_h))]
                if len(cand) > 0:
                    pv = df_hour.loc[cand, "PV_kWh"].fillna(0.0)
                    order = pv.sort_values(ascending=False).index  # PV-priority

                    remaining = deficit
                    for ts in order:
                        if remaining <= 0:
                            break
                        put = min(charger_power_kw, remaining)
                        ev.at[ts] += put
                        remaining -= put

                    charged = deficit - max(remaining, 0.0)
                    soc = min(soc + charged, batt_capacity_kwh)

        # Partenza: consumo viaggio
        soc = max(soc - need_kwh, 0.0)
        prev_dep = dep

    return ev

# -----------------------------------------------------------------------------
# Sidebar – upload e preset batteria
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Impostazioni")
uploaded = st.sidebar.file_uploader("Carica un CSV…", type=["csv"])

try:
    df = load_data_auto(uploaded_file=uploaded)
except Exception as e:
    st.error(f"Errore nel caricamento del CSV: {e}")
    st.stop()

if df.empty:
    st.error("⚠️ Il DataFrame è vuoto.")
    st.stop()

# -----------------------------------------------------------------------------
# Filtri
# -----------------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.header("Filtri dati")

if "Band" not in df.columns:
    df["Band"] = compute_arera_band(df.index)

bands = sorted(df["Band"].dropna().unique().tolist())
selected_bands = st.sidebar.multiselect("Fasce ARERA:", bands, default=bands)

min_date, max_date = df.index.min().date(), df.index.max().date()
start_date, end_date = st.sidebar.date_input(
    "Intervallo date:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

mask = (
    (df.index.date >= start_date)
    & (df.index.date <= end_date)
    & (df["Band"].isin(selected_bands))
)
df_filt = df.loc[mask].copy()

# -----------------------------------------------------------------------------
# Simulazione batteria – controlli UI (con preset)
# -----------------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.header("🔋 Batteria (simulazione)")

PRESETS = {
    "Residenziale 5 kWh": dict(cap=5.0,  p_ch=3.0, p_dis=3.0, eta_ch=0.95, eta_dis=0.95, soc_min=0.05, soc_init=0.50),
    "Residenziale 10 kWh":dict(cap=10.0, p_ch=4.0, p_dis=4.0, eta_ch=0.96, eta_dis=0.96, soc_min=0.05, soc_init=0.50),
    "PMI 15 kWh":         dict(cap=15.0, p_ch=5.0, p_dis=5.0, eta_ch=0.96, eta_dis=0.96, soc_min=0.07, soc_init=0.50),
}
with st.sidebar.expander("Preset batteria", expanded=False):
    preset_name = st.selectbox("Scegli preset", list(PRESETS.keys()) + ["Personalizzato"])
    if preset_name != "Personalizzato" and st.button("Applica preset", use_container_width=True):
        for k, v in PRESETS[preset_name].items():
            st.session_state[k] = v
        st.experimental_rerun()

use_batt = st.sidebar.checkbox("Attiva batteria", value=False, key="use_batt")
cap     = st.sidebar.number_input("Capacità (kWh)",      min_value=0.0, value=st.session_state.get("cap", 5.0),  step=0.5, key="cap")
p_ch    = st.sidebar.number_input("Potenza carica (kW)", min_value=0.0, value=st.session_state.get("p_ch", 3.0), step=0.5, key="p_ch")
p_dis   = st.sidebar.number_input("Potenza scarica (kW)",min_value=0.0, value=st.session_state.get("p_dis",3.0), step=0.5, key="p_dis")
eta_ch  = st.sidebar.slider("Efficienza carica ηch",  min_value=0.80, max_value=1.00, value=st.session_state.get("eta_ch",0.95), step=0.01, key="eta_ch")
eta_dis = st.sidebar.slider("Efficienza scarica ηdis",min_value=0.80, max_value=1.00, value=st.session_state.get("eta_dis",0.95), step=0.01, key="eta_dis")
soc_min = st.sidebar.slider("SOC minimo (%)", min_value=0, max_value=50, value=int(100*st.session_state.get("soc_min",0.05)), step=1, key="soc_min_pct") / 100.0
soc_init= st.sidebar.slider("SOC iniziale (%)", min_value=0, max_value=100, value=int(100*st.session_state.get("soc_init",0.50)), step=5, key="soc_init_pct") / 100.0

# -----------------------------------------------------------------------------
# Preparazione dati (fix duplicati → griglia oraria → Band)
# -----------------------------------------------------------------------------
n_dup = int(df_filt.index.duplicated().sum())
if n_dup > 0:
    st.info(f"🔧 Risolti {n_dup} duplicati orari nell'intervallo selezionato.")

# 1) Collassa duplicati (somma solo colonne numeriche kWh/ora)
num_cols = df_filt.select_dtypes(include="number").columns
df_no_dups = (
    df_filt
    .sort_index()
    .groupby(level=0)[num_cols]
    .sum()
)

# 2) Griglia oraria continua
df_hour = df_no_dups.resample("1h").sum(min_count=1)

# 3) Ricalcola Band sull’indice nuovo
df_hour["Band"] = compute_arera_band(df_hour.index)

# 4) Riempi eventuali buchi
for c in ["Total", "PV_kWh", "Autocons_kWh"]:
    if c in df_hour.columns:
        df_hour[c] = df_hour[c].fillna(0)

# -----------------------------------------------------------------------------
# 🚗 Auto elettrica – modalità semplice o planner per km
# -----------------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.header("🚗 Ricarica auto elettrica")

ev_mode_ui = st.sidebar.radio("Modalità EV", ["Per sessione (semplice)", "Pianificatore settimanale (km)"], index=1)
use_ev = st.sidebar.checkbox("Attiva ricarica EV", value=False)

# salva carico base
df_hour["Total_base"] = df_hour.get("Total", 0).copy()

if use_ev and ev_mode_ui == "Per sessione (semplice)":
    ev_mode = st.sidebar.selectbox("Logica sessione", ["Orario fisso", "PV-priority"], index=1)
    ev_power = st.sidebar.number_input("Potenza caricatore (kW)", 0.5, 22.0, 3.7, 0.1)
    ev_energy = st.sidebar.number_input("Energia per sessione (kWh nel pacco EV)", 1.0, 120.0, 10.0, 0.5)
    ev_eff = st.sidebar.slider("Efficienza AC→batt EV", 0.70, 1.00, 0.92, 0.01)
    ev_start = st.sidebar.time_input("Ora inizio finestra", value=time(22, 0))
    ev_window = st.sidebar.slider("Durata finestra (ore)", 1, 12, 8)
    ev_days = st.sidebar.multiselect("Giorni con sessione", DAYS, default=["Lun", "Mer", "Ven"])

    ev_profile = build_ev_profile(
        index=df_hour.index, mode=ev_mode, power_kw=ev_power,
        energy_session_kwh=ev_energy, efficiency_ac_to_batt=ev_eff,
        start_time=ev_start, window_h=int(ev_window), days_sel=ev_days,
        pv_series=df_hour.get("PV_kWh", None)
    )
    df_hour["EV_kWh"] = ev_profile

elif use_ev and ev_mode_ui == "Pianificatore settimanale (km)":
    st.sidebar.markdown("Imposta i km per giorno e i parametri veicolo/caricatore.")
    c1, c2 = st.sidebar.columns(2)
    km_lun = c1.number_input("Lun (km)", 0, 1000, 140, 10)
    km_mar = c2.number_input("Mar (km)", 0, 1000, 140, 10)
    km_mer = c1.number_input("Mer (km)", 0, 1000, 140, 10)
    km_gio = c2.number_input("Gio (km)", 0, 1000, 0, 10)
    km_ven = c1.number_input("Ven (km)", 0, 1000, 0, 10)
    km_sab = c2.number_input("Sab (km)", 0, 1000, 0, 10)
    km_dom = c1.number_input("Dom (km)", 0, 1000, 0, 10)

    veh_eff = st.sidebar.number_input("Consumo EV (kWh/100 km)", 10.0, 30.0, 16.0, 0.5)
    batt_cap = st.sidebar.number_input("Capacità batteria EV (kWh)", 10.0, 120.0, 60.0, 1.0)
    soc0 = st.sidebar.slider("SOC iniziale EV (%)", 0, 100, 60) / 100.0
    soc_res = st.sidebar.slider("RISERVA minima EV (%)", 0, 50, 10) / 100.0
    p_chg = st.sidebar.number_input("Potenza caricatore (kW)", 0.5, 22.0, 7.4, 0.1)
    dep_time = st.sidebar.time_input("Ora partenza giorni con viaggio", value=time(7, 0))
    ovn_start = st.sidebar.slider("Overnight: da (h)", 17, 23, 20)
    ovn_end = st.sidebar.slider("Overnight: a (h)", 5, 10, 7)

    km_week = {"Lun": km_lun, "Mar": km_mar, "Mer": km_mer, "Gio": km_gio, "Ven": km_ven, "Sab": km_sab, "Dom": km_dom}

    ev_profile = plan_ev_from_km(
        df_hour=df_hour,
        km_per_day=km_week,
        veh_eff_kwh_per_100km=veh_eff,
        batt_capacity_kwh=batt_cap,
        soc_init_frac=soc0,
        soc_min_frac=soc_res,
        charger_power_kw=p_chg,
        depart_time=dep_time,
        overnight_start_h=int(ovn_start),
        overnight_end_h=int(ovn_end)
    )
    df_hour["EV_kWh"] = ev_profile

else:
    df_hour["EV_kWh"] = 0.0

# Applica EV al carico e ricalcola grandezze senza batteria
df_hour["Total"] = df_hour["Total_base"] + df_hour["EV_kWh"]
df_hour["Autocons_kWh"] = np.minimum(df_hour["PV_kWh"], df_hour["Total"])
df_hour["NetGrid_KWh"]  = (df_hour["Total"] - df_hour["Autocons_kWh"]).clip(lower=0)
df_hour["Export_KWh"]   = (df_hour["PV_kWh"] - df_hour["Autocons_kWh"]).clip(lower=0)

# Export/NetGrid naming robusto
export_col = "Export_KWh" if "Export_KWh" in df_hour.columns else ("Export_kWh" if "Export_KWh" in df_hour.columns else None)
if export_col is None:
    df_hour["Export_KWh"] = (df_hour["PV_kWh"] - df_hour["Autocons_kWh"]).clip(lower=0)
    export_col = "Export_KWh"

net_col = "NetGrid_KWh" if "NetGrid_KWh" in df_hour.columns else ("NetGrid_kWh" if "NetGrid_KWh" in df_hour.columns else None)
if net_col is None and {"Total","Autocons_kWh"}.issubset(df_hour.columns):
    df_hour["NetGrid_KWh"] = (df_hour["Total"] - df_hour["Autocons_kWh"]).clip(lower=0)
    net_col = "NetGrid_KWh"

# -----------------------------------------------------------------------------
# Scenario attivo: con o senza batteria
# -----------------------------------------------------------------------------
if use_batt:
    sim = simulate_battery(
        df_hour,
        capacity_kwh=cap,
        p_charge_kw=p_ch,
        p_discharge_kw=p_dis,
        eta_ch=eta_ch,
        eta_dis=eta_dis,
        soc_min_frac=soc_min,
        soc_init_frac=soc_init
    )
    AUTOCONS = "Autocons_kWh_batt"
    NET      = "NetGrid_KWh_batt"
    EXPORT   = "Export_KWh_batt"
    df_use   = sim
else:
    AUTOCONS = "Autocons_kWh"
    NET      = net_col
    EXPORT   = export_col
    df_use   = df_hour

# Normalizza/alias nomi batteria per evitare KeyError a valle
if "Batt_Discharge_kWh" in df_use.columns and "Batt_Discharge_KWh" not in df_use.columns:
    df_use["Batt_Discharge_KWh"] = df_use["Batt_Discharge_kWh"]
if "Batt_Discharge_KWh" in df_use.columns and "Batt_Discharge_kWh" not in df_use.columns:
    df_use["Batt_Discharge_kWh"] = df_use["Batt_Discharge_KWh"]

# -----------------------------------------------------------------------------
# KPI principali
# -----------------------------------------------------------------------------
st.subheader("KPI")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Prelievo rete (kWh)", f"{df_use[NET].sum():,.0f}")
col2.metric("Produzione FV (kWh)", f"{df_use['PV_kWh'].sum():,.0f}")
col3.metric("Autoconsumo (kWh)", f"{df_use[AUTOCONS].sum():,.0f}")
col4.metric("Export (kWh)", f"{df_use[EXPORT].sum():,.0f}")
autonomy = (df_use[AUTOCONS].sum() / df_use["Total"].sum() * 100) if df_use["Total"].sum() else 0
col5.metric("Autonomia (%)", f"{autonomy:.1f}%")

# -----------------------------------------------------------------------------
# KPI "Contributo batteria" (vs baseline senza batteria)
# -----------------------------------------------------------------------------
baseline = df_hour.copy()
if "NetGrid_KWh" not in baseline.columns and {"Total","Autocons_kWh"}.issubset(baseline.columns):
    baseline["NetGrid_KWh"] = (baseline["Total"] - baseline["Autocons_kWh"]).clip(lower=0)

prelievo_baseline = baseline["NetGrid_KWh"].sum() if "NetGrid_KWh" in baseline.columns else 0.0
autocons_baseline = baseline["Autocons_kWh"].sum() if "Autocons_kWh" in baseline.columns else 0.0

prelievo_scenario = df_use["NetGrid_KWh_batt"].sum() if use_batt else prelievo_baseline
autocons_scenario = df_use[AUTOCONS].sum()

kwh_prelievo_ev = prelievo_baseline - prelievo_scenario
kwh_autocons_add = autocons_scenario - autocons_baseline

c1, c2 = st.columns(2)
c1.metric("Prelievo evitato (kWh)", f"{kwh_prelievo_ev:,.0f}")
c2.metric("Autoconsumo FV aggiuntivo (kWh)", f"{kwh_autocons_add:,.0f}")

# -----------------------------------------------------------------------------
# KPI ricarica EV – copertura da PV/Batteria/Rete
# -----------------------------------------------------------------------------
if use_ev:
    cover = split_ev_coverage(df_hour, df_use if use_batt else None, use_batt=use_batt)
    ev_tot = cover["total"]
    with st.expander("🔎 Ricarica EV – ripartizione fonti", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Energia EV totale (kWh)", f"{ev_tot:,.0f}")
        c2.metric("Coperta da FV diretto (kWh)", f"{cover['pv']:,.0f}")
        c3.metric("Coperta da Batteria (kWh)", f"{cover['batt']:,.0f}")
        c4.metric("Dalla Rete (kWh)", f"{cover['grid']:,.0f}")

        ev_df = pd.DataFrame({
            "Fonte": ["FV diretto", "Batteria", "Rete"],
            "kWh": [cover["pv"], cover["batt"], cover["grid"]]
        })
        fig_ev = px.bar(ev_df, x="Fonte", y="kWh", text_auto=".0f", title="Copertura ricarica EV")
        fig_ev.update_layout(yaxis_title="kWh", xaxis_title="")
        st.plotly_chart(fig_ev, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------------
# Andamento giornaliero – Carichi vs Produzione FV
# -----------------------------------------------------------------------------
st.subheader("Andamento giornaliero – Carichi vs Produzione FV")
daily = df_use[["Total", "PV_kWh"]].resample("D").sum()
if use_ev and "EV_kWh" in df_hour.columns:
    daily["EV_kWh"] = df_hour["EV_kWh"].resample("D").sum()

y_cols = ["Total", "PV_kWh"] + (["EV_kWh"] if "EV_kWh" in daily.columns else [])
fig_daily = px.area(
    daily,
    x=daily.index,
    y=y_cols,
    labels={"value": "kWh al giorno", "variable": ""},
)
fig_daily.update_layout(legend_orientation="h", legend_y=-0.2)
st.plotly_chart(fig_daily, use_container_width=True)

# -----------------------------------------------------------------------------
# Grafico SOC batteria
# -----------------------------------------------------------------------------
if use_batt and "Batt_SOC_kWh" in df_use.columns:
    st.subheader("Andamento SOC batteria")
    fig_soc = px.line(
        df_use[["Batt_SOC_kWh"]],
        y="Batt_SOC_kWh",
        labels={"Batt_SOC_kWh": "SOC (kWh)", "index": "Data/Ora"},
    )
    st.plotly_chart(fig_soc, use_container_width=True)

# -----------------------------------------------------------------------------
# Bilancio mensile energia
# -----------------------------------------------------------------------------
st.subheader("Bilancio mensile energia")
monthly = df_use[[AUTOCONS, NET, EXPORT]].resample("MS").sum()
fig_month = go.Figure()
fig_month.add_bar(x=monthly.index.strftime("%Y-%m"), y=monthly[AUTOCONS], name="Autoconsumo")
fig_month.add_bar(x=monthly.index.strftime("%Y-%m"), y=monthly[NET], name="Prelievo rete")
fig_month.add_bar(x=monthly.index.strftime("%Y-%m"), y=monthly[EXPORT], name="Export")
fig_month.update_layout(barmode="stack", yaxis_title="kWh", xaxis_title="Mese")
st.plotly_chart(fig_month, use_container_width=True)

# -----------------------------------------------------------------------------
# Heat-map autoconsumo (ora × settimana ISO)
# -----------------------------------------------------------------------------
st.subheader("Heat-map autoconsumo (ora × settimana ISO)")
week_series = df_use.index.isocalendar().week
pivot = (
    df_use.assign(week=week_series)
          .pivot_table(index=df_use.index.hour, columns="week", values=AUTOCONS, aggfunc="sum")
)
fig_heat = px.imshow(
    pivot,
    labels=dict(x="Settimana ISO", y="Ora", color="kWh autocons"),
    aspect="auto",
)
fig_heat.update_yaxes(dtick=1)
st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# =============================================================================
# 💶 BLOCCO ECONOMICO (con fix formato € e KPI "solo rete")
# =============================================================================
st.sidebar.header("💶 Tariffe")
price_f1 = st.sidebar.number_input("Prezzo F1 (€/kWh)", min_value=0.0, value=0.30, step=0.01, format="%.3f")
price_f2 = st.sidebar.number_input("Prezzo F2 (€/kWh)", min_value=0.0, value=0.26, step=0.01, format="%.3f")
price_f3 = st.sidebar.number_input("Prezzo F3 (€/kWh)", min_value=0.0, value=0.24, step=0.01, format="%.3f")
credit_export = st.sidebar.number_input("Credito export (€/kWh)", min_value=0.0, value=0.10, step=0.01, format="%.3f")
use_export_credit = st.sidebar.checkbox("Considera credito export", value=True)
st.sidebar.caption("Nota: stima semplificata. Quote fisse, potenza impegnata, oneri/IVA non inclusi.")

def _fmt_eur(x: float) -> str:
    """Formatta importi stile italiano: '€ 1.234,56'."""
    s = f"{x:,.2f}"           # es. 1,234,567.89
    s = s.replace(",", "X")   # swap separatori
    s = s.replace(".", ",")   # -> 1,234,567,89
    s = s.replace("X", ".")   # -> 1.234.567,89
    return f"€ {s}"

def compute_costs(df_in: pd.DataFrame, import_col: str, export_col: str):
    if "Band" not in df_in.columns:
        df_in = df_in.copy()
        df_in["Band"] = compute_arera_band(df_in.index)
    price_map = {"F1": price_f1, "F2": price_f2, "F3": price_f3}
    prices = df_in["Band"].map(price_map).fillna(price_f3)

    tmp = df_in[[import_col, export_col, "Band"]].copy()
    tmp["price"] = prices
    tmp["import_cost"] = tmp[import_col] * tmp["price"]
    tmp["export_rev"] = tmp[export_col] * (credit_export if use_export_credit else 0.0)
    tmp["net_cost"] = tmp["import_cost"] - tmp["export_rev"]

    total_import_cost = float(tmp["import_cost"].sum())
    total_export_rev  = float(tmp["export_rev"].sum())
    total_net_cost    = float(tmp["net_cost"].sum())
    monthly_econ = tmp.resample("MS")[["import_cost", "export_rev", "net_cost"]].sum()
    return total_import_cost, total_export_rev, total_net_cost, monthly_econ

# Baseline (senza batteria)
if "NetGrid_KWh" not in baseline.columns and {"Total","Autocons_kWh"}.issubset(baseline.columns):
    baseline["NetGrid_KWh"] = (baseline["Total"] - baseline["Autocons_kWh"]).clip(lower=0)
if "Export_KWh" not in baseline.columns and "Export_kWh" in baseline.columns:
    baseline = baseline.rename(columns={"Export_kWh": "Export_KWh"})

imp_b, rev_b, net_b, mon_b = compute_costs(baseline, "NetGrid_KWh", "Export_KWh")

# Scenario attivo (con o senza batteria)
imp_s, rev_s, net_s, mon_s = compute_costs(
    df_use,
    "NetGrid_KWh_batt" if use_batt else "NetGrid_KWh",
    "Export_KWh_batt"  if use_batt else "Export_KWh",
)

# KPI economici principali
st.subheader("KPI economici")
e1, e2, e3 = st.columns(3)
e1.metric("Spesa baseline (€/anno)", _fmt_eur(net_b))
e2.metric("Spesa scenario (€/anno)", _fmt_eur(net_s))
e3.metric("Risparmio (€/anno)", _fmt_eur(net_b - net_s))

# 🔹 KPI aggiuntivo: costo se tutta l'energia fosse prelevata dalla rete (no FV)
all_grid_df = df_hour.copy()
all_grid_df["Import_ALLGRID"] = all_grid_df["Total"]
all_grid_df["Export_ALLGRID"] = 0.0
_, _, net_all, _ = compute_costs(all_grid_df, "Import_ALLGRID", "Export_ALLGRID")
st.caption(f"Costo se tutta l'energia fosse prelevata dalla rete (no FV): {_fmt_eur(net_all)}")

# Grafico economico mensile
econ = pd.DataFrame({
    "Baseline": mon_b["net_cost"],
    "Scenario": mon_s["net_cost"],
})
econ.index.name = "Mese"
fig_econ = go.Figure()
fig_econ.add_bar(x=econ.index.strftime("%Y-%m"), y=econ["Baseline"], name="Baseline")
fig_econ.add_bar(x=econ.index.strftime("%Y-%m"), y=econ["Scenario"], name="Scenario")
fig_econ.update_layout(barmode="group", yaxis_title="€ / mese", xaxis_title="Mese", title="Costo netto mensile (energia)")
st.plotly_chart(fig_econ, use_container_width=True)

# Download CSV riepilogo economico mensile
csv_econ = econ.copy()
csv_econ["Risparmio"] = csv_econ["Baseline"] - csv_econ["Scenario"]
st.download_button(
    "💾 Scarica CSV economico mensile",
    data=csv_econ.to_csv(index=True).encode(),
    file_name="riepilogo_economico_mensile.csv",
    mime="text/csv",
)

# =============================================================================
# 📈 RIENTRO INVESTIMENTO (ROI / Payback / NPV / IRR)
# =============================================================================
st.divider()
st.subheader("📈 Rientro investimento")

# --- Parametri finanziari (sidebar) ---
st.sidebar.header("💸 Parametri investimento")
eval_mode = st.sidebar.selectbox(
    "Valutazione investimento",
    ["PV + Batteria vs Solo Rete", "Solo Batteria vs PV esistente"]
)
capex_pv = st.sidebar.number_input("CAPEX FV (€)", min_value=0, value=7000, step=100)
capex_batt = st.sidebar.number_input("CAPEX Batteria (€)", min_value=0, value=4000, step=100)
om_pct = st.sidebar.number_input("O&M annuo (% CAPEX considerato)", min_value=0.0, max_value=10.0, value=1.0, step=0.1) / 100.0

# Incentivo semplice: % del CAPEX, ripartito su 'incent_years' anni (es. detrazione)
incent_pct = st.sidebar.number_input("Incentivo % su CAPEX (rateizzato)", min_value=0.0, max_value=100.0, value=0.0, step=1.0) / 100.0
incent_years = st.sidebar.number_input("Anni incentivo", min_value=1, max_value=20, value=10, step=1)

# Sostituzione batteria (opzionale)
batt_repl_year = st.sidebar.number_input("Sostituzione batteria (anno, 0=mai)", min_value=0, max_value=30, value=0, step=1)
batt_repl_cost = st.sidebar.number_input("Costo sostituzione batteria (€)", min_value=0, max_value=50000, value=0, step=100)

# Proiezione
years = st.sidebar.number_input("Orizzonte (anni)", min_value=1, max_value=30, value=20, step=1)
disc = st.sidebar.number_input("Tasso di sconto %", min_value=0.0, max_value=20.0, value=5.0, step=0.5) / 100.0
esc = st.sidebar.number_input("Crescita prezzo energia % (annua)", min_value=0.0, max_value=30.0, value=0.0, step=0.5) / 100.0
salvage_pct = st.sidebar.number_input("Valore residuo a fine vita (% CAPEX considerato)", min_value=0.0, max_value=100.0, value=0.0, step=1.0) / 100.0

# --- Risparmio annuo base dal blocco economico ---
if eval_mode == "PV + Batteria vs Solo Rete":
    saving0 = max(net_all - net_s, 0.0)
    capex_considered = capex_pv + capex_batt
else:
    saving0 = max(net_b - net_s, 0.0)
    capex_considered = capex_batt

om_year = om_pct * capex_considered
incent_annual = capex_considered * incent_pct / max(incent_years, 1)

initial_cf = -capex_considered
years_idx = np.arange(1, years + 1, 1)

cash_year = []
cash_year_disc = []
parts = {"saving": [], "incent": [], "om": [], "repl": []}
cum = []
cum_disc = []

def _npv(rate, cfs):
    return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cfs))

def _irr(cfs, lo=-0.99, hi=1.5, tol=1e-6, it=200):
    def f(r): return _npv(r, cfs)
    a, b = lo, hi
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        return None
    for _ in range(it):
        m = (a + b) / 2
        fm = f(m)
        if abs(fm) < tol:
            return m
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return (a + b) / 2

cum_val = initial_cf
cum_disc_val = initial_cf

for t in years_idx:
    saving_t = saving0 * ((1 + esc) ** (t - 1))
    incent_t = incent_annual if t <= incent_years else 0.0
    om_t = om_year
    repl_t = (-batt_repl_cost) if (batt_repl_year > 0 and t == batt_repl_year) else 0.0
    salvage_t = salvage_pct * capex_considered if t == years else 0.0

    cf_t = saving_t + incent_t - om_t + repl_t + salvage_t

    parts["saving"].append(saving_t)
    parts["incent"].append(incent_t)
    parts["om"].append(-om_t)
    parts["repl"].append(repl_t)

    cash_year.append(cf_t)
    cash_year_disc.append(cf_t / ((1 + disc) ** t))

    cum_val += cf_t
    cum_disc_val += cf_t / ((1 + disc) ** t)
    cum.append(cum_val)
    cum_disc.append(cum_disc_val)

npv_val = sum(cash_year_disc) + initial_cf
irr_val = _irr([initial_cf] + cash_year)

try:
    cum_simple = np.cumsum([initial_cf] + cash_year)
    simple_pb = next(i for i in range(1, len(cum_simple)) if cum_simple[i] >= 0)
except StopIteration:
    simple_pb = None

try:
    cum_disc_series = np.cumsum([initial_cf] + cash_year_disc)
    disc_pb = next(i for i in range(1, len(cum_disc_series)) if cum_disc_series[i] >= 0)
except StopIteration:
    disc_pb = None

# --- KPI rientro investimento ---
k1, k2, k3, k4 = st.columns(4)
k1.metric("NPV (valore attuale netto)", _fmt_eur(npv_val))
k2.metric("IRR (tasso interno di rendimento)", f"{irr_val*100:.1f}%" if irr_val is not None else "n/d")
k3.metric("Payback semplice", f"{simple_pb} anni" if simple_pb else "n/d")
k4.metric("Payback scontato", f"{disc_pb} anni" if disc_pb else "n/d")

# Tabella dettaglio
roi_df = pd.DataFrame({
    "Anno": years_idx,
    "Risparmio €": parts["saving"],
    "Incentivi €": parts["incent"],
    "O&M €": parts["om"],
    "Sostituzioni €": parts["repl"],
    "Flusso netto €": cash_year,
    "Cum €": cum,
    "Cum scontato €": cum_disc,
})
st.dataframe(roi_df.style.format({
    "Risparmio €": _fmt_eur, "Incentivi €": _fmt_eur, "O&M €": _fmt_eur,
    "Sostituzioni €": _fmt_eur, "Flusso netto €": _fmt_eur,
    "Cum €": _fmt_eur, "Cum scontato €": _fmt_eur,
}), use_container_width=True)

st.download_button(
    "💾 Scarica cashflow ROI (CSV)",
    data=roi_df.to_csv(index=False).encode(),
    file_name="roi_cashflow.csv",
    mime="text/csv",
)

# Grafico: cumulato scontato
fig_roi = go.Figure()
fig_roi.add_scatter(x=years_idx, y=cum_disc, mode="lines+markers", name="Cum scontato")
fig_roi.add_hline(y=0, line_dash="dot")
fig_roi.update_layout(title="Cumulato scontato vs anni", xaxis_title="Anno", yaxis_title="€")
st.plotly_chart(fig_roi, use_container_width=True)

# -----------------------------------------------------------------------------
# Tabella dati + download CSV scenario corrente
# -----------------------------------------------------------------------------
st.divider()
st.subheader("Anteprima dati (scenario corrente)")
st.dataframe(df_use.head(500))
st.download_button(
    "💾 Scarica CSV (scenario corrente)",
    data=df_use.to_csv(index=True).encode(),
    file_name=("consumi_pv_con_batteria.csv" if use_batt else "consumi_pv_senza_batteria.csv"),
    mime="text/csv",
)
