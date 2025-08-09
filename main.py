"""
Streamlit dashboard: Consumi & Produzione FV – 12 mesi (Auto EV Optimizer)
==========================================================================

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

    for c in ["Total", "PV_kWh", "Autocons_kWh"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

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

    pv_direct = np.zeros(n)
    pv_surplus = np.zeros(n)
    batt_charge_in = np.zeros(n)
    batt_discharge_out = np.zeros(n)
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

    out["Batt_SOC_kWh"] = soc
    out["Batt_Charge_kWh"] = batt_charge_in
    out["Batt_Discharge_kWh"] = batt_discharge_out
    out["Batt_Discharge_KWh"] = out["Batt_Discharge_kWh"]

    out["PV_Direct_kWh"] = pv_direct
    out["PV_Surplus_KWh"] = pv_surplus
    out["NetGrid_KWh_batt"] = import_grid
    out["Export_KWh_batt"] = export_grid
    out["Autocons_kWh_batt"] = out["PV_Direct_kWh"] + out["Batt_Discharge_KWh"]
    return out

# -----------------------------------------------------------------------------
# EV – Auto Optimizer (PV-first) con look-ahead e multi-slot
# -----------------------------------------------------------------------------
DAYS: List[str] = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]
DAY_IDX: Dict[str, int] = {name: i for i, name in enumerate(DAYS)}

def _weekly_plug_mask_multi(slots_cfg: Dict[str, List[Dict]]) -> np.ndarray:
    """
    slots_cfg[day] = [{"active":bool, "start":time, "end":time}, ...]
    Ritorna [7 x 24] True=collegata, False=via. Gestisce slot oltre mezzanotte.
    """
    plug = np.ones((7,24), dtype=bool)
    for dname, slots in slots_cfg.items():
        wd = DAY_IDX[dname]
        for s in slots:
            if not s.get("active", False):
                continue
            h1 = s["start"].hour
            h2 = s["end"].hour
            if h1 == h2:
                plug[wd, :] = False
                continue
            if h1 < h2:
                plug[wd, h1:h2] = False
            else:
                plug[wd, h1:24] = False
                plug[(wd+1)%7, 0:h2] = False
    return plug

def _build_deadlines_multi(index: pd.DatetimeIndex,
                           slots_cfg: Dict[str, List[Dict]],
                           veh_eff_kwh_per_100km: float,
                           km_per_slot: Dict[str, List[float]]) -> List[Dict]:
    """
    Crea eventi 'partenza' per ogni slot attivo: (ts, need_kwh).
    km_per_slot[day] = [km_slot1, km_slot2, ...] (stessa lunghezza di slots_cfg[day]).
    """
    eff = veh_eff_kwh_per_100km / 100.0
    tz = index.tz
    days = pd.to_datetime(index.tz_convert(tz).date).unique()
    events = []
    for d in days:
        dt = pd.Timestamp(d).tz_localize(tz)
        dname = DAYS[dt.weekday()]
        slots = slots_cfg.get(dname, [])
        km_list = km_per_slot.get(dname, [0.0]*len(slots))
        for i, s in enumerate(slots):
            if not s.get("active", False):
                continue
            ts = dt + pd.Timedelta(hours=s["start"].hour, minutes=s["start"].minute)
            if index[0] < ts <= index[-1]:
                need = max(float(km_list[i]) * eff, 0.0)
                events.append({"ts": ts, "need": need})
    events.sort(key=lambda e: e["ts"])
    return events

def _is_plugged_from_mask(ts: pd.Timestamp, mask_week: np.ndarray) -> bool:
    return bool(mask_week[ts.weekday(), ts.hour])

def plan_ev_multideadline(df_hour: pd.DataFrame,
                          slots_cfg: Dict[str, List[Dict]],
                          km_per_slot: Dict[str, List[float]],
                          veh_eff_kwh_per_100km: float = 16.0,
                          batt_capacity_kwh: float = 60.0,
                          soc_init_frac: float = 0.6,
                          soc_min_frac: float = 0.1,
                          charger_power_kw: float = 7.4,
                          lookahead_days: int = 7,
                          prefer_pv_only: bool = True,
                          house_batt_headroom_kw: float = 0.0,
                          headroom_series: Optional[pd.Series] = None,
                          price_map: Optional[Dict[str, float]] = None) -> pd.Series:
    """
    Look-ahead multi-deadline (PV-first).
    In ogni segmento tra partenze carica per tempo fino a coprire TUTTE le uscite entro lookahead_days
    (entro capacità EV). PV prima; se non basta, usa le ore "rest" ordinate per prezzo se fornito.
    """
    idx = df_hour.index
    ev = pd.Series(0.0, index=idx, dtype=float)

    plug_mask = _weekly_plug_mask_multi(slots_cfg)
    deadlines = _build_deadlines_multi(idx, slots_cfg, veh_eff_kwh_per_100km, km_per_slot)
    if not deadlines:
        return ev

    if "Band" not in df_hour.columns:
        df_hour = df_hour.copy()
        df_hour["Band"] = compute_arera_band(df_hour.index)

    soc = soc_init_frac * batt_capacity_kwh
    soc_min = soc_min_frac * batt_capacity_kwh
    prev_t = idx[0]

    j = 0
    while j < len(deadlines):
        dep = deadlines[j]["ts"]
        need_now = deadlines[j]["need"]

        horizon_end = dep + pd.Timedelta(days=lookahead_days)
        cum_need = 0.0
        k = j
        while k < len(deadlines) and deadlines[k]["ts"] <= horizon_end:
            cum_need += deadlines[k]["need"]
            k += 1

        target_soc = min(cum_need + soc_min, batt_capacity_kwh)
        deficit_total = max(target_soc - soc, 0.0)

        seg_mask = (idx > prev_t) & (idx < dep)
        if seg_mask.any() and deficit_total > 1e-9:
            cand_idx = idx[seg_mask & idx.map(lambda t: _is_plugged_from_mask(t, plug_mask))]
            if len(cand_idx) > 0:
                cand["surplus_after_base"] = cand["PV_kWh"] - cand["Total_base"]
                # Quanta potenza realmente la batteria può prendere (da simulazione base-only)
                if headroom_series is not None:
                    batt_real = headroom_series.reindex(cand.index).fillna(0.0)
                else:
                    batt_real = pd.Series(0.0, index=cand.index)
                
                # Limita la rivendicazione batteria al surplus disponibile
                batt_claim = np.minimum(batt_real, np.maximum(cand["surplus_after_base"], 0.0))
                
                # --- URGENZA EV: calcola difficoltà di ricarica entro deadline ---
                plugged_hours = float(cand.shape[0])  # ore utili nel segmento
                if plugged_hours > 0:
                    kW_needed_avg = deficit_total / plugged_hours
                else:
                    kW_needed_avg = 0.0
                
                # Trasforma la difficoltà in riduzione priorità batteria
                urgency_raw = kW_needed_avg / max(charger_power_kw, 1e-6)  # 0..~1
                urgency_boost = float(np.clip(urgency_raw, 0.0, 1.0)) * urgency_max_relax
                
                # Priorità batteria effettiva
                prio_eff = float(np.clip(prio_pct * (1.0 - urgency_boost), 0.0, 1.0))
                
                # Quota batteria riservata
                batt_claim_eff = batt_claim * prio_eff
                
                # Surplus FV disponibile per EV
                cand["pv_surplus_ev"] = np.maximum(cand["surplus_after_base"] - batt_claim_eff, 0.0)
                
                # Ore "good" e "rest"
                good = cand[cand["pv_surplus_ev"] > 1e-9].sort_values("pv_surplus_ev", ascending=False)
                rest = cand.loc[cand.index.difference(good.index)]
                if price_map is not None and len(rest) > 0:
                    rest = rest.assign(price=rest["Band"].map(price_map).fillna(np.inf)) \
                               .sort_values(by=["price", "PV_kWh"], ascending=[True, False])
                else:
                    rest = rest.sort_values("PV_kWh", ascending=False)


                # 1) Solo surplus FV
                remaining = deficit_total
                for ts in good.index:
                    if remaining <= 0:
                        break
                    room = batt_capacity_kwh - soc
                    if room <= 1e-9:
                        break
                    put = min(charger_power_kw, remaining, room)
                    ev.at[ts] += put
                    soc = min(soc + put, batt_capacity_kwh)
                    remaining -= put

                # 2) Se non basta, usa ore rest (minimo indispensabile, ordinate per prezzo/PV)
                if remaining > 1e-9 and len(rest) > 0:
                    for ts in rest.index:
                        if remaining <= 0:
                            break
                        room = batt_capacity_kwh - soc
                        if room <= 1e-9:
                            break
                        put = min(charger_power_kw, remaining, room)
                        ev.at[ts] += put
                        soc = min(soc + put, batt_capacity_kwh)
                        remaining -= put

        # Partenza: consumo dello slot corrente
        soc = max(soc - need_now, 0.0)
        prev_t = dep
        j += 1

    return ev


def plan_ev_auto(df_hour: pd.DataFrame,
                 slots_cfg: dict,
                 km_per_slot: dict,
                 veh_eff_kwh_per_100km: float,
                 batt_capacity_kwh: float,
                 soc_init_frac: float,
                 soc_min_frac: float,
                 charger_power_kw: float,
                 use_batt: bool,
                 house_batt_p_charge_kw: float,
                 price_map: dict,
                 headroom_series: Optional[pd.Series] = None) -> pd.Series:
    """
    Ottimizzatore automatico:
      - Look-ahead = 7 giorni
      - PV-first rigido: prima usa solo ore con surplus FV al netto della batteria di casa
      - Se non basta entro la deadline: sblocca il minimo di ore restanti, ordinate per prezzo (F3→F2→F1)
    """
    headroom = house_batt_p_charge_kw if use_batt else 0.0

    return plan_ev_multideadline(
        df_hour=df_hour,
        slots_cfg=slots_cfg,
        km_per_slot=km_per_slot,
        veh_eff_kwh_per_100km=veh_eff_kwh_per_100km,
        batt_capacity_kwh=batt_capacity_kwh,
        soc_init_frac=soc_init_frac,
        soc_min_frac=soc_min_frac,
        charger_power_kw=charger_power_kw,
        lookahead_days=7,
        prefer_pv_only=True,
        house_batt_headroom_kw=headroom,
        headroom_series=headroom_series,
        price_map=price_map,
    )

# -----------------------------------------------------------------------------
# Sidebar – upload
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
# Filtri base
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
# Preparazione dati (fix duplicati → griglia oraria → Band)
# -----------------------------------------------------------------------------
n_dup = int(df_filt.index.duplicated().sum())
if n_dup > 0:
    st.info(f"🔧 Risolti {n_dup} duplicati orari nell'intervallo selezionato.")

# 1) Collassa duplicati (somma numeriche kWh/ora)
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
# 💶 Tariffe (usate anche per l'ordinamento ore non-FV nell'EV)
# -----------------------------------------------------------------------------
st.sidebar.header("💶 Tariffe")
price_f1 = st.sidebar.number_input("Prezzo F1 (€/kWh)", min_value=0.0, value=0.30, step=0.01, format="%.3f")
price_f2 = st.sidebar.number_input("Prezzo F2 (€/kWh)", min_value=0.0, value=0.26, step=0.01, format="%.3f")
price_f3 = st.sidebar.number_input("Prezzo F3 (€/kWh)", min_value=0.0, value=0.24, step=0.01, format="%.3f")
credit_export = st.sidebar.number_input("Credito export (€/kWh)", min_value=0.0, value=0.10, step=0.01, format="%.3f")
use_export_credit = st.sidebar.checkbox("Considera credito export", value=True)
st.sidebar.caption("Nota: stima semplificata. Quote fisse, potenza impegnata, oneri/IVA non inclusi.")
_price_map_ev = {"F1": price_f1, "F2": price_f2, "F3": price_f3}

# -----------------------------------------------------------------------------
# 🔋 Batteria (simulazione)
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
# 🚗 Ricarica EV (Auto)
# -----------------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.header("🚗 Ricarica EV (Auto)")
use_ev = st.sidebar.checkbox("Attiva ricarica EV (ottimizzata FV)", value=True)

# salva carico base prima dell'EV
df_hour["Total_base"] = df_hour.get("Total", 0).copy()

if use_ev:
    # Parametri veicolo/caricatore
    veh_eff = st.sidebar.number_input("Consumo EV (kWh/100 km)", 10.0, 30.0, 16.0, 0.5)
    batt_cap = st.sidebar.number_input("Capacità batteria EV (kWh)", 10.0, 120.0, 60.0, 1.0)
    soc0     = st.sidebar.slider("SOC iniziale EV (%)", 0, 100, 60) / 100.0
    soc_res  = st.sidebar.slider("RISERVA minima EV (%)", 0, 50, 10) / 100.0
    p_chg    = st.sidebar.number_input("Potenza caricatore (kW)", 0.5, 22.0, 7.4, 0.1)
    prio_pct = st.sidebar.slider("Priorità batteria vs EV (%)", 0, 100, 70) / 100.0
    urgency_max_relax = st.sidebar.slider("Urgenza EV: riduzione max priorità batteria (%)", 0, 100, 50) / 100.0
    urgency_sensitivity = st.sidebar.slider("Sensibilità urgenza (km-equivalenti / ora collegata)", 1.0, 20.0, 8.0)

    # Routine: fino a 2 slot "via" per giorno + km/slot
    slots_cfg = {d: [] for d in DAYS}
    km_per_slot = {d: [] for d in DAYS}

    st.sidebar.caption("Imposta quando l'auto è VIA (non collegata). Il resto del tempo è collegata e carica FV-first.")
    for d in DAYS:
        with st.sidebar.expander(d, expanded=(d in ["Lun","Ven","Sab","Dom"])):
            # Slot 1 (tipicamente Ufficio)
            s1_act = st.checkbox("Slot 1 attivo", key=f"{d}_s1_act", value=(d in ["Lun","Mar","Mer"]))
            s1_s   = st.time_input("Slot 1 inizio", key=f"{d}_s1_s", value=time(7,0))
            s1_e   = st.time_input("Slot 1 fine",   key=f"{d}_s1_e", value=time(18,0))
            s1_km  = st.number_input("KM slot 1", 0, 1000, 140 if d in ["Lun","Mar","Mer"] else 0, 10, key=f"{d}_s1_km")
            slots_cfg[d].append({"active": s1_act, "start": s1_s, "end": s1_e}); km_per_slot[d].append(float(s1_km))

            # Slot 2 (tipicamente Serale)
            s2_act = st.checkbox("Slot 2 attivo", key=f"{d}_s2_act", value=(d in ["Ven","Sab","Dom"]))
            s2_s   = st.time_input("Slot 2 inizio", key=f"{d}_s2_s", value=time(19,0))
            s2_e   = st.time_input("Slot 2 fine",   key=f"{d}_s2_e", value=time(23,59))
            s2_km  = st.number_input("KM slot 2", 0, 1000, 50 if d in ["Ven","Sab","Dom"] else 0, 10, key=f"{d}_s2_km")
            slots_cfg[d].append({"active": s2_act, "start": s2_s, "end": s2_e}); km_per_slot[d].append(float(s2_km))

    # Esegui ottimizzatore automatico
    # Headroom dinamico: simula la batteria di casa sul SOLO carico base (senza EV)
    if use_batt:
        df_base_only = df_hour.copy()
        df_base_only["Total"] = df_base_only["Total_base"]
        sim_base = simulate_battery(
            df_base_only,
            capacity_kwh=cap,
            p_charge_kw=p_ch,
            p_discharge_kw=p_dis,
            eta_ch=eta_ch,
            eta_dis=eta_dis,
            soc_min_frac=soc_min,
            soc_init_frac=soc_init
        )
        headroom_dyn = sim_base["Batt_Charge_kWh"].reindex(df_hour.index).fillna(0.0)
    else:
        headroom_dyn = None

    ev_profile = plan_ev_auto(
        df_hour=df_hour,
        slots_cfg=slots_cfg,
        km_per_slot=km_per_slot,
        veh_eff_kwh_per_100km=veh_eff,
        batt_capacity_kwh=batt_cap,
        soc_init_frac=soc0,
        soc_min_frac=soc_res,
        charger_power_kw=p_chg,
        use_batt=use_batt,
        house_batt_p_charge_kw=p_ch,  # priorità batteria di casa
        price_map=_price_map_ev,
        headroom_series=headroom_dyn,
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
export_col = "Export_KWh" if "Export_KWh" in df_hour.columns else ("Export_kWh" if "Export_kWh" in df_hour.columns else None)
if export_col is None:
    df_hour["Export_KWh"] = (df_hour["PV_kWh"] - df_hour["Autocons_kWh"]).clip(lower=0)
    export_col = "Export_KWh"

net_col = "NetGrid_KWh" if "NetGrid_KWh" in df_hour.columns else ("NetGrid_kWh" if "NetGrid_kWh" in df_hour.columns else None)
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
baseline = df_hour.copy()  # baseline = stesso carico (incluso EV), ma senza batteria
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
def split_ev_coverage(df_base: pd.DataFrame,
                      df_with_batt: Optional[pd.DataFrame],
                      use_batt: bool) -> Dict[str, float]:
    """Ripartisce EV_kWh tra FV diretto, Batteria e Rete."""
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

y_cols = [c for c in ["Total","PV_kWh","EV_kWh"] if c in daily.columns]

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
# 💶 BLOCCO ECONOMICO
# =============================================================================

def _fmt_eur(x: float) -> str:
    s = f"{x:,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
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
    monthly_econ = tmp.resample("MS")[
        ["import_cost", "export_rev", "net_cost"]
    ].sum()
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

st.sidebar.header("💸 Parametri investimento")
eval_mode = st.sidebar.selectbox(
    "Valutazione investimento",
    ["PV + Batteria vs Solo Rete", "Solo Batteria vs PV esistente"]
)
capex_pv = st.sidebar.number_input("CAPEX FV (€)", min_value=0, value=7000, step=100)
capex_batt = st.sidebar.number_input("CAPEX Batteria (€)", min_value=0, value=4000, step=100)
om_pct = st.sidebar.number_input("O&M annuo (% CAPEX considerato)", min_value=0.0, max_value=10.0, value=1.0, step=0.1) / 100.0

incent_pct = st.sidebar.number_input("Incentivo % su CAPEX (rateizzato)", min_value=0.0, max_value=100.0, value=0.0, step=1.0) / 100.0
incent_years = st.sidebar.number_input("Anni incentivo", min_value=1, max_value=20, value=10, step=1)

batt_repl_year = st.sidebar.number_input("Sostituzione batteria (anno, 0=mai)", min_value=0, max_value=30, value=0, step=1)
batt_repl_cost = st.sidebar.number_input("Costo sostituzione batteria (€)", min_value=0, max_value=50000, value=0, step=100)

years = st.sidebar.number_input("Orizzonte (anni)", min_value=1, max_value=30, value=20, step=1)
disc = st.sidebar.number_input("Tasso di sconto %", min_value=0.0, max_value=20.0, value=5.0, step=0.5) / 100.0
esc = st.sidebar.number_input("Crescita prezzo energia % (annua)", min_value=0.0, max_value=30.0, value=0.0, step=0.5) / 100.0
salvage_pct = st.sidebar.number_input("Valore residuo a fine vita (% CAPEX considerato)", min_value=0.0, max_value=100.0, value=0.0, step=1.0) / 100.0

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

