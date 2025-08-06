"""
Streamlit dashboard: Consumi & Produzione FV – 12 mesi (Feb 2024 → Gen 2025)
============================================================================

Prerequisiti:
    pip install streamlit pandas plotly numpy

Esecuzione (Windows):
    cd C:\\Users\\TommasoMalaguti\\Desktop\\fv_sim_project
    streamlit run main.py
"""

from pathlib import Path
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
    Regole:
      - Domenica e festivi: F3 tutto il giorno.
      - Sabato: F2 07–22, altrimenti F3.
      - Lun–Ven (non festivi): F1 08–18; F2 07 e 19–22; F3 resto.
    """
    if index_tzaware.tz is None:
        raise ValueError("L'indice deve essere timezone-aware (Europe/Rome).")
    years = sorted(set(index_tzaware.year))
    hol = set()
    for y in years:
        hol |= _italy_holidays(int(y))

    dow = pd.Series(index_tzaware.weekday, index=index_tzaware)
    hour = pd.Series(index_tzaware.hour, index=index_tzaware)
    is_holiday = pd.Series([ts.date() in hol for ts in index_tzaware], index=index_tzaware, dtype=bool)

    bands = pd.Series('F3', index=index_tzaware, dtype=object)  # default F3
    # Sabato (non festivo): F2 07–22
    sat_mask = (dow == 5) & (~is_holiday)
    bands.loc[sat_mask & hour.between(7, 22)] = 'F2'
    # Lun–Ven (non festivi)
    mf_mask = dow.between(0, 4) & (~is_holiday)
    bands.loc[mf_mask & hour.between(8, 18)] = 'F1'
    bands.loc[mf_mask & ((hour == 7) | hour.between(19, 22))] = 'F2'
    # Domenica e festivi rimangono F3
    return bands

# -----------------------------------------------------------------------------
# Helper caricamento CSV + normalizzazione colonne
# -----------------------------------------------------------------------------
DEFAULT_NAME = "combined_Feb2024_to_Jan2025_hourly_F123_withNetGrid_lower.csv"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PATH = BASE_DIR / DEFAULT_NAME

@st.cache_data(show_spinner=True)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Timestamp → Europe/Rome (accetta stringhe con offset oppure naive/UTC)
    # Forziamo come UTC e poi convertiamo a Europe/Rome (funziona anche se Datetime ha offset)
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce").dt.tz_convert("Europe/Rome")
    df = df.dropna(subset=["Datetime"]).set_index("Datetime").sort_index()

    # Allinea nomi colonne (Export, NetGrid)
    if "Export_KWh" not in df.columns and "Export_kWh" in df.columns:
        df = df.rename(columns={"Export_kWh": "Export_KWh"})
    if "NetGrid_KWh" not in df.columns and "NetGrid_kWh" in df.columns:
        df = df.rename(columns={"NetGrid_kWh": "NetGrid_KWh"})

    # Se manca NetGrid_KWh lo ricostruiamo
    if "NetGrid_KWh" not in df.columns and {"Total", "Autocons_kWh"}.issubset(df.columns):
        df["NetGrid_KWh"] = (df["Total"] - df["Autocons_kWh"]).clip(lower=0)

    # Se manca Band la calcoliamo
    if "Band" not in df.columns:
        df["Band"] = compute_arera_band(df.index)

    return df

# -----------------------------------------------------------------------------
# Funzione di simulazione batteria (carica solo da surplus FV)
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
    Ritorna df con colonne aggiuntive *_batt e con flussi post-batteria:
      - Autocons_kWh_batt, NetGrid_KWh_batt, Export_KWh_batt, ecc.
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
    batt_discharge_out = np.zeros(n) # energia dalla batteria ai carichi (già con efficienza di scarica)
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
        charge_batt_side_max = min(headroom, p_discharge_kw)  # limite prudente: usa p_discharge_kw o p_charge_kw a scelta
        charge_batt_side_max = min(headroom, p_charge_kw)     # (preferiamo esplicito: limite potenza di carica)

        # energia lato batteria dopo efficienza
        charge_batt_side = min(charge_batt_side_max, pv_surplus[t] * eta_ch)
        batt_charge_in[t] = charge_batt_side
        # consumo FV lato AC per caricare:
        pv_used_for_charge_ac = charge_batt_side / eta_ch if eta_ch > 0 else 0.0
        pv_surplus[t] -= min(pv_surplus[t], pv_used_for_charge_ac)

        soc_tmp = soc_prev + batt_charge_in[t]

        # 3) Scarica per coprire carico residuo
        available_batt_side = max(soc_tmp - soc_min, 0.0)
        deliverable_ac = min(rem_load, available_batt_side * eta_dis, p_discharge_kw * eta_dis)
        discharge_batt_side = deliverable_ac / eta_dis if eta_dis > 0 else 0.0
        batt_discharge_out[t] = deliverable_ac

        soc_new = soc_tmp - discharge_batt_side

        # 4) Rete
        rem_after_batt = rem_load - batt_discharge_out[t]
        import_grid[t] = max(rem_after_batt, 0.0)
        export_grid[t] = max(pv_surplus[t], 0.0)

        soc[t] = np.clip(soc_new, soc_min, soc_max)
        soc_prev = soc[t]

    out["Batt_SOC_kWh"] = soc
    out["Batt_Charge_kWh"] = batt_charge_in
    out["Batt_Discharge_KWh"] = batt_discharge_out
    out["PV_Direct_kWh"] = pv_direct
    out["PV_Surplus_KWh"] = pv_surplus
    out["NetGrid_KWh_batt"] = import_grid
    out["Export_KWh_batt"] = export_grid
    out["Autocons_kWh_batt"] = out["PV_Direct_kWh"] + out["Batt_Discharge_KWh"]
    return out

# -----------------------------------------------------------------------------
# Sidebar – selezione CSV e filtri
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Impostazioni")

csv_file = None
if DEFAULT_PATH.exists():
    st.sidebar.write(f"CSV di default: **{DEFAULT_NAME}**")
    use_default = st.sidebar.checkbox("Usa file di default", value=True)
    if use_default:
        csv_file = DEFAULT_PATH

if not csv_file:
    uploaded = st.sidebar.file_uploader("Carica un CSV…", type=["csv"])
    if uploaded is not None:
        tmp_path = Path(uploaded.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        csv_file = tmp_path

if csv_file is None or not csv_file.exists():
    st.error("⚠️ Seleziona o carica il CSV per continuare.")
    st.stop()

# Caricamento
try:
    df = load_data(csv_file)
except Exception as e:
    st.error(f"Errore nel caricamento del CSV: {e}")
    st.stop()

# Filtri
st.sidebar.divider()
st.sidebar.header("Filtri dati")

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
# Simulazione batteria – controlli UI
# -----------------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.header("🔋 Batteria (simulazione)")

use_batt = st.sidebar.checkbox("Attiva batteria", value=False)
cap = st.sidebar.number_input("Capacità (kWh)", min_value=0.0, value=5.0, step=0.5)
p_ch = st.sidebar.number_input("Potenza carica (kW)", min_value=0.0, value=3.0, step=0.5)
p_dis = st.sidebar.number_input("Potenza scarica (kW)", min_value=0.0, value=3.0, step=0.5)
eta_ch = st.sidebar.slider("Efficienza carica ηch", min_value=0.80, max_value=1.00, value=0.95, step=0.01)
eta_dis = st.sidebar.slider("Efficienza scarica ηdis", min_value=0.80, max_value=1.00, value=0.95, step=0.01)
soc_min = st.sidebar.slider("SOC minimo (%)", min_value=0, max_value=50, value=5, step=1) / 100.0
soc_init = st.sidebar.slider("SOC iniziale (%)", min_value=0, max_value=100, value=50, step=5) / 100.0

# -----------------------------------------------------------------------------
# Preparazione dati (FIX duplicati → resample orario → Band)
# -----------------------------------------------------------------------------
# Diagnostica duplicati (facoltativa)
n_dup = int(df_filt.index.duplicated().sum())
if n_dup > 0:
    st.info(f"🔧 Risolti {n_dup} duplicati orari nell'intervallo selezionato.")

# 1) Collassa i duplicati sommando SOLO le colonne numeriche (kWh per ora)
num_cols = df_filt.select_dtypes(include="number").columns
df_no_dups = (
    df_filt
    .sort_index()
    .groupby(level=0)[num_cols]
    .sum()
)

# 2) Griglia oraria continua (usa '1h' – niente FutureWarning)
df_hour = df_no_dups.resample("1h").sum(min_count=1)

# 3) Ricalcola la fascia ARERA sull’indice nuovo (Europe/Rome)
df_hour["Band"] = compute_arera_band(df_hour.index)

# 4) Riempi eventuali buchi per la simulazione
for c in ["Total", "PV_kWh", "Autocons_kWh"]:
    if c in df_hour.columns:
        df_hour[c] = df_hour[c].fillna(0)

# Export/NetGrid naming robusto (ricostruisce se mancano)
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

# -----------------------------------------------------------------------------
# KPI
# -----------------------------------------------------------------------------
st.subheader("KPI")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Prelievo rete (kWh)", f"{df_use[NET].sum():,.0f}")
col2.metric("Produzione FV (kWh)", f"{df_use['PV_kWh'].sum():,.0f}")
col3.metric("Autoconsumo (kWh)", f"{df_use[AUTOCONS].sum():,.0f}")
col4.metric("Export (kWh)", f"{df_use[EXPORT].sum():,.0f}")
autonomy = (df_use[AUTOCONS].sum() / df_use["Total"].sum() * 100) if df_use["Total"].sum() else 0
col5.metric("Autonomia (%)", f"{autonomy:.1f}%")

st.divider()

# -----------------------------------------------------------------------------
# Grafico area giornaliero (Total vs PV)
# -----------------------------------------------------------------------------
st.subheader("Andamento giornaliero – Prelievo vs Produzione")
daily = df_use[["Total", "PV_kWh"]].resample("D").sum()
fig_daily = px.area(
    daily,
    x=daily.index,
    y=["Total", "PV_kWh"],
    labels={"value": "kWh al giorno", "variable": ""},
)
fig_daily.update_layout(legend_orientation="h", legend_y=-0.2)
st.plotly_chart(fig_daily, use_container_width=True)

# -----------------------------------------------------------------------------
# Grafico barre impilate mensili (scenario attivo)
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

# -----------------------------------------------------------------------------
# Tabella dati + download CSV scenario corrente
# -----------------------------------------------------------------------------
st.subheader("Anteprima dati (scenario corrente)")
st.dataframe(df_use.head(500))

st.download_button(
    "💾 Scarica CSV (scenario corrente)",
    data=df_use.to_csv(index=True).encode(),
    file_name=("consumi_pv_con_batteria.csv" if use_batt else "consumi_pv_senza_batteria.csv"),
    mime="text/csv",
)
