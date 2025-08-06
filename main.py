"""
Streamlit dashboard: Consumi & Produzione FV – 12 mesi (Feb 2024 → Gen 2025)
============================================================================

Prerequisiti:
    pip install streamlit pandas plotly numpy

Esecuzione locale (Windows):
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
        pd.Timestamp(year, 1, 1),
        pd.Timestamp(year, 1, 6),
        pd.Timestamp(year, 4, 25),
        pd.Timestamp(year, 5, 1),
        pd.Timestamp(year, 6, 2),
        pd.Timestamp(year, 8, 15),
        pd.Timestamp(year, 11, 1),
        pd.Timestamp(year, 12, 8),
        pd.Timestamp(year, 12, 25),
        pd.Timestamp(year, 12, 26),
    ]
    easter = _easter_sunday(year)
    easter_monday = easter + pd.Timedelta(days=1)
    return set(d.date() for d in fixed + [easter_monday])

def compute_arera_band(index_tzaware: pd.DatetimeIndex) -> pd.Series:
    if index_tzaware.tz is None:
        raise ValueError("L'indice deve essere timezone-aware (Europe/Rome).")
    years = sorted(set(index_tzaware.year))
    hol = set()
    for y in years:
        hol |= _italy_holidays(int(y))

    dow = pd.Series(index_tzaware.weekday, index=index_tzaware)  # 0=Mon..6=Sun
    hour = pd.Series(index_tzaware.hour, index=index_tzaware)
    is_holiday = pd.Series([ts.date() in hol for ts in index_tzaware], index=index_tzaware, dtype=bool)

    bands = pd.Series('F3', index=index_tzaware, dtype=object)
    sat_mask = (dow == 5) & (~is_holiday)
    bands.loc[sat_mask & hour.between(7, 22)] = 'F2'
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
    "https://raw.githubusercontent.com/<utente>/<repo>/<branch>/" + DEFAULT_NAME
)

@st.cache_data(show_spinner=True)
def load_data_auto(uploaded_file=None) -> pd.DataFrame:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        try:
            df = pd.read_csv(DEFAULT_PATH)
            st.caption(f"Fonte dati: file nella repo → {DEFAULT_NAME}")
        except Exception:
            st.caption("Fonte dati: GitHub raw (fallback)")
            df = pd.read_csv(RAW_URL)

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

        pv_direct[t] = min(L, E)
        rem_load = L - pv_direct[t]
        pv_surplus[t] = E - pv_direct[t]

        headroom = max(soc_max - soc_prev, 0.0)
        charge_batt_side_max = min(headroom, p_charge_kw)
        charge_batt_side = min(charge_batt_side_max, pv_surplus[t] * eta_ch)
        batt_charge_in[t] = charge_batt_side

        pv_used_for_charge_ac = charge_batt_side / eta_ch if eta_ch > 0 else 0.0
        pv_surplus[t] -= min(pv_surplus[t], pv_used_for_charge_ac)

        soc_tmp = soc_prev + batt_charge_in[t]

        available_batt_side = max(soc_tmp - soc_min, 0.0)
        deliverable_ac = min(rem_load, available_batt_side * eta_dis, p_discharge_kw * eta_dis)
        discharge_batt_side = deliverable_ac / eta_dis if eta_dis > 0 else 0.0
        batt_discharge_out[t] = deliverable_ac

        soc_new = soc_tmp - discharge_batt_side

        rem_after_batt = rem_load - batt_discharge_out[t]
        import_grid[t] = max(rem_after_batt, 0.0)
        export_grid[t] = max(pv_surplus[t], 0.0)

        soc[t] = np.clip(soc_new, soc_min, soc_max)
        soc_prev = soc[t]

    out["Batt_SOC_kWh"] = soc
    out["Batt_Charge_kWh"] = batt_charge_in
    out["Batt_Discharge_kWh"] = batt_discharge_out
    out["PV_Direct_kWh"] = pv_direct
    out["PV_Surplus_KWh"] = pv_surplus
    out["NetGrid_KWh_batt"] = import_grid
    out["Export_KWh_batt"] = export_grid
    out["Autocons_kWh_batt"] = out["PV_Direct_kWh"] + out["Batt_Discharge_KWh"]
    return out

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
    "Intervallo date:", value=(min_date, max_date), min_value=min_date, max_value=max_date
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

num_cols = df_filt.select_dtypes(include="number").columns
df_no_dups = df_filt.sort_index().groupby(level=0)[num_cols].sum()
df_hour = df_no_dups.resample("1h").sum(min_count=1)
df_hour["Band"] = compute_arera_band(df_hour.index)

for c in ["Total", "PV_kWh", "Autocons_kWh"]:
    if c in df_hour.columns:
        df_hour[c] = df_hour[c].fillna(0)

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

st.divider()

# -----------------------------------------------------------------------------
# Andamento giornaliero – Carichi vs Produzione FV
# -----------------------------------------------------------------------------
st.subheader("Andamento giornaliero – Carichi vs Produzione FV")
daily = df_use[["Total", "PV_kWh"]].resample("D").sum()
fig_daily = px.area(daily, x=daily.index, y=["Total", "PV_kWh"], labels={"value": "kWh al giorno", "variable": ""})
fig_daily.update_layout(legend_orientation="h", legend_y=-0.2)
st.plotly_chart(fig_daily, use_container_width=True)

# -----------------------------------------------------------------------------
# Grafico SOC batteria
# -----------------------------------------------------------------------------
if use_batt and "Batt_SOC_kWh" in df_use.columns:
    st.subheader("Andamento SOC batteria")
    fig_soc = px.line(df_use[["Batt_SOC_kWh"]], y="Batt_SOC_kWh", labels={"Batt_SOC_kWh": "SOC (kWh)", "index": "Data/Ora"})
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
pivot = df_use.assign(week=week_series).pivot_table(index=df_use.index.hour, columns="week", values=AUTOCONS, aggfunc="sum")
fig_heat = px.imshow(pivot, labels=dict(x="Settimana ISO", y="Ora", color="kWh autocons"), aspect="auto")
fig_heat.update_yaxes(dtick=1)
st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# =============================================================================
# 💶 BLOCCO ECONOMICO
# =============================================================================
st.sidebar.header("💶 Tariffe")
price_f1 = st.sidebar.number_input("Prezzo F1 (€/kWh)", min_value=0.0, value=0.30, step=0.01, format="%.3f")
price_f2 = st.sidebar.number_input("Prezzo F2 (€/kWh)", min_value=0.0, value=0.26, step=0.01, format="%.3f")
price_f3 = st.sidebar.number_input("Prezzo F3 (€/kWh)", min_value=0.0, value=0.24, step=0.01, format="%.3f")
credit_export = st.sidebar.number_input("Credito export (€/kWh)", min_value=0.0, value=0.10, step=0.01, format="%.3f")
use_export_credit = st.sidebar.checkbox("Considera credito export", value=True)
st.sidebar.caption("Nota: stima semplificata. Quote fisse, potenza impegnata, oneri/IVA non inclusi.")

def _fmt_eur(x: float) -> str:
    return f"€ {x:,.0f}".replace(",", " ").replace(".", ",").replace(" ", ".")

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

# KPI economici
st.subheader("KPI economici")
e1, e2, e3 = st.columns(3)
e1.metric("Spesa baseline (€/anno)", _fmt_eur(net_b))
e2.metric("Spesa scenario (€/anno)", _fmt_eur(net_s))
e3.metric("Risparmio (€/anno)", _fmt_eur(net_b - net_s))

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
