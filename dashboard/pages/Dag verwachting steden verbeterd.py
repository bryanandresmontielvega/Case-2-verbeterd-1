# dashboard_open_meteo_24h_and_week_osi.py
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

# ==================== CONFIG ====================
CITY_COORDS = {
    "Amsterdam": (52.3676, 4.9041),
    "Rotterdam": (51.9244, 4.4777),
    "Den Haag": (52.0705, 4.3007),
    "Utrecht": (52.0907, 5.1214),
}
STEDEN = list(CITY_COORDS.keys())

METRICS = {
    "Neerslag (mm)": ("rain_mm", "Neerslag (mm/u)", "Neerslag"),
    "Temperatuur (°C)": ("temp_c", "Temperatuur (°C)", "Temperatuur"),
    "Windsnelheid (km/h)": ("wind_kmh", "Wind (km/h)", "Wind"),
}

st.set_page_config(page_title="Weer Dashboard – Open-Meteo + ERA5 + OSI", layout="wide")
st.title("🌤️ Weer Dashboard – Open-Meteo (24u) & Climatologie (ERA5, week vooruit)")

# ==================== HELPERS ====================
def _ensure_len_24(df: pd.DataFrame) -> pd.DataFrame:
    return df.head(24).copy()

def _label_from_times(times: list[str]) -> list[str]:
    return [t[11:16] if len(t) >= 16 else str(t) for t in times]

# ==================== 24h FORECAST (Open-Meteo) ====================
@st.cache_data(ttl=900)
def fetch_openmeteo_24h(city: str) -> pd.DataFrame:
    lat, lon = CITY_COORDS[city]
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,windspeed_10m",
        "timezone": "auto",
        "windspeed_unit": "kmh",
        "forecast_days": 2,
    }
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame({
        "time": h["time"],
        "temp_c": h["temperature_2m"],
        "rain_mm": h["precipitation"],
        "wind_kmh": h["windspeed_10m"],
    })
    df = _ensure_len_24(df)
    df["time_label"] = _label_from_times(df["time"].tolist())
    return df[["time", "time_label", "temp_c", "rain_mm", "wind_kmh"]]

# ==================== WEEK VOORUIT (ERA5 climatologie) ====================
@st.cache_data(ttl=6*3600)
def fetch_week_climatology(city: str, years: int = 1) -> pd.DataFrame:
    """Komende 7 dagen per uur: gemiddelde van dezelfde week en hetzelfde uur uit vorig jaar of afgelopen 5 jaar (ERA5)."""
    lat, lon = CITY_COORDS[city]
    ref = fetch_openmeteo_24h(city)
    future_start = datetime.fromisoformat(ref["time"].iloc[0])
    hours = [future_start + timedelta(hours=i) for i in range(7*24)]
    future_times = [dt.strftime("%Y-%m-%dT%H:00") for dt in hours]

    stacks_temp, stacks_rain, stacks_wind = [], [], []
    for y in range(1, years + 1):
        start_past = (future_start - relativedelta(years=y)).date().isoformat()
        end_past   = (future_start + timedelta(days=7) - relativedelta(years=y)).date().isoformat()
        url = "https://archive-api.open-meteo.com/v1/era5"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_past,
            "end_date": end_past,
            "hourly": "temperature_2m,precipitation,windspeed_10m",
            "timezone": "auto",
            "windspeed_unit": "kmh",
        }
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        h = r.json()["hourly"]
        temp = np.array(h["temperature_2m"][:168], dtype=float)
        rain = np.array(h["precipitation"][:168], dtype=float)
        wind = np.array(h["windspeed_10m"][:168], dtype=float)
        stacks_temp.append(temp); stacks_rain.append(rain); stacks_wind.append(wind)

    df = pd.DataFrame({
        "time": future_times[:168],
        "temp_c": np.mean(np.vstack(stacks_temp), axis=0),
        "rain_mm": np.mean(np.vstack(stacks_rain), axis=0),
        "wind_kmh": np.mean(np.vstack(stacks_wind), axis=0),
    })
    df["time_label"] = _label_from_times(df["time"].tolist())
    df["dag"] = [t[:10] for t in df["time"]]
    return df[["time", "time_label", "dag", "temp_c", "rain_mm", "wind_kmh"]]

# ==================== UI KEUZES ====================
metric_label = st.selectbox("Kies grootheid", list(METRICS.keys()), index=1)
metric_col, y_label, legend_base = METRICS[metric_label]

st.subheader("Steden (overlay)")
cols = st.columns(4)
sel = []
for i, stad in enumerate(STEDEN):
    with cols[i]:
        if st.checkbox(stad, value=(stad == "Amsterdam")):
            sel.append(stad)
if not sel:
    st.info("Selecteer ten minste één stad.")
    st.stop()

# ==================== LAAD 24H DATA ====================
data_24h, errors = {}, []
for stad in sel:
    try:
        data_24h[stad] = fetch_openmeteo_24h(stad)
    except Exception as e:
        errors.append(f"{stad}: {e}")
if errors:
    st.warning("Kon sommige steden niet laden:\n- " + "\n- ".join(errors))
if not data_24h:
    st.stop()

# ==================== PLOT 24 UUR ====================
eerste = next(iter(data_24h))
x_labels = data_24h[eerste]["time_label"].tolist()
x = range(len(x_labels))

fig, ax = plt.subplots(figsize=(20, 10))
ax.set_xticks(list(x)); ax.set_xticklabels(x_labels, rotation=45, ha="right")
ax.set_ylabel(y_label); ax.set_title(f"Komende 24 uur – {metric_label} (Open-Meteo)")
for stad, df in data_24h.items():
    serie = df.loc[: len(x_labels) - 1, metric_col]
    ax.plot(range(len(serie)), serie, linewidth=2, label=stad)
ax.legend(title="Stad", loc="upper left"); ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

# ==================== WEEK VOORUIT (ERA5) ====================
st.subheader("Week vooruit – Climatologische verwachting (ERA5, zelfde week & uur)")
keuze_jaren = st.radio("Vergelijkingsbasis", ["Afgelopen jaar", "Afgelopen 5 jaar"], horizontal=True)
years = 1 if keuze_jaren == "Afgelopen jaar" else 5

week_dfs, err_week = {}, []
for stad in sel:
    try:
        week_dfs[stad] = fetch_week_climatology(stad, years=years)
    except Exception as e:
        err_week.append(f"{stad}: {e}")
if err_week:
    st.warning("Kon historische weekgegevens niet laden:\n- " + "\n- ".join(err_week))

if week_dfs:
    plot_df = []
    for stad, df in week_dfs.items():
        tmp = df[["time", "time_label", metric_col]].copy()
        tmp["stad"] = stad
        plot_df.append(tmp)
    plot_df = pd.concat(plot_df, ignore_index=True)
    figw = px.line(
        plot_df, x="time", y=metric_col, color="stad",
        labels={metric_col: y_label, "time": "Tijd"},
        title=f"Climatologische verwachting komende 7 dagen – {metric_label} ({keuze_jaren.lower()})",
    )
    figw.update_xaxes(tickformat="%d-%m %H:%M")
    st.plotly_chart(figw, use_container_width=True)

# ==================== WETENSCHAPPELIJKE OSI-RANKING (24u) ====================
st.subheader("Beste omstandigheden voor een dagje uit (komende 24 uur) – OSI (UCCI-geïnspireerd)")

# --- Utility-curves (0..1) ---
def u_temp(temp_c: np.ndarray) -> np.ndarray:
    t = np.asarray(temp_c, dtype=float)
    u = np.piecewise(
        t,
        [(t>=20)&(t<=23),
         ((t>=15)&(t<20))|((t>23)&(t<=26)),
         ((t>=12)&(t<15))|((t>26)&(t<=29))],
        [1.0, 0.75, 0.5, 0.0]
    )
    # zachte daling buiten de banden
    u = np.maximum(u, np.clip(1.0 - np.abs(t-21.5)/15.0, 0, 1))
    return u

def u_rain(rain_mm: np.ndarray) -> np.ndarray:
    r = np.asarray(rain_mm, dtype=float)
    return np.select(
        [r==0, (r>0)&(r<=0.2), (r>0.2)&(r<=0.5), (r>0.5)&(r<=1.0), (r>1.0)&(r<=2.0), r>2.0],
        [1.0, 0.9, 0.7, 0.4, 0.2, 0.0], default=0.0
    )

def u_wind(wind_kmh: np.ndarray) -> np.ndarray:
    w = np.asarray(wind_kmh, dtype=float)
    return np.select(
        [w<15, (w>=15)&(w<25), (w>=25)&(w<35), (w>=35)&(w<50), w>=50],
        [1.0, 0.8, 0.5, 0.2, 0.0], default=0.0
    )

# --- UCCI-geïnspireerde gewichten hergeschaald naar 3 variabelen ---
W_TEMP, W_RAINABS, W_WINDABS = 0.375, 0.333, 0.292

def osi_uci_like(temp_c, wind_kmh, rain_mm):
    ut = u_temp(temp_c); ur = u_rain(rain_mm); uw = u_wind(wind_kmh)
    score01 = W_TEMP*ut + W_RAINABS*ur + W_WINDABS*uw
    return 100.0 * score01

def meets_constraints(temp_c, wind_kmh, rain_mm):
    return (rain_mm <= 0.2) & (wind_kmh < 35) & (temp_c >= 8) & (temp_c <= 32)

# --- Ranking ---
ranking = []
for stad, df in data_24h.items():
    osi = osi_uci_like(df["temp_c"].values, df["wind_kmh"].values, df["rain_mm"].values)
    ok  = meets_constraints(df["temp_c"].values, df["wind_kmh"].values, df["rain_mm"].values)
    valid_idx = np.where(ok)[0]
    if valid_idx.size:
        best_i = valid_idx[np.argmax(osi[valid_idx])]
    else:
        best_i = int(np.argmax(osi))
    ranking.append({
        "stad": stad,
        "score": float(osi[best_i]),
        "temp_c": float(df["temp_c"].iloc[best_i]),
        "wind_kmh": float(df["wind_kmh"].iloc[best_i]),
        "rain_mm": float(df["rain_mm"].iloc[best_i]),
    })

rank_df = pd.DataFrame(ranking).sort_values("score", ascending=False).reset_index(drop=True)
rank_df.index = rank_df.index + 1  # start index bij 1
st.dataframe(rank_df.head(3), use_container_width=True)

st.caption(
    "OSI-wegingen geïnspireerd door **UCCI** (toeristische comfortindex); hergeschaald naar "
    "temperatuur, afwezigheid van regen en afwezigheid van sterke wind. Harde voorwaarden: "
    "regen ≤ 0.2 mm/u, wind < 35 km/h, temperatuur 8–32 °C."
)
