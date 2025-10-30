
import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from typing import Dict, List, Tuple
import plotly.express as px
from key import API_KEY


try:
    from key import API_KEY  
except Exception:
    API_KEY = os.environ.get("WEERLIVE_API_KEY", "")

if not API_KEY:
    st.stop()

# Stedenlijst
STEDEN = ["Amsterdam", "Rotterdam", "Den Haag", "Utrecht"]

# Metric mapping: (kolomnaam in df, y-as label, legenda label)
METRICS: Dict[str, Tuple[str, str, str]] = {
    "Neerslag (mm)": ("rain_mm", "Neerslag (mm)", "Neerslag"),
    "Temperatuur (°C)": ("temp_c", "Temperatuur (°C)", "Temperatuur"),
    "Windsnelheid (km/h)": ("wind_kmh", "Wind (km/h)", "Wind"),
}


@st.cache_data(ttl=600)
def fetch_hourly(locatie: str) -> pd.DataFrame:
    """
    Haalt uurlijkse verwachting op voor een locatie.
    Retourneert eerste 24 uur als DataFrame met kolommen:
    time_label (HH:MM), temp_c, rain_mm, wind_kmh
    """
    url = "https://weerlive.nl/api/weerlive_api_v2.php"
    params = {"key": API_KEY, "locatie": locatie}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()

    df = pd.DataFrame(js.get("uur_verw", []))
    if df.empty:
        raise ValueError(f"Geen uurlijkse data ontvangen voor '{locatie}'.")

    
    colmap = {
        "uur": "time",
        "tijd": "time",
        "temp": "temp_c",
        "neersl": "rain_mm",
        "neerslag": "rain_mm",
        "neerslag_mm": "rain_mm",
        "windkmh": "wind_kmh",
        "wind_kmh": "wind_kmh",
    }
    existing = {k: v for k, v in colmap.items() if k in df.columns}
    df = df.rename(columns=existing)

    needed = ["time", "temp_c", "rain_mm", "wind_kmh"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Ontbrekende velden voor '{locatie}': {missing}. Ontvangen kolommen: {list(df.columns)}"
        )

    df = df.head(24).copy()

    df["time_label"] = df["time"].astype(str).str.extract(r"(\d{2}:\d{2})")

    df["time_label"] = df["time_label"].fillna(df["time"].astype(str))

 
    for c in ["temp_c", "rain_mm", "wind_kmh"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df[["time_label", "temp_c", "rain_mm", "wind_kmh"]]



st.set_page_config(page_title="Weer Dashboard (Uurverwachting)", layout="wide")
st.title("Uurverwachting – Multi-stad Dashboard")
st.caption("Bron: WeerLive API • Selecteer metric en steden voor overlay")


metric_label = st.selectbox("Kies een grootheid", list(METRICS.keys()), index=1)  # standaard Temperatuur
metric_col, y_label, legend_base = METRICS[metric_label]


st.subheader("Steden")
c1, c2, c3, c4 = st.columns(4)
with c1:
    sel_ams = st.checkbox("Amsterdam", value=True)
with c2:
    sel_rtm = st.checkbox("Rotterdam", value=False)
with c3:
    sel_dh = st.checkbox("Den Haag", value=False)
with c4:
    sel_ut = st.checkbox("Utrecht", value=False)

geselecteerde_steden: List[str] = []
if sel_ams: geselecteerde_steden.append("Amsterdam")
if sel_rtm: geselecteerde_steden.append("Rotterdam")
if sel_dh: geselecteerde_steden.append("Den Haag")
if sel_ut: geselecteerde_steden.append("Utrecht")

if not geselecteerde_steden:
    st.info("Selecteer ten minste één stad.")
    st.stop()


data_per_stad: Dict[str, pd.DataFrame] = {}
errors: List[str] = []
for stad in geselecteerde_steden:
    try:
        data_per_stad[stad] = fetch_hourly(stad)
    except Exception as e:
        errors.append(f"{stad}: {e}")

if errors:
    st.warning("Kon sommige steden niet laden:\n- " + "\n- ".join(errors))

if not data_per_stad:
    st.error("Geen data beschikbaar om te plotten.")
    st.stop()

eerste_stad = next(iter(data_per_stad.keys()))
x_labels = data_per_stad[eerste_stad]["time_label"].tolist()
x = range(len(x_labels))


fig, ax = plt.subplots(figsize=(20, 10))
ax.set_xticks(list(x))
ax.set_xticklabels(x_labels, rotation=45, ha="right")
ax.set_ylabel(y_label)
ax.set_title(f"Komende 24 uur – {metric_label}")

for stad, df in data_per_stad.items():
    n = min(len(df), len(x_labels))
    serie = df.loc[: n - 1, metric_col]
    ax.plot(range(n), serie, linewidth=2, label=stad)

ax.legend(title="Stad", loc="upper left")
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

st.subheader("Beste weersomstandigheden voor dagje uit")


date_col = next((c for c in ["time_label", "Datum", "date"] if c in df.columns), None)
temp_col = next((c for c in ["temp_c", "Temperatuur", "temp"] if c in df.columns), None)
wind_col = next((c for c in ["wind_kmh", "Windsnelheid", "wind_ms", "wind"] if c in df.columns), None)
rain_col = next((c for c in ["rain_mm", "Neerslag", "precip_mm", "rain"] if c in df.columns), None)
loc_col  = next((c for c in ["location", "Stad", "city", "plaats"] if c in df.columns), None)

if not all([temp_col, wind_col, rain_col]):
    st.error("Ontbrekende kolommen: verwacht temperatuur, wind en neerslag (bv. temp_c, wind_kmh, rain_mm).")
else:
    import numpy as np
    df_num = df.copy()
    for c in [temp_col, wind_col, rain_col]:
        df_num[c] = pd.to_numeric(df_num[c], errors="coerce")

    def score_row(t, w, r):
        
        s_temp = np.clip((t - 0) / (30 - 0), 0, 1) if pd.notna(t) else 0.5   
        
        if wind_col == "wind_kmh":
            w_kmh = w
        elif wind_col == "wind_ms":
            w_kmh = w * 3.6
        else:
            w_kmh = w
        s_wind = 1 - np.clip((w_kmh - 0) / (50 - 0), 0, 1) if pd.notna(w_kmh) else 0.5  
        s_rain = 1 - np.clip((r - 0) / (10 - 0), 0, 1) if pd.notna(r) else 0.5          
        return (s_temp + s_wind + s_rain) / 3

    lines = []

    if loc_col:
        
        if date_col and date_col in df_num.columns:
            idx = df_num.groupby(loc_col)[date_col].idxmax()
            last = df_num.loc[idx]
        else:
            last = df_num.sort_index().groupby(loc_col).tail(1)
        last = last[[loc_col, temp_col, wind_col, rain_col]].copy()

        last["Score"] = last.apply(lambda r: score_row(r[temp_col], r[wind_col], r[rain_col]), axis=1)
        last = last.sort_values("Score", ascending=False)

        for _, r in last.iterrows():
            w = float(r[wind_col])
            w_ms = w/3.6 if wind_col == "wind_kmh" else (w if wind_col == "wind_ms" else w/3.6)
            lines.append(
                f"🌆 **{r[loc_col]}** — Score: **{r['Score']:.2f}**  \n"
                f"• Temp: {float(r[temp_col]):.1f}°C  • Wind: {w:.1f}"
                f"{' km/h' if wind_col!='wind_ms' else ' m/s'}"
                f"{'' if wind_col=='wind_ms' else f' ({w_ms:.1f} m/s)'}  • Neerslag: {float(r[rain_col]):.1f} mm  \n"
                f"_Opbouw:_ temp ↑, wind ↓, neerslag ↓ (gelijke weging)"
            )
    else:
    
        try:
            stadnaam = geselecteerde_steden[0] if 'geselecteerde_steden' in locals() and len(geselecteerde_steden)==1 else "geselecteerde stad"
        except Exception:
            stadnaam = "geselecteerde stad"

        
        if date_col and date_col in df_num.columns:
            r = df_num.loc[df_num[date_col].idxmax()]
        else:
            r = df_num.iloc[-1]

        t, w, rn = float(r[temp_col]), float(r[wind_col]), float(r[rain_col])
        score = score_row(t, w, rn)
        w_ms = w/3.6 if wind_col == "wind_kmh" else (w if wind_col == "wind_ms" else w/3.6)

        lines.append(
            f"🌆 **{stadnaam}** — Score: **{score:.2f}**  \n"
            f"• Temp: {t:.1f}°C  • Wind: {w:.1f}{' km/h' if wind_col!='wind_ms' else ' m/s'}"
            f"{'' if wind_col=='wind_ms' else f' ({w_ms:.1f} m/s)'}  • Neerslag: {rn:.1f} mm  \n"
            f"_Opbouw:_ temp ↑ (0–30°C), wind ↓ (0–50 km/h), neerslag ↓ (0–10 mm), gelijke weging"
        )

    st.markdown("\n\n".join(lines))

