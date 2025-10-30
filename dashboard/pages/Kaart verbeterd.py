# weather_map_nl_week_compare_all_vars.py
import requests
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import streamlit as st
from datetime import datetime, timezone, timedelta, date

st.set_page_config(page_title="NL Weer – Nu vs weekgemiddelde (zelfde uur)", layout="wide")
st.title("🇳🇱 Nu vs weekgemiddelde (zelfde uur) – Temperatuur / Wind / Neerslag")

# ----------------- Gebied & resolutie -----------------
LAT_MIN, LAT_MAX = 50.5, 53.7
LON_MIN, LON_MAX = 3.2, 7.3
LAT_STEP = 0.4
LON_STEP = 0.4
GRID_N_LAT, GRID_N_LON = 100, 120

# ----------------- UI boven de kaart -----------------
c1, c2 = st.columns([1.2, 1.2])
with c1:
    var = st.selectbox("Variabele", ["Temperatuur (°C)", "Windsnelheid (m/s)", "Neerslag (mm)"])
with c2:
    compare_choice = st.selectbox("Vergelijkingsbasis", ["Afgelopen jaar", "Afgelopen 5 jaar"], index=0)

VAR_FIELD = {"Temperatuur (°C)": "temp", "Windsnelheid (m/s)": "wind", "Neerslag (mm)": "rain"}[var]
CMAP = {"temp": "coolwarm", "wind": "viridis", "rain": "Blues"}[VAR_FIELD]

# ----------------- Helpers -----------------
def iso_hour_now_for_response(meta: dict):
    """Return (target_iso_hour_str, local_now_dt, utc_offset_seconds)."""
    offset = int(meta.get("utc_offset_seconds", 0))
    now_utc = datetime.utcnow().replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    local_now = now_utc + timedelta(seconds=offset)
    return local_now.strftime("%Y-%m-%dT%H:00"), local_now, offset

def choose_hour_index(times, target_iso):
    if target_iso in times:
        return times.index(target_iso)
    to_dt = lambda s: datetime.strptime(s, "%Y-%m-%dT%H:%M")
    target_dt = to_dt(target_iso)
    dts = [to_dt(t) for t in times]
    past = [i for i, t in enumerate(dts) if t <= target_dt]
    if past:  # laatste ≤ target
        return past[-1]
    # anders dichtstbijzijnde
    return int(np.argmin([abs((t - target_dt).total_seconds()) for t in dts]))

def week_window_for_past_year(local_now: datetime, years_back: int):
    """Week (7 dagen) eindigend op dezelfde datum, years_back geleden."""
    ref = local_now.date().replace(year=local_now.year - years_back)
    start = ref - timedelta(days=6)
    return start.isoformat(), ref.isoformat()

def interpolate_field(samples, field, nlat=GRID_N_LAT, nlon=GRID_N_LON):
    pts = np.array([(s["lon"], s["lat"]) for s in samples], dtype=float)
    vals = np.array([s[field] for s in samples], dtype=float)
    glons = np.linspace(LON_MIN, LON_MAX, nlon)
    glats = np.linspace(LAT_MIN, LAT_MAX, nlat)
    XI, YI = np.meshgrid(glons, glats)
    ZI = griddata(pts, vals, (XI, YI), method="linear")
    mask = np.isnan(ZI)
    if mask.any():
        ZI[mask] = griddata(pts, vals, (XI, YI), method="nearest")[mask]
    return XI, YI, ZI

# ----------------- Live (alle variabelen, huidig uur) -----------------
def fetch_point_now_all(lat, lon, session, target_iso=None):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,precipitation,windspeed_10m",
        "forecast_days": 1, "timezone": "auto",
    }
    r = session.get(url, params=params, timeout=6)
    r.raise_for_status()
    j = r.json()
    h = j["hourly"]
    if target_iso is None:
        target_iso, _, _ = iso_hour_now_for_response(j)
    idx = choose_hour_index(h["time"], target_iso)
    return {
        "lat": lat, "lon": lon,
        "temp": float(h["temperature_2m"][idx]),
        "rain": float(h["precipitation"][idx]),
        "wind": float(h["windspeed_10m"][idx]),
        "time": h["time"][idx],
        "target_iso": target_iso,
        "offset": int(j.get("utc_offset_seconds", 0)),
    }

@st.cache_data(show_spinner=False, ttl=600)
def fetch_grid_now_all(lat_step, lon_step, target_iso):
    lats = np.arange(LAT_MIN, LAT_MAX + 1e-6, lat_step)
    lons = np.arange(LON_MIN, LON_MAX + 1e-6, lon_step)
    coords = [(float(la), float(lo)) for la in lats for lo in lons]
    out = []
    session = requests.Session()
    prog = st.progress(0, text="Live data ophalen…")
    total = len(coords)
    ts_effective = None
    for i, (la, lo) in enumerate(coords, 1):
        rec = fetch_point_now_all(la, lo, session, target_iso)
        ts_effective = rec["time"]
        out.append(rec)
        if i % 2 == 0 or i == total:
            prog.progress(int(i / total * 100), text=f"Live… {i}/{total}")
    prog.empty()
    return out, ts_effective

# ----------------- Historisch (week, zelfde uur) -----------------
def aggregate_week_same_hour_all(h, target_hour: str):
    """Return dict temp_mean, wind_mean, rain_sum over geselecteerde uren binnen week."""
    times = h["time"]
    sel = [i for i, t in enumerate(times) if t[11:13] == target_hour]
    if not sel:
        sel = range(len(times))
    temp_val = float(np.mean(np.array(h["temperature_2m"])[sel]))
    wind_val = float(np.mean(np.array(h["windspeed_10m"])[sel]))
    rain_val = float(np.sum(np.array(h["precipitation"])[sel]))
    return {"temp": temp_val, "wind": wind_val, "rain": rain_val}

def fetch_point_archive_week_agg(lat, lon, start_date, end_date, session, target_hour: str):
    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": "temperature_2m,precipitation,windspeed_10m",
        "timezone": "auto",
    }
    r = session.get(url, params=params, timeout=10)
    r.raise_for_status()
    j = r.json()
    return aggregate_week_same_hour_all(j["hourly"], target_hour)

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_grid_archive_multi_years_all(lat_step, lon_step, local_now, target_hour, years):
    lats = np.arange(LAT_MIN, LAT_MAX + 1e-6, lat_step)
    lons = np.arange(LON_MIN, LON_MAX + 1e-6, lon_step)
    coords = [(float(la), float(lo)) for la in lats for lo in lons]

    out = []
    session = requests.Session()
    prog = st.progress(0, text="Historische weekgegevens ophalen…")
    total = len(coords)

    for i, (la, lo) in enumerate(coords, 1):
        acc = {"temp": [], "wind": [], "rain": []}
        for y in range(1, years + 1):
            start_iso, end_iso = week_window_for_past_year(local_now, y)
            agg = fetch_point_archive_week_agg(la, lo, start_iso, end_iso, session, target_hour)
            for k in acc:
                acc[k].append(agg[k])
        # gemiddelde over jaren: temp/wind = mean van means; rain = mean van wekensommen
        out.append({
            "lat": la, "lon": lo,
            "temp": float(np.mean(acc["temp"])),
            "wind": float(np.mean(acc["wind"])),
            "rain": float(np.mean(acc["rain"])),
        })
        if i % 2 == 0 or i == total:
            prog.progress(int(i / total * 100), text=f"Historisch… {i}/{total}")

    prog.empty()
    return out

# ----------------- Bepaal referentie-uur en lokale tijd -----------------
try:
    ref = fetch_point_now_all((LAT_MIN+LAT_MAX)/2, (LON_MIN+LON_MAX)/2, requests.Session(), target_iso=None)
    target_iso = ref["target_iso"]
    target_hour = target_iso[11:13]  # bv. '14'
    local_now = datetime.utcnow().replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc) + timedelta(seconds=ref["offset"])
except Exception as e:
    st.error(f"Kon referentietijd niet bepalen: {e}")
    st.stop()

# ----------------- Ophalen live & historisch -----------------
try:
    samples_now, ts_now = fetch_grid_now_all(LAT_STEP, LON_STEP, target_iso)
except Exception as e:
    st.error(f"Fout bij ophalen live data: {e}")
    st.stop()

years = 1 if compare_choice == "Afgelopen jaar" else 5
try:
    samples_hist = fetch_grid_archive_multi_years_all(LAT_STEP, LON_STEP, local_now, target_hour, years)
except Exception as e:
    st.error(f"Fout bij ophalen historische data: {e}")
    st.stop()

# ----------------- Interpoleren o.b.v. gekozen variabele -----------------
XI, YI, ZI_now = interpolate_field(samples_now, VAR_FIELD)
_,  _,  ZI_hist = interpolate_field(samples_hist, VAR_FIELD)

# ----------------- Twee kaarten + één colorbar -----------------
fig, axes = plt.subplots(
    1, 2, figsize=(14, 6),
    subplot_kw={'projection': ccrs.PlateCarree()},
    constrained_layout=True
)

def setup_ax(ax):
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"))
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linestyle=":")

ax1, ax2 = axes
setup_ax(ax1); setup_ax(ax2)

vmin = float(np.nanmin([np.nanmin(ZI_now), np.nanmin(ZI_hist)]))
vmax = float(np.nanmax([np.nanmax(ZI_now), np.nanmax(ZI_hist)]))

m1 = ax1.pcolormesh(XI, YI, ZI_now, cmap=CMAP, shading="auto", vmin=vmin, vmax=vmax)
ax1.set_title(f"Nu – {ts_now} (uur {target_hour}:00)")

ax2.pcolormesh(XI, YI, ZI_hist, cmap=CMAP, shading="auto", vmin=vmin, vmax=vmax)
ax2.set_title(f"Weekgemiddelde (zelfde uur) • {compare_choice.lower()}")

cbar = fig.colorbar(m1, ax=axes, orientation="horizontal", fraction=0.046, pad=0.08)
label_map = {"temp": "Temperatuur (°C)", "wind": "Windsnelheid (m/s)", "rain": "Neerslag (mm)"}
cbar.set_label(label_map[VAR_FIELD])

st.pyplot(fig)

# ----------------- Diagnostiek -----------------
with st.expander("Diagnostiek"):
    st.write(f"Variabele: {var}  •  Gekozen jaren: {years}")
    st.write(f"Referentie-uur: {target_hour}:00  •  target_iso: {target_iso}")
    st.write(f"Aantal samplepunten (live): {len(samples_now)}  •  (hist): {len(samples_hist)}")
