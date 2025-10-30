# %%
from key import API_KEY
import json
import requests
import streamlit as st 
import pandas as pd


# %%
from key import API_KEY
import requests
import streamlit as st
import pandas as pd

# ======================== Functie ========================
def request_data_location(location, key=API_KEY):
    """Haalt weerdata op via Weerlive API voor opgegeven locatie"""
    url_string = f'https://weerlive.nl/api/weerlive_api_v2.php?key={key}&locatie={location}'
    response = requests.get(url_string, timeout=10)
    response.raise_for_status()
    json_data = response.json()
    live_weer_df = pd.DataFrame(json_data.get("liveweer", []))
    dagverwachting_df = pd.DataFrame(json_data.get("wk_verw", []))
    uursverwachting_df = pd.DataFrame(json_data.get("uur_verw", []))
    return live_weer_df, dagverwachting_df, uursverwachting_df

# ======================== Pagina-config ========================
st.set_page_config(page_title="Weekverwachting", layout="wide")

# ======================== Session-state defaults ========================
if "location" not in st.session_state:
    st.session_state["location"] = "Amsterdam"
if "dag_index" not in st.session_state:
    st.session_state["dag_index"] = 0
if "data_loaded" not in st.session_state:
    st.session_state["data_loaded"] = False
# dataframes: live, dag, uur (optioneel aanwezig)
# st.session_state["dagverwachting_df"] etc. will be created on load

# ======================== Sidebar ========================
st.sidebar.header("Instellingen")

# input met expliciete key (blijft behouden tussen reruns)
st.session_state["location"] = st.sidebar.text_input(
    "Voer een stad in:", value=st.session_state["location"], key="location_input"
)

# knop om gegevens op te halen (unieke key)
if st.sidebar.button("Toon weekverwachting", key="load_week"):
    try:
        live_df, dag_df, uur_df = request_data_location(st.session_state["location"])
        # bewaar data in session_state zodat interacties later geen reload vereisen
        st.session_state["live_weer_df"] = live_df
        st.session_state["dagverwachting_df"] = dag_df.reset_index(drop=True)
        st.session_state["uursverwachting_df"] = uur_df
        st.session_state["dag_index"] = 0  # reset naar eerste dag na load
        st.session_state["data_loaded"] = True
        st.sidebar.success(f"Data geladen voor {st.session_state['location']}")
    except Exception as e:
        st.session_state["data_loaded"] = False
        st.sidebar.error(f"Data ophalen mislukt: {e}")

# ======================== Als data geladen is (of al in session_state) ========================
if st.session_state.get("data_loaded") and "dagverwachting_df" in st.session_state:
    dagverwachting_df = st.session_state["dagverwachting_df"]

    # achtergrond (optioneel)
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://images.unsplash.com/photo-1532767153582-b1a0e5145009?q=80&w=2814&auto=format&fit=crop');
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # veiligheidscheck: zorg dat kolom 'dag' bestaat en niet leeg is
    if dagverwachting_df.empty or "dag" not in dagverwachting_df.columns:
        st.error("Dagverwachting niet beschikbaar voor deze locatie.")
    else:
        # Zorg dat datum-kolom goede dtype heeft en maak korte dagnaam
        try:
            dagverwachting_df["dag_vd_week"] = pd.to_datetime(
                dagverwachting_df["dag"], dayfirst=True, errors="coerce"
            ).dt.strftime("%a")
        except Exception:
            dagverwachting_df["dag_vd_week"] = dagverwachting_df["dag"].astype(str)

        # Ensure dag_index valid
        max_index = len(dagverwachting_df) - 1
        if st.session_state["dag_index"] > max_index:
            st.session_state["dag_index"] = max_index
        if st.session_state["dag_index"] < 0:
            st.session_state["dag_index"] = 0

        # keuze-knoppen en slider (geef keys zodat state consistent blijft)
        col1, col2, col3 = st.columns([1, 6, 1])
        with col1:
            if st.button("◀️", key="prev_day"):
                if st.session_state["dag_index"] > 0:
                    st.session_state["dag_index"] -= 1
        with col3:
            if st.button("▶️", key="next_day"):
                if st.session_state["dag_index"] < max_index:
                    st.session_state["dag_index"] += 1

        # weekdagen lijst en slider met key; slider bewaart waarde in session_state
        weekdagen = dagverwachting_df["dag_vd_week"].fillna("").tolist()
        # fallback: als weekdagen lege strings heeft, gebruik dag strings
        if not any(weegd for weegd in weekdagen):
            weekdagen = dagverwachting_df["dag"].astype(str).tolist()

        # slider: we gebruiken key "day_slider" en zorgen dat de value gebaseerd is op dag_index
        current_slider_value = weekdagen[st.session_state["dag_index"]]
        chosen_day = st.select_slider(
            "Kies een dag",
            options=weekdagen,
            value=current_slider_value,
            key="day_slider",
            label_visibility="collapsed",
        )

        # Synchroniseer slider -> dag_index
        if chosen_day != weekdagen[st.session_state["dag_index"]]:
            st.session_state["dag_index"] = weekdagen.index(chosen_day)

        # geselecteerde dag ophalen
        selected_day = dagverwachting_df.iloc[st.session_state["dag_index"]]
        # icon en kleurmap (zelfde als eerder)
        weer_icons = {
            "zon": "☀️",
            "halfbewolkt": "⛅",
            "bewolkt": "☁️",
            "regen": "🌧️",
            "onweer": "⛈️",
            "sneeuw": "❄️",
        }
        theme_colors = {
            "zon": "linear-gradient(150deg, #FFD500, #F2F5F1)",
            "halfbewolkt": "linear-gradient(135deg, #87CEEB, #FFFFFF)",
            "bewolkt": "linear-gradient(135deg, #B0C4DE, #F0F0F0)",
            "regen": "linear-gradient(135deg, #00BFFF, #F0F0F0)",
            "onweer": "linear-gradient(135deg, #708090, #2F4F4F)",
            "sneeuw": "linear-gradient(135deg, #E0FFFF, #FFFFFF)",
        }

        icon = weer_icons.get(selected_day.get("image", ""), "🌤️")
        bg_color = theme_colors.get(selected_day.get("image", ""), "linear-gradient(135deg, #89CFF0, #FFFFFF)")

        # kaartje tonen
        st.markdown(
            f"""
            <div style="padding:20px; border-radius:30px; background:{bg_color}; 
                        box-shadow: 0px 4px 12px rgba(0,0,0,0.15); text-align:center; color:black;">
                <h2 style="margin-bottom:10px;">{icon} Weer op {selected_day.get('dag_vd_week', '')} {selected_day.get('dag', '')}</h2>
                <p style="font-size:20px; margin:5px;">🌡️ <b>Max:</b> {selected_day.get('max_temp', '—')}°C</p>
                <p style="font-size:20px; margin:5px;">🌡️ <b>Min:</b> {selected_day.get('min_temp', '—')}°C</p>
                <p style="font-size:20px; margin:5px;">💨 <b>Wind:</b> {selected_day.get('windbft', '—')} Bft ({selected_day.get('windkmh','—')} km/h, {selected_day.get('windr','—')})</p>
                <p style="font-size:20px; margin:5px;">☔ <b>Kans op regen:</b> {selected_day.get('neersl_perc_dag', '—')}%</p>
                <p style="font-size:20px; margin:5px;">☀️ <b>Zonkans:</b> {selected_day.get('zond_perc_dag', '—')}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    # nog geen data geladen - informeer en geef shortcut: laad met huidige locatie
    st.info("Voer een stad in de sidebar in en klik op 'Toon weekverwachting' om de verwachte week te zien.")