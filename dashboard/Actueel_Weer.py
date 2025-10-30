from key import API_KEY
import json
import requests
import pandas as pd
import streamlit as st


# ======================== Functie ========================
def request_data_location(location, key=API_KEY):
    url_string = f'https://weerlive.nl/api/weerlive_api_v2.php?key={key}&locatie={location}'
    response = requests.get(url_string)
    json_data = response.json()

    live_weer_df = pd.DataFrame(json_data["liveweer"])
    dagverwachting_df = pd.DataFrame(json_data["wk_verw"])
    uursverwachting_df = pd.DataFrame(json_data["uur_verw"])

    return live_weer_df, dagverwachting_df, uursverwachting_df


# ======================== Streamlit ========================
st.set_page_config(page_title="Actueel Weer Dashboard", layout="wide")

st.sidebar.header("Instellingen")
location = st.sidebar.text_input("Voer een stad in:", value="Amsterdam")

if st.sidebar.button("Toon weerdata"):
    try:
        live_weer_df, dagverwachting_df, uursverwachting_df = request_data_location(location)

        st.sidebar.success(f"Locatie geselecteerd: {location}")
        st.title(f"Actueel weer in {live_weer_df['plaats'][0]}")
        st.caption(f"Laatst geüpdatet: {live_weer_df['time'][0]}")

        # HOOFDWEER
        st.subheader("Vandaag in beeld")

        st.caption(f"Weersverwachting: {live_weer_df['verw'][0]}")

        col01, col02, _, _, _  = st.columns(5)

        col01.image(f"{live_weer_df['image'][0]}.png")

        # col01.image(f"images/mist.png")
        col02.caption(f"Het is nu {live_weer_df['samenv'][0]}")


        st.subheader("Weers waarschuwingen")
        st.caption(f"Code {live_weer_df['wrschklr'].iloc[0]}")
        st.text(f"{live_weer_df['lkop'].iloc[0]}")

        if live_weer_df['alarm'].iloc[0] == 1:
            st.text(f"{live_weer_df['ltekst'].iloc[0]}")




        st.subheader(":green[Algemeen weer]")
        col1, col2, col3 = st.columns(3)


        col1.metric(
                "Actuele temperatuur", 
                f"{float(live_weer_df['temp'].iloc[0])} °C", 
                f"Gevoel: {float(live_weer_df['gtemp'].iloc[0])} °C",
                delta_color="off",
                border=True)
        col2.metric(
                "Luchtvochtigheid", 
                f"{float(live_weer_df['lv'].iloc[0])} %", 
                f"Dauwpunt: {float(live_weer_df['dauwp'].iloc[0])} °C",
                delta_color="off",
                border=True)
        col3.metric(
                "Luchtdruk", 
                f"{float(live_weer_df['luchtd'].iloc[0])} mbar", 
                f"{float(live_weer_df['ldmmhg'].iloc[0])} mmHg", 
                delta_color="off",
                border=True)



        st.subheader(":blue[Wind]")
        col8, col9, col10 = st.columns(3)


        col8.metric(
            "Windrichting",
            f"{str(live_weer_df['windr'].iloc[0])}", 
            f"Richting: {int(live_weer_df['windrgr'].iloc[0])} °",
            delta_color="off",
            border=True
        )
        col9.metric(
            "Windsnelheid",
            f"{float(live_weer_df['windkmh'].iloc[0])} km/u", 
            f"{float(live_weer_df['windms'].iloc[0])} m/s",
            delta_color="off",
            border=True
        )
        col10.metric(
            "Windkracht (Beaufort)",
            f"{int(live_weer_df['windbft'].iloc[0])} bf", 
            f"{int(live_weer_df['windknp'].iloc[0])} kn", 
            delta_color="off",
            border=True
        )



        st.subheader(":orange[Duurzaamheid]")
        col5, col6, col7 = st.columns(3)


        col5.metric(
            "Globale zonnestraling",
            f"{str(live_weer_df['gr'].iloc[0])} W/M^2", 
            border=True
        )

        col6.metric(
            "Zon op",
            f"{str(live_weer_df['sup'].iloc[0])}", 
            border=True
        )

        col7.metric(
            "Zon onder",
            f"{str(live_weer_df['sunder'].iloc[0])}", 
            border=True
        )
    except Exception as e:
        st.error(f"Er is iets misgegaan: {e}")
else:
    st.info("Voer een stad in en klik op 'Toon weerdata'.")