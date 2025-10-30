from key import API_KEY
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import re
import streamlit as st
import json

#FUNCTIONS
def reformat_with_join(number):
    num_str = str(int(number))
    formatted = '.'.join([num_str[:2], num_str[2:]])
    return float(formatted)

def request_data_location(location='Amsterdam', key=API_KEY):
    url_string = f'https://weerlive.nl/api/weerlive_api_v2.php?key={API_KEY}&locatie={location}'
    response = requests.get(url_string)
    json_data = response.json()
    live_weer_df = pd.DataFrame(json_data["liveweer"])
    return live_weer_df

#VARIABLES
url = 'https://www.webuildinternet.com/articles/2015-07-19-geojson-data-of-the-netherlands/townships.geojson'
url2 = 'https://service.pdok.nl/cbs/gebiedsindelingen/1995/wfs/v1_0?request=GetFeature&service=WFS&version=1.1.0&outputFormat=application%2Fjson%3B%20subtype%3Dgeojson&typeName=gebiedsindelingen:provincie_gegeneraliseerd'
geojson_prov = requests.get(url2).json()
geojson_data = requests.get(url).json()

pnamen = pd.read_csv('Plaatsnamen.csv',sep=';')
matches = pd.DataFrame(sorted([x['properties']['name'] for x in geojson_data['features']]))
pnamen = pnamen[np.isin(pnamen['Plaatsnaam'], matches)].reset_index(drop=True)

for i in range(len(pnamen)):
    pnamen['Longitude'][i], pnamen['Latitude'][i] = re.sub(r'[^\w\s]', '', pnamen['Longitude'][i]), re.sub(r'[^\w\s]', '', pnamen['Latitude'][i])
    pnamen['Latitude'][i] = reformat_with_join(pnamen['Latitude'][i])
    pnamen['Longitude'][i] = pnamen['Longitude'][i][0]+'.'+''.join(pnamen['Longitude'][i][1:])
    pnamen['Plaatsnaam'][i] = pnamen['Plaatsnaam'][i].replace("'",'')
pnamen[['Latitude','Longitude']] = pnamen[['Latitude','Longitude']].astype(float)
pnamen = pnamen.drop_duplicates(subset=['Plaatsnaam'],keep=False).groupby('Provincie', as_index=False).head(2).reset_index(drop=True)

pnamen[['temp','gtemp','samenv','windr']] = 0
for x in range(len(pnamen)):
    a= pnamen.loc[x,'Plaatsnaam']
    data = request_data_location(location=a)
    pnamen['temp'][x] = data['temp']
    pnamen['gtemp'][x] = data['gtemp']
    pnamen['samenv'][x] = data['samenv']
    pnamen['windr'][x] = data['windr']

#MAP
# Dit plot gebruikt dorpen/steden die zowel in de geojson als in de KNMI plaatsnamenlijst terug kwamen, vandaar de gelimiteerde selectie
df = pnamen
button_dict = {}
provincie = {}
for p in df['Provincie'].unique():
    provincie[p] = df.loc[df['Provincie'] == p, :]

fig = px.choropleth_mapbox(
    df,
    geojson=geojson_data,
    locations='Plaatsnaam',  
    featureidkey = 'properties.name',
    color='temp',
    color_continuous_scale='temps',
    hover_name="Plaatsnaam",
    hover_data=['Provincie','temp','gtemp','samenv','windr'],
    zoom=5.8,
    center={"lat": 52.092876, "lon": 5.104480},
    title='Verdeling van API data onder Nederlandse dropen en steden',
)

buttons = [dict(label="Nederland",
                method="relayout",
                args=[{"mapbox.center": {"lat": 52.092876, "lon": 5.104480},
                       "title.text": "Nederland",
                       "mapbox.zoom":5.8}]
        )]
for p in provincie:
    lat = provincie[p]['Latitude'].median()
    lon = provincie[p]['Longitude'].median()
    buttons.append(
        dict(label=p,
             method="relayout",
             args=[{"mapbox.center": {"lat": lat, "lon": lon},
                    "title.text": p,
                    "mapbox.zoom":7.5}]))

fig.update_layout(
    updatemenus=[
        dict(active=0,
            buttons=buttons,
            x=0.5,y=1.09,
            xanchor="left",
            yanchor="top")],
    title={"text":"Select Provincie",
                "y":0.96,"x":0.05,
                "xanchor":"left",
                "yanchor":"top"},
    margin={"r": 0, "t": 50, "l": 0, "b": 0},
    mapbox_style="carto-positron")
fig.update_geos(fitbounds="locations", visible=True,resolution=50,)

st.plotly_chart(fig)
