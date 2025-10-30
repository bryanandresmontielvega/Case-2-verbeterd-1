#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 18:26:03 2025

@author: bryanandresmontielvega
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
 
st.set_page_config(page_title="November Temperatuur Voorspelling", layout="wide")
st.title("Temperatuur voorspelling model")

# --- Upload or load dataset ---
file = st.file_uploader("Upload .csv", type=["csv"])

if file:
    df = pd.read_csv(file, skipinitialspace=True)
else:
    st.info("Default dataset geladen: alle_novembers_2004 t/m 2024.csv")
    df = pd.read_csv("alle_novembers.csv", skipinitialspace=True)

# Units fix
for col in ["TG", "TN", "TX", "FG", "RH", "PG"]:
    df[col] = df[col] / 10

# Date features
df["YYYYMMDD"] = pd.to_datetime(df["YYYYMMDD"], format="%Y%m%d")
df["dag_van_het_jaar"] = df["YYYYMMDD"].dt.dayofyear
df["jaar"] = df["YYYYMMDD"].dt.year

features = ["FG", "Q", "RH", "PG", "UG", "dag_van_het_jaar", "jaar"]
targets = ["TN", "TG", "TX"]

df = df.dropna(subset=features + targets)

st.write("### Dataset voorbeeld")
st.dataframe(df.head())

# --- Data preview + uitleg ---

st.write("""
**Uitleg van variabelen:**  
- **FG** : Etmaalgemiddelde windsnelheid (0.1 m/s)  
- **TG** : Gemiddelde temperatuur (0.1 °C)  
- **TN** : Minimum temperatuur (0.1 °C)  
- **TX** : Maximum temperatuur (0.1 °C)  
- **Q**  : Globale straling (J/cm²)  
- **RH** : Neerslag (0.1 mm)  
- **PG** : Luchtdruk op zeeniveau (0.1 hPa)  
- **UG** : Relatieve vochtigheid (%)  

**Model evaluatie metrics:**  
- **RMSE** : Root Mean Squared Error → gemiddelde fout in °C  
- **MAE**  : Mean Absolute Error → gemiddelde absolute fout in °C  
- **R²**   : R-kwadraat → hoe goed de voorspellingen de variantie van de echte data verklaren
""")
if st.button("Train model & genereer voorspelling"):
    modellen = {}
    results = {}

    # Train models identiek
    for target in targets:
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=4,
            random_state=42
        )

        model.fit(X_train, y_train)
        modellen[target] = model

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[target] = (rmse, mae, r2)
        joblib.dump(model, f"model_{target}_november.pkl")

    # Predict November 2025 (same as script)
    nov_2025 = pd.date_range("2025-11-01", "2025-11-30")
    voorspel_df = pd.DataFrame({
        "YYYYMMDD": nov_2025,
        "dag_van_het_jaar": nov_2025.dayofyear,
        "jaar": [2025] * len(nov_2025)
    })

    for col in ["FG", "Q", "RH", "PG", "UG"]:
        voorspel_df[col] = df[col].mean()

    voorspel_X = voorspel_df[features]

    for target in targets:
        voorspel_df[f"verwachte_{target}"] = modellen[target].predict(voorspel_X)

    # Historical daily average
    df["maand_dag"] = df["YYYYMMDD"].dt.strftime("%m-%d")
    hist = df.groupby("maand_dag")[["TG","TN","TX"]].mean().reset_index()
    hist["YYYYMMDD"] = pd.to_datetime("2025-" + hist["maand_dag"])

    st.success("Model getraind & voorspelling klaar ")
    st.write("### 📈 Model Prestatie")
    for t,(rmse,mae,r2) in results.items():
        st.write(f"**{t}** → RMSE: `{rmse:.2f}`°C | MAE: `{mae:.2f}`°C | R²: `{r2:.3f}`")
    
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Feature Importance","SHAP", "Voorspelling 2025", "Historisch vs 2025", "Model Evaluatie"])

    # Feature importance
    with tab1:
        
        
        model = modellen["TG"]
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            "Feature": features,
            "Belang": importances
        }).sort_values(by="Belang", ascending=False)

        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(data=feature_importance_df, x="Belang", y="Feature", ax=ax)
        ax.set_title("Belangrijkste factoren temperatuur voorspelling (TG)")
        
        st.pyplot(fig)

        st.write(feature_importance_df)
        
       



    # Forecast November 2025
    with tab2:
        
        st.write("### SHAP Feature Impact gemmidelde temperatuur")
        
        # SHAP uitlegger maken
        explainer = shap.TreeExplainer(modellen["TG"])
        
        # Alleen een klein sample pakken zodat het snel blijft
        sample = X_test.sample(min(200, len(X_test)), random_state=42)
        shap_values = explainer.shap_values(sample)
        
        # Plot
        fig, ax = plt.subplots(figsize=(7,5))
        shap.summary_plot(shap_values, sample, show=False)
        st.pyplot(fig)
        
    with tab3:
        fig, ax = plt.subplots(figsize=(20,8))
        ax.plot(voorspel_df["YYYYMMDD"], voorspel_df["verwachte_TG"], label="Voorspelde TG")
        ax.fill_between(voorspel_df["YYYYMMDD"], voorspel_df["verwachte_TN"], voorspel_df["verwachte_TX"], alpha=0.3)
        ax.set_title("Voorspelde temperaturen november 2025")
        ax.grid(True)
        st.pyplot(fig)
        
    # Historical vs forecast
    with tab4:

        
        fig, ax = plt.subplots(figsize=(20,8))
        ax.plot(hist["YYYYMMDD"], hist["TG"], 
                 label="Historisch gemiddelde nov (2004–2024)", color="gray", linestyle="--")
        ax.plot(voorspel_df["YYYYMMDD"], voorspel_df["verwachte_TG"], 
                 label="Voorspeld temperatuur nov 2025", color="red", linewidth=2)
        ax.set_title("Vergelijking: Historisch november gemiddelde (2004-2024) vs. Voorspelling november 2025")

        ax.legend()
        ax.grid(True)
   
        st.pyplot(fig)

    # Model eval — scatter + residu
    with tab5:
        model = modellen["TG"]
        X = df[features]
        y = df["TG"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred

        fig, ax = plt.subplots(1,2, figsize=(10,4))
        ax[0].scatter(y_test, y_pred, alpha=0.6)
        ax[0].set_ylabel("voorspeelde temperatuur °C")
        ax[0].set_xlabel("Gemeten temperatuur °C")
        ax[0].grid(True)
        ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax[0].set_title("Voorspeld vs Echte")
        

        sns.histplot(residuals, kde=True, ax=ax[1])
        ax[1].set_title("Residuen")
        ax[1].set_ylabel("Aantal fouten")
        ax[1].set_xlabel("Voorspellingsfout °C")
        st.pyplot(fig)
        
        


    st.download_button(
        "📥 Download voorspelling CSV",
        voorspel_df.to_csv(index=False).encode("utf-8"),
        "voorspelling_november_2025.csv"
    )
