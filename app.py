import streamlit as st
import pandas as pd
from custom_transformers import CustomFeaturesAdder
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import requests
import io
import tempfile
from sklearn.base import BaseEstimator, TransformerMixin
import random

# Konfiguracja strony
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URLs do modeli na Google Drive (do uzupełnienia)
MODEL_URLS = {
    "XGBoost Pipeline": "https://drive.google.com/uc?id=1azA3B593CPUASgtxpw1FF8bZtyw22K5S",
    "Random Forest Pipeline": "https://drive.google.com/uc?id=1tVm4jiAk8M9M6KFtrrL0x9BEGM0bEEcE",
    "Best Overall Model": "https://drive.google.com/uc?id=1ALDixWPGP8YExM2lAngRjOWcAFEpBlu4"
}


# Cache dla ładowania modelu z URL
@st.cache_resource
def load_model_from_url(url, model_name):
    """Ładuje model z URL Google Drive"""
    try:
        with st.spinner(f"Ładowanie modelu {model_name}..."):
            response = requests.get(url)
            response.raise_for_status()
            
            # Zapisz do tymczasowego pliku
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                tmp_file.write(response.content)
                tmp_file_path = tmp_file.name
            
            # Wczytaj model
            model = joblib.load(tmp_file_path)
            return model, model_name
            
    except Exception as e:
        st.error(f"Błąd podczas ładowania modelu {model_name}: {str(e)}")
        return None, None

@st.cache_data
def load_data():
    """Ładuje dane do analizy"""
    try:
        df = pd.read_csv("data/housing.csv")
        return df
    except FileNotFoundError:
        st.warning("Nie znaleziono lokalnego pliku z danymi. Możesz przesłać własny plik CSV.")
        return None


def get_county_options():
    """Zwraca listę dostępnych hrabstw"""
    return [
        "Los Angeles", "Orange", "San Diego", "Alameda", "Santa Clara", 
        "Riverside", "San Bernardino", "Contra Costa", "Fresno", "Sacramento",
        "Ventura", "San Francisco", "Kern", "San Joaquin", "Stanislaus",
        "Sonoma", "Tulare", "Santa Barbara", "Solano", "Monterey",
        "Placer", "San Luis Obispo", "Merced", "Santa Cruz", "Marin",
        "Butte", "Yolo", "El Dorado", "Imperial", "Kings", "Madera",
        "Napa", "Shasta", "Sutter", "Tehama", "Tuolumne", "Calaveras",
        "Lake", "Mendocino", "Nevada", "Humboldt", "Del Norte", "Lassen",
        "Modoc", "Mono", "Inyo", "Alpine", "Amador", "Colusa", "Glenn",
        "Mariposa", "Plumas", "Sierra", "Siskiyou", "Trinity", "Yuba"
    ]

def get_ocean_proximity_options():
    """Zwraca listę opcji bliskości oceanu"""
    return ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]

def validate_csv_data(df):
    """Waliduje dane z CSV"""
    required_columns = [
        'housing_median_age', 'total_rooms', 'total_bedrooms', 
        'population', 'households', 'median_income', 'ocean_proximity'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Brakujące kolumny: {', '.join(missing_columns)}"
    
    # Sprawdź czy kolumny kategoryczne mają odpowiednie wartości
    valid_ocean_proximity = get_ocean_proximity_options()
    invalid_ocean = df[~df['ocean_proximity'].isin(valid_ocean_proximity)]['ocean_proximity'].unique()
    if len(invalid_ocean) > 0:
        return False, f"Nieprawidłowe wartości w kolumnie ocean_proximity: {', '.join(invalid_ocean)}"
    
    # Sprawdź kolumnę county jeśli istnieje
    if 'county' in df.columns:
        valid_counties = get_county_options()
        invalid_counties = df[~df['county'].isin(valid_counties)]['county'].unique()
        if len(invalid_counties) > 0:
            st.warning(f"Uwaga: Nieznane hrabstwa w danych: {', '.join(invalid_counties)}")
    
    return True, "Dane są prawidłowe"

def process_uploaded_data(df):
    """Przetwarza przesłane dane"""
    # Jeśli nie ma kolumny county, dodaj domyślną
    if 'county' not in df.columns:
        df['county'] = 'Los Angeles'
    
    return df

def create_prediction_interface(model):
    """Tworzy interfejs do pojedynczej predykcji"""
    st.subheader("🔮 Pojedyncza predykcja")
    
    # Sekcja z inputami
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📍 Lokalizacja**")
        county = st.selectbox(
            "Hrabstwo (County)",
            options=get_county_options(),
            index=0,
            key="single_county"
        )
        
        ocean_proximity = st.selectbox(
            "Bliskość oceanu",
            options=get_ocean_proximity_options(),
            index=0,
            key="single_ocean"
        )
    
    with col2:
        st.markdown("**🏘️ Charakterystyka obszaru**")
        housing_median_age = st.slider(
            "Mediana wieku domów (lata)",
            min_value=1, max_value=52, value=29,
            key="single_age"
        )
        
        population = st.number_input(
            "Populacja",
            min_value=3, max_value=35682, value=3500,
            key="single_population"
        )
        
        households = st.number_input(
            "Liczba gospodarstw domowych",
            min_value=1, max_value=6082, value=700,
            key="single_households"
        )
    
    with col3:
        st.markdown("**🏠 Parametry nieruchomości**")
        total_rooms = st.number_input(
            "Całkowita liczba pokojów",
            min_value=2, max_value=39320, value=5000,
            key="single_rooms"
        )
        
        total_bedrooms = st.number_input(
            "Całkowita liczba sypialni",
            min_value=1, max_value=6445, value=1000,
            key="single_bedrooms"
        )
        
        median_income = st.slider(
            "Mediana dochodu (w 10k USD)",
            min_value=0.5, max_value=15.0, value=5.0, step=0.1,
            key="single_income"
        )
    
    # Przycisk predykcji
    if st.button("🔮 Przewiduj cenę", type="primary"):
        # Przygotowanie danych do predykcji
        input_data = pd.DataFrame({
            'housing_median_age': [housing_median_age],
            'total_rooms': [total_rooms],
            'total_bedrooms': [total_bedrooms],
            'population': [population],
            'households': [households],
            'median_income': [median_income],
            'ocean_proximity': [ocean_proximity],
            'county': [county],
            'index_right': [0]
        })
        
        try:
            # Predykcja
            prediction_log = model.predict(input_data)
            prediction = np.expm1(prediction_log[0])  # Konwersja z log1p scale (odwrotność log1p)
            
            # Wyświetlenie wyniku
            st.success(f"💰 Przewidywana cena: **${prediction:,.0f}**")
            
            # Dodatkowe statystyki
            col1, col2, col3 = st.columns(3)
            with col1:
                rooms_per_household = total_rooms / households
                st.metric("Pokoje/gospodarstwo", f"{rooms_per_household:.1f}")
            
            with col2:
                bedrooms_per_room = total_bedrooms / total_rooms
                st.metric("Sypialnie/pokoje", f"{bedrooms_per_room:.2f}")
            
            with col3:
                population_per_household = population / households
                st.metric("Osoby/gospodarstwo", f"{population_per_household:.1f}")
            
        except Exception as e:
            st.error(f"Błąd podczas predykcji: {str(e)}")

def create_batch_prediction_interface(model):
    """Tworzy interfejs do predykcji wsadowej"""
    st.subheader("📂 Predykcja wsadowa z CSV")
    
    uploaded_file = st.file_uploader(
        "Prześlij plik CSV z danymi",
        type=['csv'],
        help="Plik powinien zawierać kolumny: housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity. Opcjonalnie: county"
    )
    
    if uploaded_file is not None:
        try:
            # Wczytaj dane
            df = pd.read_csv(uploaded_file)
            
            st.info(f"Wczytano {len(df)} wierszy danych")
            
            # Walidacja danych
            is_valid, message = validate_csv_data(df)
            
            if not is_valid:
                st.error(message)
                return
            
            st.success(message)
            
            # Przetwórz dane
            df_processed = process_uploaded_data(df.copy())
            
            # Pokaż próbkę danych
            st.subheader("Podgląd danych:")
            st.dataframe(df_processed.head())
            
            # Przycisk do predykcji
            if st.button("🚀 Wykonaj predykcje dla wszystkich wierszy", type="primary"):
                with st.spinner("Wykonywanie predykcji..."):
                    try:
                        # Przygotuj dane do predykcji
                        prediction_columns = ['housing_median_age', 'total_rooms', 'total_bedrooms', 
                                            'population', 'households', 'median_income', 
                                            'ocean_proximity', 'county']
                        
                        X_pred = df_processed[prediction_columns]
                        
                        # Wykonaj predykcje
                        predictions_log = model.predict(X_pred)
                        predictions = np.expm1(predictions_log)  # Konwersja z log1p scale
                        
                        # Dodaj predykcje do DataFrame
                        df_with_predictions = df_processed.copy()
                        df_with_predictions['predicted_price'] = predictions
                        
                        # Wyświetl wyniki
                        st.success(f"✅ Wykonano predykcje dla {len(predictions)} wierszy")
                        
                        # Statystyki predykcji
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Średnia cena", f"${predictions.mean():,.0f}")
                        with col2:
                            st.metric("Mediana ceny", f"${np.median(predictions):,.0f}")
                        with col3:
                            st.metric("Min cena", f"${predictions.min():,.0f}")
                        with col4:
                            st.metric("Max cena", f"${predictions.max():,.0f}")
                        
                        # Wyświetl tabelę z wynikami
                        st.subheader("Wyniki predykcji:")
                        st.dataframe(df_with_predictions)
                        
                        # Przycisk do pobrania wyników
                        csv = df_with_predictions.to_csv(index=False)
                        st.download_button(
                            label="💾 Pobierz wyniki jako CSV",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                        
                        # Wykres rozkładu predykcji
                        fig = px.histogram(
                            df_with_predictions,
                            x='predicted_price',
                            nbins=50,
                            title="Rozkład przewidywanych cen"
                        )
                        fig.update_layout(xaxis_title="Przewidywana cena ($)", yaxis_title="Liczba")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Błąd podczas wykonywania predykcji: {str(e)}")
                        
        except Exception as e:
            st.error(f"Błąd podczas wczytywania pliku: {str(e)}")

def main():
    # Tytuł aplikacji
    st.title("🏠 California Housing Price Predictor")
    st.markdown("---")
    
    # Sidebar z wyborem modelu
    st.sidebar.header("🤖 Wybór modelu")
    
    # Dodaj opcję URL-i do modeli
    st.sidebar.markdown("**Konfiguracja URL-i modeli:**")
    model_urls = {}
    for model_name in MODEL_URLS.keys():
        url = st.sidebar.text_input(
            f"URL {model_name}",
            value=MODEL_URLS[model_name],
            help=f"URL do pliku {model_name} na Google Drive"
        )
        model_urls[model_name] = url
    
    # Wybór modelu
    selected_model_name = st.sidebar.selectbox(
        "Wybierz model",
        options=list(model_urls.keys()),
        index=0
    )
    
    # Przycisk do załadowania modelu
    if st.sidebar.button("📥 Załaduj model"):
        if model_urls[selected_model_name] and model_urls[selected_model_name] != "https://drive.google.com/uc?id=YOUR_XGBOOST_FILE_ID":
            model, model_name = load_model_from_url(model_urls[selected_model_name], selected_model_name)
            if model:
                st.session_state['model'] = model
                st.session_state['model_name'] = model_name
                st.sidebar.success(f"✅ Model {model_name} załadowany pomyślnie!")
        else:
            st.sidebar.error("Proszę wprowadzić prawidłowy URL do modelu")
    
    # Sprawdź czy model jest załadowany
    if 'model' not in st.session_state:
        st.warning("⚠️ Proszę załadować model z sidebar")
        st.info("1. Wprowadź URL-e do modeli w sidebar\n2. Wybierz model\n3. Kliknij 'Załaduj model'")
        return
    
    # Informacje o załadowanym modelu
    st.sidebar.info(f"Aktualny model: {st.session_state['model_name']}")
    
    # Główna sekcja - tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔮 Predykcja", "📊 Analiza danych", "🗺️ Mapa", "📈 Ewaluacja", "ℹ️ Pomoc"])
    
    with tab1:
        st.header("Przewidywanie ceny nieruchomości")
        
        # Wybór trybu predykcji
        prediction_mode = st.radio(
            "Wybierz tryb predykcji:",
            ["Pojedyncza predykcja", "Predykcja wsadowa (CSV)"],
            horizontal=True
        )
        
        if prediction_mode == "Pojedyncza predykcja":
            create_prediction_interface(st.session_state['model'])
        else:
            create_batch_prediction_interface(st.session_state['model'])
    
    with tab2:
        st.header("📊 Analiza danych")
        
        # Ładowanie danych
        df = load_data()
        if df is not None:
            # Podstawowe statystyki
            st.subheader("Podstawowe statystyki")
            st.dataframe(df.describe())
            
            # Wykresy
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram cen
                fig = px.histogram(
                    df, 
                    x='median_house_value', 
                    nbins=50,
                    title='Rozkład cen nieruchomości'
                )
                fig.update_layout(xaxis_title="Cena ($)", yaxis_title="Liczba")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Scatter plot - dochód vs cena
                fig = px.scatter(
                    df.sample(n=min(1000, len(df))), 
                    x='median_income', 
                    y='median_house_value',
                    color='ocean_proximity',
                    title='Dochód vs Cena nieruchomości'
                )
                fig.update_layout(xaxis_title="Mediana dochodu (10k USD)", yaxis_title="Cena ($)")
                st.plotly_chart(fig, use_container_width=True)
            
            # Korelacje
            st.subheader("Macierz korelacji")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Korelacje między zmiennymi"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Brak danych do analizy. Prześlij plik CSV w zakładce 'Predykcja'.")
    
    with tab3:
        st.header("🗺️ Mapa Kalifornii")
        
        df = load_data()
        if df is not None:
            # Sample danych dla wydajności
            df_sample = df.sample(n=min(1000, len(df)))
            
            # Mapa z Folium
            m = folium.Map(location=[36.7783, -119.4179], zoom_start=6)
            
            # Dodanie punktów
            for idx, row in df_sample.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=3,
                    popup=f"Cena: ${row['median_house_value']:,.0f}",
                    color='red' if row['median_house_value'] > 300000 else 'blue',
                    fill=True,
                    opacity=0.7
                ).add_to(m)
            
            # Legenda
            st.markdown("🔴 Drogie nieruchomości (>$300k) | 🔵 Tańsze nieruchomości (<$300k)")
            folium_static(m)
        else:
            st.info("Brak danych do wyświetlenia na mapie.")
    
    with tab4:
        st.header("📈 Ewaluacja modelu")
        
        df = load_data()
        if df is not None:
            if st.button("Przeprowadź ewaluację na danych testowych"):
                with st.spinner("Przygotowywanie danych i ewaluacja..."):
                    try:
                        # Przygotowanie danych (uproszczona wersja)
                        df_clean = df.dropna()
                        
                        # Jeśli nie mamy kolumny county, dodajemy przykładową
                        if 'county' not in df_clean.columns:
                            df_clean['county'] = 'Los Angeles'  # Domyślne hrabstwo
                        
                        X_eval = df_clean[['housing_median_age', 'total_rooms', 'total_bedrooms', 
                                         'population', 'households', 'median_income', 
                                         'ocean_proximity', 'county']]
                        y_eval = df_clean['median_house_value']
                        
                        # Predykcja
                        y_pred_log = st.session_state['model'].predict(X_eval)
                        y_pred = np.expm1(y_pred_log)  # Konwersja z log1p scale
                        
                        # Metryki
                        mae = mean_absolute_error(y_eval, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_eval, y_pred))
                        r2 = r2_score(y_eval, y_pred)
                        
                        # Wyświetlenie metryk
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("MAE", f"${mae:,.0f}")
                        with col2:
                            st.metric("RMSE", f"${rmse:,.0f}")
                        with col3:
                            st.metric("R²", f"{r2:.3f}")
                        
                        # Wykres predykcji vs rzeczywiste wartości
                        fig = px.scatter(
                            x=y_eval, 
                            y=y_pred,
                            title="Predykcje vs Rzeczywiste wartości",
                            labels={'x': 'Rzeczywiste ceny ($)', 'y': 'Przewidywane ceny ($)'}
                        )
                        
                        # Dodaj linię idealnej predykcji
                        min_val = min(y_eval.min(), y_pred.min())
                        max_val = max(y_eval.max(), y_pred.max())
                        fig.add_shape(
                            type="line",
                            x0=min_val, y0=min_val,
                            x1=max_val, y1=max_val,
                            line=dict(color="red", dash="dash"),
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Rozkład błędów
                        errors = y_pred - y_eval
                        fig2 = px.histogram(
                            x=errors,
                            nbins=50,
                            title="Rozkład błędów predykcji"
                        )
                        fig2.update_layout(xaxis_title="Błąd predykcji ($)", yaxis_title="Częstość")
                        st.plotly_chart(fig2, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Błąd podczas ewaluacji: {str(e)}")
                        st.info("Upewnij się, że model został prawidłowo wytrenowany i zapisany.")
        else:
            st.info("Brak danych do ewaluacji.")
    
    with tab5:
        st.header("ℹ️ Pomoc")
        
        st.markdown("""
        ### Jak używać aplikacji:
        
        #### 1. Konfiguracja modelu
        1. W sidebar wprowadź URL-e do modeli na Google Drive
        2. Wybierz model do użycia
        3. Kliknij "Załaduj model"
        
        #### 2. Przewidywanie cen
        **Pojedyncza predykcja:**
        - Wypełnij formularz z parametrami nieruchomości
        - Kliknij "Przewiduj cenę"
        
        **Predykcja wsadowa:**
        - Prześlij plik CSV z danymi
        - Plik musi zawierać kolumny: `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `ocean_proximity`
        - Opcjonalnie: `county`
        
        #### 3. Format pliku CSV
        ```
        housing_median_age,total_rooms,total_bedrooms,population,households,median_income,ocean_proximity,county
        29,5000,1000,3500,700,5.0,<1H OCEAN,Los Angeles
        35,4500,900,3200,650,4.5,INLAND,Orange
        ```
        
        #### 4. Wartości kategoryczne
        **ocean_proximity:** `<1H OCEAN`, `INLAND`, `ISLAND`, `NEAR BAY`, `NEAR OCEAN`
        
        **county:** Lista wszystkich hrabstw Kalifornii (Los Angeles, Orange, San Diego, itd.)
        
        #### 5. Uzyskiwanie URL-i z Google Drive
        - Model przewiduje wartości w skali logarytmicznej (log1p)
        - Aplikacja automatycznie konwertuje wyniki za pomocą `np.expm1()` (odwrotność log1p)
        - Wszystkie ceny są wyświetlane w dolarach amerykańskich
        1. Prześlij plik .pkl na Google Drive
        2. Ustaw uprawnienia na "Każdy z linkiem"
        3. Skopiuj ID pliku z URL
        4. Użyj formatu: `https://drive.google.com/uc?id=YOUR_FILE_ID`
        """)

if __name__ == "__main__":
    main()