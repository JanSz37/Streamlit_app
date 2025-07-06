import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CustomFeaturesAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Oblicz mediany dla wszystkich kolumn numerycznych
        self.median_total_bedrooms = X["total_bedrooms"].median()
        self.median_values = {}
        
        # Zapisz mediany dla wszystkich kolumn numerycznych
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.median_values[col] = X[col].median()
        
        return self

    def transform(self, X):
        X = X.copy()
        
        # Wypełnij brakujące wartości medianami
        for col, median_val in self.median_values.items():
            if col in X.columns:
                X[col] = X[col].fillna(median_val)
        
        # Dodaj nowe cechy
        X["rooms_per_household"] = X["total_rooms"] / X["households"]
        X["bedrooms_per_room"] = X["total_bedrooms"] / X["total_rooms"]
        X["population_per_household"] = X["population"] / X["households"]
        
        # Obsłuż nieskończone wartości i NaN powstałe z dzielenia przez zero
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Wypełnij pozostałe NaN wartościami 0 lub medianami
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
        
        # Obsłuż kategorialne kolumny z NaN
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if X[col].isna().any():
                X[col] = X[col].fillna('Unknown')
        
        return X