"""
Entraîne un modèle GradientBoosting + calibration MAPIE sur le dataset Ames Housing.
- Exclut les colonnes d'identifiant 'Order' et 'PID' des features.
- Utilise des chemins robustes pour les imports et les données.
Usage:
    # A exécuter depuis la racine du projet (Conformal_Prediction)
    python backend/scripts/train_ames_gradient.py
"""
from backend.app.conformal import extract_lower_upper as extract_lower_upper_from_mapie
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
from pathlib import Path

# Ajoute la racine du projet au PYTHONPATH pour des imports robustes
# monte de 3 niveaux: scripts -> backend -> Conformal_Prediction
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# Import de la fonction centralisée (chemin absolu depuis la racine du projet)

# Import MAPIE de façon résiliente
try:
    from mapie.regression import MapieRegressor
except Exception:
    try:
        from mapie.regression.mapie_regressor import MapieRegressor
    except Exception as exc:
        raise ImportError(
            "Impossible d'importer MapieRegressor depuis le paquet 'mapie'. "
            "Vérifie que 'mapie' est installé et compatible. "
            f"Détails: {exc}"
        )

# Chemins définis de manière absolue depuis la racine du projet
DATA_PATH = PROJECT_ROOT / "backend" / "data" / "ames.csv"
MODELS_DIR = PROJECT_ROOT / "backend" / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_ames(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV Ames non trouvé : {csv_path}. Place le fichier et relance.")
    df = pd.read_csv(csv_path)
    return df


def make_onehot_encoder_compat():
    """
    Retourne un OneHotEncoder compatible avec la version installée de scikit-learn.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def build_pipeline_and_types(df: pd.DataFrame, exclude_ids: list = None):
    """
    Construit la pipeline et renvoie (pipeline, feature_cols, feature_types)
    """
    if exclude_ids is None:
        exclude_ids = ["Order", "PID"]

    df_cols = df.columns.tolist()
    cols_to_drop = [c for c in exclude_ids if c in df_cols]
    if cols_to_drop:
        print(f"Excluding ID columns from features: {cols_to_drop}")

    X = df.drop(columns=["SalePrice"] + cols_to_drop, errors="ignore")
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(
        include=["object", "category"]).columns.tolist()

    num_tf = SimpleImputer(strategy="median")
    cat_tf = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("ohe", make_onehot_encoder_compat())
    ])

    preproc = ColumnTransformer(
        transformers=[
            ("num", num_tf, numeric_cols),
            ("cat", cat_tf, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0
    )

    # Utilisation du GradientBoostingRegressor
    gb = GradientBoostingRegressor(
        n_estimators=200, random_state=42, learning_rate=0.1, max_depth=3)
    pipeline = Pipeline(steps=[("preproc", preproc), ("model", gb)])

    feature_cols = numeric_cols + categorical_cols
    feature_types = {
        c: ("numeric" if c in numeric_cols else "categorical") for c in feature_cols}

    return pipeline, feature_cols, feature_types


def main():
    print("Chargement du dataset Ames...")
    df = load_ames(DATA_PATH)

    if "SalePrice" not in df.columns:
        raise ValueError("Le CSV doit contenir la colonne cible 'SalePrice'.")

    pipeline, feature_cols, feature_types = build_pipeline_and_types(
        df, exclude_ids=["Order", "PID"])

    X = df.drop(columns=["SalePrice"] + [c for c in ("Order",
                "PID") if c in df.columns], errors="ignore")
    y = df["SalePrice"]

    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print("Entraînement du pipeline (préproc + GradientBoosting)...")
    pipeline.fit(X_train, y_train)

    print("Calibration MAPIE (cv='prefit', method='plus') sur l'ensemble de calibration...")
    mapie = MapieRegressor(pipeline, cv="prefit", method="plus")
    mapie.fit(X_cal, y_cal)

    y_pred_cal, y_pis_cal = mapie.predict(X_cal, alpha=0.05)
    lower, upper = extract_lower_upper_from_mapie(y_pred_cal, y_pis_cal)
    coverage = float(np.mean((y_cal.values >= lower)
                     & (y_cal.values <= upper)))
    print(f"Couverture empirique sur calibration (alpha=0.05): {coverage:.3f}")

    saved = {
        "model": mapie,
        "feature_names": feature_cols,
        "target_name": "SalePrice",
        "alpha_default": 0.05,
        "feature_types": feature_types,
    }
    model_path = MODELS_DIR / "ames_gb_mapie.joblib"
    joblib.dump(saved, model_path)
    print(f"Modèle sauvegardé: {model_path}")

    print("\nRésumé :")
    print(f" - Nombre de features (originales) : {len(feature_cols)}")
    print(f" - Exemples de features : {feature_cols[:5]}")
    print(f" - Chemin modèle : {model_path}")


if __name__ == "__main__":
    main()
