# -*- coding: utf-8 -*-
import os
import sys
import warnings
import joblib
from sklearn.model_selection   import train_test_split
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics           import accuracy_score
from sklearn.pipeline          import Pipeline
from sklearn.preprocessing     import StandardScaler

# Fix Windows console encoding so all print() calls work without crashing
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Suppress ConvergenceWarning for clean output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*converge.*")

# Make sure Python can find preprocess.py in the same folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import (
    load_and_clean, compute_team_stats, save_team_stats,
    add_features, encode, save_encoders,
    get_features_and_target, save_processed_csv, MODELS_DIR
)

# -------------------------------------------------------------
#  THREE MODELS TO COMPARE
#  - Logistic Regression : Pipeline with StandardScaler (needed
#    for convergence on mixed-scale features)
#  - Random Forest       : More trees + constrained depth to
#    reduce overfitting on the small dataset
#  - Gradient Boosting   : Shallower trees + lower learning rate
#    (slower but more accurate generalisation)
# -------------------------------------------------------------
MODELS = {
    "Logistic Regression" : Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(max_iter=3000, random_state=42)),
    ]),
    "Random Forest"       : RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
    ),
    "Gradient Boosting"   : GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.85,
        random_state=42,
    ),
}


def train():
    # -- 1. Load, compute stats, add features, encode ----------
    df_clean    = load_and_clean()
    stats       = compute_team_stats(df_clean)   # compute win rates & h2h
    save_team_stats()                            # save for predict.py
    df_featured = add_features(df_clean, stats)  # add 5 numeric features
    save_processed_csv(df_featured)              # saves data/processed.csv
    df_encoded  = encode(df_featured, fit=True)  # fits label encoders
    save_encoders()                              # saves to models/

    # -- 2. Separate features and target --------------------------
    X, y = get_features_and_target(df_encoded)

    # -- 3. 80 / 20 train-test split ------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n[INFO] Train rows : {len(X_train)}")
    print(f"[INFO] Test  rows : {len(X_test)}")

    # -- 4. Train every model & record accuracy -------------------
    results        = {}
    trained_models = {}

    print("\n-- Training all 3 models --")
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)

        results[name]        = acc
        trained_models[name] = model

        bar = "#" * int(acc * 30)
        print(f"  {name:<25}  {acc*100:5.1f}%  {bar}")

    # -- 5. Pick the best model -----------------------------------
    best_name  = max(results, key=results.get)
    best_model = trained_models[best_name]

    print(f"\n[BEST] '{best_name}'  with  {results[best_name]*100:.1f}% accuracy")

    # -- 6. Save best model to disk -------------------------------
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"[INFO] Best model saved  ->  {model_path}")

    # Also save all three (useful for evaluate.py)
    all_path = os.path.join(MODELS_DIR, "all_models.pkl")
    joblib.dump(trained_models, all_path)
    print(f"[INFO] All models saved  ->  {all_path}")

    # Save test data so evaluate.py can use the same split
    test_path = os.path.join(MODELS_DIR, "test_data.pkl")
    joblib.dump({"X_test": X_test, "y_test": y_test}, test_path)
    print(f"[INFO] Test data saved   ->  {test_path}")

    return trained_models, X_test, y_test


# -------------------------------------------------------------
#  Run  ->  python src/train.py
# -------------------------------------------------------------
if __name__ == "__main__":
    train()
    print("\n[OK] train.py")