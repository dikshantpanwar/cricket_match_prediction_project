# -*- coding: utf-8 -*-
import os
import sys
import joblib
import pandas as pd

# Fix Windows console encoding so all print() calls work without crashing
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import (
    load_encoders, load_team_stats,
    MODELS_DIR, FEATURE_COLS, CAT_COLS, HOME_VENUES
)

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")


def _encode_input(input_df):
    """
    Encode only the categorical columns using saved LabelEncoders.
    h2h_diff and home_adv are already numeric -- they are left as-is.
    """
    from preprocess import encoders
    enc = input_df.copy()
    for col in CAT_COLS:          # <-- only categorical cols, NOT h2h_diff / home_adv
        le  = encoders[col]
        val = str(enc.at[0, col])
        enc[col] = int(le.transform([val])[0]) if val in le.classes_ else -1
    return enc


def _build_input_row(team1, team2, toss_winner, toss_decision, venue):
    """
    Build the raw one-row DataFrame with all FEATURE_COLS,
    computing numeric features from the saved team_stats.

    team_stats is loaded from models/team_stats.pkl by load_model(),
    guaranteeing we use the exact same values as during training.
    """
    from preprocess import team_stats

    win_rates = team_stats.get("win_rates", {})
    h2h_wins  = team_stats.get("h2h_wins",  {})
    h2h_total = team_stats.get("h2h_total", {})

    # Overall win rates (default 0.5 = unknown team)
    t1_wr = win_rates.get(team1, 0.5)
    t2_wr = win_rates.get(team2, 0.5)

    # Head-to-head win rate for team1 against team2
    total = h2h_total.get((team1, team2), 0)
    wins  = h2h_wins.get((team1, team2), 0)
    h2h_wr = wins / total if total > 0 else 0.5

    # Home advantage flag
    home_adv = 1 if HOME_VENUES.get(team1) == venue else 0

    return pd.DataFrame([{
        "team1"          : team1,
        "team2"          : team2,
        "toss_winner"    : toss_winner,
        "toss_decision"  : toss_decision,
        "venue"          : venue,
        "team1_win_rate" : t1_wr,
        "team2_win_rate" : t2_wr,
        "win_rate_diff"  : t1_wr - t2_wr,
        "h2h_win_rate"   : h2h_wr,
        "home_adv"       : home_adv,
    }])


def load_model():
    """Load the best model, label encoders, and team stats."""
    model = joblib.load(MODEL_PATH)
    load_encoders()
    load_team_stats()   # load win rates & h2h so _build_input_row works
    return model


def predict_winner(team1, team2, toss_winner, toss_decision, venue):
    """
    Predict the winner for a new match.

    Parameters
    ----------
    team1, team2   : str  e.g. "Mumbai Indians", "Chennai Super Kings"
    toss_winner    : str  must be team1 or team2
    toss_decision  : str  "bat" or "field"
    venue          : str  e.g. "Wankhede Stadium"

    Returns
    -------
    str -- predicted winning team name
    """
    model = load_model()
    from preprocess import encoders

    input_df     = _build_input_row(team1, team2, toss_winner, toss_decision, venue)
    input_enc    = _encode_input(input_df)
    pred_encoded = model.predict(input_enc[FEATURE_COLS])
    winner       = encoders["winner"].inverse_transform(pred_encoded)[0]
    return winner


def predict_with_proba(team1, team2, toss_winner, toss_decision, venue):
    """
    Same as predict_winner but also returns win probabilities for every team.

    Returns
    -------
    winner     : str
    proba_dict : dict  { team_name: probability% }  sorted descending
    """
    # Load model + encoders once -- no redundant second call
    model = load_model()
    from preprocess import encoders

    input_df  = _build_input_row(team1, team2, toss_winner, toss_decision, venue)
    input_enc = _encode_input(input_df)

    proba   = model.predict_proba(input_enc[FEATURE_COLS])[0]
    classes = encoders["winner"].inverse_transform(model.classes_)

    proba_dict = dict(sorted(
        zip(classes, proba * 100),
        key=lambda x: x[1], reverse=True
    ))
    winner = max(proba_dict, key=proba_dict.get)
    return winner, proba_dict


# -------------------------------------------------------------
#  Interactive demo  ->  python src/predict.py
# -------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  IPL MATCH WINNER PREDICTOR")
    print("="*55)

    # -- Change these values to predict any match ---------------
    team1         = "Mumbai Indians"
    team2         = "Chennai Super Kings"
    toss_winner   = "Mumbai Indians"
    toss_decision = "bat"
    venue         = "Wankhede Stadium"

    print(f"\n  Team 1        : {team1}")
    print(f"  Team 2        : {team2}")
    print(f"  Toss winner   : {toss_winner}")
    print(f"  Toss decision : {toss_decision}")
    print(f"  Venue         : {venue}")

    try:
        winner, proba_dict = predict_with_proba(
            team1, team2, toss_winner, toss_decision, venue
        )
        print(f"\n  [WIN] Predicted Winner  ->  {winner}\n")
        print("  Top 5 win probabilities:")
        for i, (team, prob) in enumerate(list(proba_dict.items())[:5], 1):
            bar = "#" * int(prob / 2)
            print(f"  {i}. {team:<32} {prob:5.1f}%  {bar}")

    except Exception as e:
        print(f"\n  [ERROR] {e}")
        print("  -> Run train.py first to generate the model files.")

    print("\n[OK] predict.py")