import json, joblib, pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

DATA = Path("data/loan.csv")
assert DATA.exists(), "data/loan.csv introuvable"

df = pd.read_csv(DATA)

# --- DÉTECTION CIBLE (tolérant à la casse) ---
candidates = [c for c in df.columns if c.lower() in ("loan_status","loanstatus","target","label")]
assert len(candidates) >= 1, "Aucune colonne target trouvée (ex: loan_status)"
target = candidates[0]

# --- NORMALISATION DE LA CIBLE (lower/strip + mapping robuste) ---
y_raw = df[target]
y_str = y_raw.astype(str).str.strip().str.lower()

map_lower = {
    "y": 1, "yes": 1, "approved": 1, "approve": 1, "1": 1, "true": 1,
    "n": 0, "no": 0, "rejected": 0, "reject": 0, "denied": 0, "declined": 0, "0": 0, "false": 0
}

y_mapped = y_str.map(map_lower)

# Si plus de deux classes (ex: 'pending', 'canceled'), on ne garde que approuvé/rejeté
mask_binary = y_mapped.isin([0, 1])
df = df.loc[mask_binary].copy()
y = y_mapped.loc[df.index].astype(int)

assert y.dropna().nunique() == 2, f"Cible non binaire. Valeurs vues (après normalisation): {sorted(y_str.unique()[:20])}"

# --- FEATURES ---
X = df.drop(columns=[target])

# Split colonnes
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

# Cast + imput minimal (safe)
for c in num_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce")
X[num_cols] = X[num_cols].fillna(X[num_cols].median())
for c in cat_cols:
    X[c] = X[c].fillna("Unknown").astype(str)

# --- SPLIT ---
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- PIPELINE ---
pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", drop=None), cat_cols),
    ("num", "passthrough", num_cols)
])

clf = LogisticRegression(max_iter=1000, class_weight="balanced")

pipe = Pipeline([
    ("pre", pre),
    ("clf", clf)
])

pipe.fit(X_tr, y_tr)
proba = pipe.predict_proba(X_te)[:, 1]
pred  = (proba >= 0.5).astype(int)

acc = accuracy_score(y_te, pred)
f1  = f1_score(y_te, pred)
try:
    auc = roc_auc_score(y_te, proba)
except:
    auc = np.nan

print(f"ACC: {acc:.3f} | F1: {f1:.3f} | ROC-AUC: {auc:.3f}")
print("\nReport:\n", classification_report(y_te, pred, digits=3))

# --- SAVE ---
Path("app").mkdir(exist_ok=True)
joblib.dump(pipe, "app/model.pkl")
with open("app/features.json","w") as f:
    json.dump({"num": num_cols, "cat": cat_cols, "target": target}, f)

print("Saved -> app/model.pkl | app/features.json")
