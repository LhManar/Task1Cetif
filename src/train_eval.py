"""Nettoyage, découpage, baseline, modèle TF-IDF + Régression Logistique, évaluation."""
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

LABELS = ["billing", "technical", "account", "general"]

# ---------- 1. Chargement et nettoyage ----------
df = pd.read_csv("data/tickets.csv")
df = df.dropna().drop_duplicates(subset="text")
df["text"] = df["text"].str.strip()
df["clean"] = (df["text"].str.lower()
               .apply(lambda s: re.sub(r"[^a-z0-9\s']", " ", s))
               .apply(lambda s: re.sub(r"\s+", " ", s).strip()))
print("Tickets après nettoyage :", len(df))

# ---------- 2. Découpage stratifié 70/15/15 ----------
train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=42)
for name, part in [("train", train), ("val", val), ("test", test)]:
    part.to_csv(f"data/{name}.csv", index=False)
print({"train": len(train), "val": len(val), "test": len(test)})

# ---------- 3. Baseline : règles par mots-clés ----------
KEYWORDS = {
    "billing": ["refund", "invoice", "charge", "charged", "payment", "billed", "card", "discount code"],
    "technical": ["crash", "error", "slow", "loading", "broken", "freez", "not working", "black screen"],
    "account": ["password", "login", "log in", "username", "account", "profile", "two-factor"],
}

def keyword_predict(text: str) -> str:
    scores = {lab: sum(k in text for k in kws) for lab, kws in KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"

base_pred = test["clean"].apply(keyword_predict)
base_f1 = f1_score(test["label"], base_pred, average="macro")
base_acc = accuracy_score(test["label"], base_pred)

# ---------- 4. TF-IDF + Régression Logistique ----------
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
    LogisticRegression(max_iter=1000, C=5.0),
)
model.fit(train["clean"], train["label"])

val_f1 = f1_score(val["label"], model.predict(val["clean"]), average="macro")
pred = model.predict(test["clean"])
acc = accuracy_score(test["label"], pred)
macro_f1 = f1_score(test["label"], pred, average="macro")
report = classification_report(test["label"], pred, labels=LABELS, output_dict=True, zero_division=0)

# Validation croisée 5 plis sur train+val
trval = pd.concat([train, val])
cv = cross_val_score(model, trval["clean"], trval["label"],
                     cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="f1_macro")

# ---------- 5. Gestion de la confiance ----------
proba = model.predict_proba(test["clean"])
conf = proba.max(axis=1)
THRESH = 0.6
confident = conf >= THRESH
acc_conf = accuracy_score(test["label"][confident], pred[confident]) if confident.any() else float("nan")

# ---------- 6. Matrice de confusion ----------
cm = confusion_matrix(test["label"], pred, labels=LABELS)
cm_df = pd.DataFrame(cm, index=[f"vrai_{l}" for l in LABELS],
                     columns=[f"pred_{l}" for l in LABELS])
print(cm_df)
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(4)); ax.set_xticklabels(LABELS, rotation=30)
ax.set_yticks(range(4)); ax.set_yticklabels(LABELS)
for i in range(4):
    for j in range(4):
        ax.text(j, i, cm[i, j], ha="center", va="center")

ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (test)")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)
plt.show()

# ---------- 7. Analyse d'erreurs ----------
errors = test.assign(predicted=pred, confidence=conf.round(2))
errors = errors[errors["label"] != errors["predicted"]][["text", "label", "predicted", "confidence"]]
errors.to_csv("results/errors.csv", index=False)

# ---------- 8. Robustesse ----------
robust = {
    "no_punctuation": test["text"].str.replace(r"[^\w\s]", "", regex=True).str.lower(),
    "uppercase_input": test["text"].str.upper().str.lower(),
}
robust_f1 = {k: f1_score(test["label"], model.predict(v), average="macro") for k, v in robust.items()}

# ---------- 9. Mots les plus importants par classe (explicabilité) ----------
vec, clf = model.named_steps["tfidfvectorizer"], model.named_steps["logisticregression"]
names = np.array(vec.get_feature_names_out())
top_words = {lab: names[np.argsort(clf.coef_[i])[-6:][::-1]].tolist() for i, lab in enumerate(clf.classes_)}

summary = {
    "baseline_keywords": {"accuracy": round(base_acc, 3), "macro_f1": round(base_f1, 3)},
    "tfidf_logreg": {"accuracy": round(acc, 3), "macro_f1": round(macro_f1, 3), "val_macro_f1": round(val_f1, 3),
                     "cv5_macro_f1_mean": round(cv.mean(), 3), "cv5_macro_f1_std": round(cv.std(), 3),
                     "recall_per_class": {l: round(report[l]["recall"], 3) for l in LABELS}},
    "confidence": {"threshold": THRESH, "share_confident": round(float(confident.mean()), 3),
                   "accuracy_on_confident": round(float(acc_conf), 3)},
    "robustness_macro_f1": {k: round(v, 3) for k, v in robust_f1.items()},
    "top_words": top_words,
    "n_errors": int(len(errors)),
}
json.dump(summary, open("results/metrics.json", "w"), indent=2)
print(json.dumps(summary, indent=2))

# ---------- 10. Prédiction d'un ticket ----------
def predict_ticket(text: str):
    clean = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s']", " ", text.lower())).strip()
    p = model.predict_proba([clean])[0]
    i = int(p.argmax())
    return model.classes_[i], float(p[i]), ("REVIEW" if p[i] < THRESH else "AUTO")

if __name__ == "__main__":
    for t in ["I was billed twice, please refund me", "The app crashes on startup", "Can't remember my password"]:
        print(t, "->", predict_ticket(t))
