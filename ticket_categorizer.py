"""
Auto Email / Ticket Categorizer
--------------------------------
A lightweight NLP classifier that reads an incoming support ticket
(subject + body) and routes it to the correct department in real time:
Billing, Technical, HR, or General.

Pipeline:
    raw text -> cleaning -> TF-IDF vectorization -> classifier
    -> category + confidence score -> priority tag -> routing decision

Run:
    python ticket_categorizer.py
"""

import re
import string
import warnings

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

DATA_PATH = "data/tickets.csv"
REVIEW_THRESHOLD = 0.60          # bonus: confidence below this -> human review
RANDOM_STATE = 42

# A small custom stopword list. We avoid nltk's full stopword corpus so the
# script has zero extra downloads and stays fast/portable for a real-time
# service, while still stripping the most common low-signal English words.
STOPWORDS = set("""
a an the is are was were be been being to of in on at for with and or
but if then so as this that these those it its i you he she we they
my your his her our their me him them us do does did have has had
will would can could should not no nor just very
""".split())

# Simple keyword rules for the bonus "priority" tag. Keyword rules are
# transparent and auditable -- important for a triage layer where a wrong
# "urgent" miss/overreaction has real operational cost.
URGENT_KEYWORDS = {
    "urgent", "down", "not working", "crash", "crashed", "crashing",
    "asap", "immediately", "critical", "emergency", "broken", "failed",
    "outage", "production", "unauthorized", "complaint",
}


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/digits/extra whitespace, drop stopwords."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # urls
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)                      # digits
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)


def tag_priority(raw_text: str) -> str:
    """Rule-based urgency tag, independent of the ML category prediction."""
    lowered = raw_text.lower()
    return "URGENT" if any(k in lowered for k in URGENT_KEYWORDS) else "NORMAL"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna(""))
    df["clean_text"] = df["text"].apply(clean_text)
    return df


def train_and_select_model(df: pd.DataFrame):
    """Train Naive Bayes and Logistic Regression, evaluate both, keep the
    better one. This is done instead of picking one blindly, because the
    right choice depends on how the two behave on THIS dataset."""

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["category"],
        test_size=0.25, random_state=RANDOM_STATE, stratify=df["category"],
    )

    # TF-IDF over Bag-of-Words: raw word counts let long tickets or common
    # words dominate the vector regardless of how informative they are.
    # TF-IDF down-weights words that are frequent across ALL tickets
    # (e.g. "please", "account") and up-weights words that are frequent
    # in a ticket but rare overall (e.g. "invoice", "crash", "resign") --
    # exactly the words that actually separate one department from another.
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1, max_df=0.9)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    candidates = {
        # Naive Bayes: the classic baseline for text classification. Assumes
        # word occurrences are conditionally independent given the class --
        # a false assumption in general, but a very cheap and effective one
        # for bag-of-words text, especially on small datasets like this one.
        # alpha (Laplace smoothing) is lowered from sklearn's default of 1.0.
        # With only ~40 training rows across 4 classes, default smoothing
        # over-flattens the probabilities (every ticket lands near 25-45%
        # confidence regardless of how clear-cut it is), which makes the
        # confidence score useless for triage. alpha=0.1 lets the model
        # commit harder when a ticket's vocabulary clearly matches one
        # class, while still spreading probability across classes for a
        # genuinely ambiguous ticket -- which is what the confidence score
        # is supposed to reflect.
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),

        # Logistic Regression: models a weighted combination of TF-IDF
        # features directly, tends to be more robust when categories share
        # vocabulary (e.g. "account", "please") and usually gives better
        # calibrated probabilities, which matters for the confidence score.
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=5.0, random_state=RANDOM_STATE
        ),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        results[name] = {
            "model": model,
            "accuracy": acc,
            "preds": preds,
            "y_test": y_test,
        }

    best_name = max(results, key=lambda n: results[n]["accuracy"])
    return vectorizer, results, best_name, (X_train_vec, X_test_vec, y_train, y_test)


def print_evaluation(results, best_name):
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    for name, r in results.items():
        marker = "  <-- selected" if name == best_name else ""
        print(f"\n{name}{marker}")
        print(f"Accuracy: {r['accuracy']:.2%}")
        print("Classification report:")
        print(classification_report(r["y_test"], r["preds"], zero_division=0))
        print("Confusion matrix (rows=actual, cols=predicted):")
        labels = sorted(r["y_test"].unique())
        cm = confusion_matrix(r["y_test"], r["preds"], labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        print(cm_df.to_string())


def predict_ticket(subject: str, body: str, vectorizer, model):
    """Real-time inference for ONE incoming ticket: cleans it, vectorizes it
    with the already-fitted vectorizer, and returns category + confidence +
    priority + routing decision. This is the function a live queue would
    call per message -- it does no retraining and is fast (milliseconds)."""

    raw_text = f"{subject} {body}"
    cleaned = clean_text(raw_text)
    vec = vectorizer.transform([cleaned])

    probs = model.predict_proba(vec)[0]
    classes = model.classes_
    best_idx = probs.argmax()
    category = classes[best_idx]
    confidence = probs[best_idx]

    # Edge case handling: if the model isn't confident about ANY category,
    # don't force a label -- flag it for a human instead of guessing.
    needs_review = confidence < REVIEW_THRESHOLD
    routed_category = "NEEDS_HUMAN_REVIEW" if needs_review else category

    priority = tag_priority(raw_text)

    return {
        "subject": subject,
        "predicted_category": category,
        "confidence": round(float(confidence), 3),
        "routed_to": routed_category,
        "priority": priority,
        "all_scores": {c: round(float(p), 3) for c, p in zip(classes, probs)},
    }


def demo_on_new_tickets(vectorizer, model):
    """Bonus requirement: predict category for at least 5 new sample
    tickets, written specifically to probe normal cases AND edge cases."""

    new_tickets = [
        # Clear-cut cases
        ("Need refund for cancelled plan",
         "I cancelled my plan last week but was still charged full price, please refund me urgently."),
        ("App is completely broken",
         "The app is not working at all since this morning, it crashes on launch every single time."),
        ("Question about maternity leave",
         "Could you tell me how many weeks of maternity leave the company policy allows?"),
        ("Do you have an office in Delhi",
         "I wanted to know if your company has a physical office in Delhi that I could visit."),
        ("Server outage affecting production",
         "Our production server is down and returning 500 errors, this is critical and urgent."),
        # Deliberately ambiguous / edge cases -- these should trip the
        # "needs human review" fallback rather than being force-labeled.
        ("Need help with my account",
         "I have an issue with my account and need someone to look into it as soon as possible."),
        ("Quick question",
         "Hi, just wondering about something, can someone get back to me please."),
        ("Just checking in",
         "Just checking in, nothing specific to report."),
    ]

    print("\n" + "=" * 60)
    print("LIVE ROUTING PREVIEW: 5+ NEW / UNSEEN TICKETS")
    print("=" * 60)
    for subject, body in new_tickets:
        result = predict_ticket(subject, body, vectorizer, model)
        print(f"\nSubject : {result['subject']}")
        print(f"Category: {result['predicted_category']}  "
              f"(confidence {result['confidence']:.0%})")
        print(f"Routed to : {result['routed_to']}")
        print(f"Priority  : {result['priority']}")
        print(f"All scores: {result['all_scores']}")


def train_final_model(df: pd.DataFrame, model_name: str):
    """Once a model type is chosen using the held-out test set, refit BOTH
    the vectorizer and the model on the FULL labeled dataset before it goes
    live. Holding back 25% of only 56 rows for the final deployed model
    would throw away real signal for no reason -- the held-out split's only
    job was to pick between model types honestly."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1, max_df=0.9)
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["category"]

    model = (
        MultinomialNB(alpha=0.1)
        if model_name == "Multinomial Naive Bayes"
        else LogisticRegression(max_iter=1000, C=5.0, random_state=RANDOM_STATE)
    )
    model.fit(X, y)
    return vectorizer, model


def main():
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} labeled tickets across categories: "
          f"{sorted(df['category'].unique())}")

    _, results, best_name, _ = train_and_select_model(df)
    print_evaluation(results, best_name)
    print(f"\nSelected model for deployment: {best_name} "
          f"(held-out accuracy {results[best_name]['accuracy']:.2%})")

    # Refit the chosen model on 100% of the labeled data for real deployment.
    vectorizer, best_model = train_final_model(df, best_name)

    demo_on_new_tickets(vectorizer, best_model)

    print("\n" + "=" * 60)
    print(f"Human review threshold: confidence < {REVIEW_THRESHOLD:.0%} "
          "-> routed to NEEDS_HUMAN_REVIEW instead of auto-assigned.")
    print("=" * 60)


if __name__ == "__main__":
    main()