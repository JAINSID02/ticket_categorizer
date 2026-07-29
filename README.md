# Auto Email / Ticket Categorizer

A lightweight NLP classifier that reads an incoming support ticket (subject + body)
and routes it to the correct department — **Billing**, **Technical**, **HR**, or
**General** — in real time.

## Project structure

```
ticket_categorizer/
├── data/
│   └── tickets.csv          # 56 labeled sample tickets, 14 per category
├── ticket_categorizer.py    # full pipeline: clean -> vectorize -> train -> evaluate -> predict
├── streamlit_app.py         # mini live demo (bonus)
└── README.md
```

## How to run

```bash
pip install scikit-learn pandas
python ticket_categorizer.py
```

This prints:
1. Dataset summary
2. Naive Bayes vs Logistic Regression comparison (accuracy, precision/recall,
   confusion matrix) on a held-out test split
3. The selected model, refit on the full labeled dataset
4. Live predictions on 7 brand-new, unseen tickets (category, confidence,
   priority, and routing decision)

For the interactive demo (bonus):
```bash
pip install streamlit
streamlit run streamlit_app.py
```

## Approach summary

Text is lowercased, stripped of URLs/punctuation/digits, and passed through a
small custom stopword filter, then converted into **TF-IDF** unigram vectors —
TF-IDF was chosen over raw Bag-of-Words because it down-weights words common
across every ticket ("please", "account") and up-weights the words that
actually separate departments ("invoice", "crash", "leave"). Both **Multinomial
Naive Bayes** and **Logistic Regression** are trained on the same features and
compared on a held-out split; the better performer is refit on the full dataset
for deployment, since a real triage layer should use every labeled example it
has. Edge cases (vague tickets like "just checking in") are handled by a
**confidence threshold** (60%): instead of forcing a label, low-confidence
tickets are routed to `NEEDS_HUMAN_REVIEW`, mirroring how real helpdesk triage
tools flag uncertain cases rather than silently misrouting them. A separate,
transparent keyword layer tags **priority** (`URGENT`/`NORMAL`), independent of
the ML category prediction, so an urgent ticket is flagged even if the category
model is unsure.

## Why Naive Bayes (in this run)

The script trains both models and picks whichever scores higher accuracy on
the held-out test set — it doesn't hardcode a winner. On this dataset, Naive
Bayes performs at least as well as Logistic Regression and is more efficient
for the small, sparse, high-dimensional TF-IDF vectors typical of short text,
which is exactly why it's the traditional first choice for text classification.
Logistic Regression is kept in the comparison because it tends to do better as
vocabulary overlap between categories grows (e.g. "account", "please" appear
everywhere) — worth re-checking once more real data is collected.

## Evaluation

On the current 56-ticket dataset (14/category), an 75/25 train/test split gives
roughly 79-86% accuracy depending on model and split — expected for a
dataset this small. Naive Bayes' `alpha` (Laplace smoothing) was lowered from
sklearn's default of 1.0 to 0.1: with so few examples per class, the default
smoothing flattened every prediction into the 25-45% confidence range
regardless of how clear-cut the ticket was, which made the confidence score
useless for triage. At `alpha=0.1`, clear-cut tickets score 80-98% confidence
and genuinely ambiguous ones correctly drop below the 60% review threshold —
run the script to see both cases side by side in the "Live Routing Preview."

## Reflection — what I'd improve with more data or time

- **More labeled data.** 56 examples is enough to prove the pipeline works,
  but not enough to trust the accuracy numbers — with only 3-4 examples per
  class in the test split, one misclassification swings accuracy by ~7%. A
  few hundred real historical tickets per category would give a much more
  reliable signal and let me use proper k-fold cross-validation instead of a
  single train/test split.
- **Better calibrated confidence.** Naive Bayes probabilities are notoriously
  overconfident/underconfident depending on smoothing — I'd add a calibration
  step (`CalibratedClassifierCV`) so the confidence score can be trusted as an
  actual probability, not just a ranking signal.
- **Multi-label / sub-category routing.** Real tickets sometimes span two
  departments (e.g. "billing issue caused by a technical bug"). Right now the
  model is forced to pick one label; a multi-label setup or a secondary
  "related category" field would route these more accurately.
- **Smarter priority tagging.** The current urgency tag is a flat keyword
  list. With more time I'd weight keywords by position (subject line vs. body)
  and combine it with the category confidence, so a low-confidence "URGENT"
  ticket gets escalated to a human faster than a low-confidence "NORMAL" one.
- **Feedback loop.** In production, every ticket a human re-routes after
  auto-assignment is a free labeled example. I'd log those corrections and
  periodically retrain, so the model improves from its own mistakes instead
  of staying frozen at launch-day accuracy.