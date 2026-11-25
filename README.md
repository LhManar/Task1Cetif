# SWYNEX-AI-Problem-Design

**Task 1 – SWYNEX Technologies Internship: AI Problem Design**

**Use case:** Automatic classification of customer support tickets into categories (text classification)

---

## 1. Problem Statement

Small support teams receive tickets by email or chat and sort them by hand: billing questions, technical faults, account problems, and general inquiries. This is slow, inconsistent between agents, and delays urgent issues.

**Goal:** Build a lightweight text classifier that reads a short ticket and predicts one of four categories, so tickets reach the right person faster.

| Label | Description | Example |
|---|---|---|
| `billing` | Payments, invoices, refunds | "I was charged twice this month." |
| `technical` | Bugs, errors, things not working | "The app crashes when I upload a photo." |
| `account` | Login, password, profile changes | "I can't reset my password." |
| `general` | Questions, feedback, everything else | "Do you offer a student discount?" |

## 2. Target User

**Primary user:** A support agent or team lead at a small business (1–10 agents) who triages incoming tickets.

**User need:** See a suggested category and confidence score for each ticket, so they can route it in seconds and review only the uncertain ones.

The model **assists** the agent; it does not make final decisions on its own.

## 3. Data Source

- **Dataset size:** about 200–400 short tickets (50–100 per class).
- **Origin:** Hand-written, synthetic tickets that imitate real support messages, with no personal data. A public customer-support dataset can be used instead or added to scale up.
- **Labeling:** One label per ticket, assigned using the definitions in the table above. A second pass over a random 10% checks for labeling consistency.
- **Format:** CSV with columns `text`, `label`.
- **Split:** 70% train / 15% validation / 15% test, stratified by label. The test set is never used for tuning.

## 4. Constraints

| Area | Constraint |
|---|---|
| **Data size** | Small dataset, so a simple model is preferred over a large one. |
| **Compute** | Must train and run on a laptop CPU, with no GPU needed. |
| **Latency** | Under 1 second per ticket. |
| **Privacy** | No real customer data or personal information in the repo. |
| **Language** | English only. |
| **Scope** | Single label per ticket. No multi-label, no priority detection. |
| **Explainability** | The agent should be able to see why a label was suggested (for example, top keywords). |

## 5. Proposed Approach

1. **Baseline:** Keyword rules (for example, "refund" or "invoice" → `billing`).
2. **Classical ML:** TF-IDF features with Logistic Regression.
3. **Optional extension:** Zero-shot or few-shot classification with an LLM, compared against the model above.

Starting simple makes it clear whether each added layer of complexity is worth it.

## 6. Success Criteria

The project is successful if, on the held-out test set:

- **Macro F1 ≥ 0.85**, so no single class is neglected.
- **Accuracy ≥ 0.88.**
- **Beats the keyword baseline** by at least 10 percentage points in macro F1.
- **No class has recall below 0.75**, since missing a whole category is the worst failure.
- **Low-confidence handling:** tickets below a confidence threshold (for example, 0.6) are flagged for manual review, and accuracy on the remaining tickets is at least 0.92.

## 7. Evaluation Approach

- **Metrics:** accuracy, precision, recall, and F1 per class, plus macro F1.
- **Confusion matrix:** to see which categories get mixed up (likely `general` vs. `account`).
- **Error analysis:** manually review about 20 misclassified tickets and group them by cause (ambiguous wording, multiple topics, short text).
- **Robustness checks:** typos, all-lowercase text, and very short messages ("help!!").
- **Comparison table:** baseline vs. TF-IDF + Logistic Regression (vs. LLM, if tried).
- **Cross-validation:** 5-fold on train+validation to check that results are stable on a small dataset.

## 8. Risks and Limitations

- **Synthetic data may be easier than real tickets**, so real-world performance would likely be lower.
- **Small dataset** means high variance. Metrics could shift noticeably with a different split.
- **Overlapping topics:** a ticket such as "I was charged and now can't log in" fits two categories.
- **Label bias:** categories reflect one person's definitions.
- **Misrouting risk:** a wrong prediction could delay a customer, which is why a human stays in the loop.

## 9. Next Steps

1. Create and label the dataset.
2. Implement the keyword baseline and TF-IDF + Logistic Regression.
3. Evaluate on the test set and write up the error analysis.
4. Record a short demo (paste a ticket, see the predicted category and confidence).

## 10. Repository Structure

```
SWYNEX-AI-Problem-Design/
├── README.md        # this document
├── data/            # tickets.csv, train.csv, val.csv, test.csv
├── src/
│   ├── make_dataset.py   # generates the synthetic dataset
│   └── train_eval.py     # cleaning, split, baseline, model, evaluation
├── results/         # metrics.json, confusion_matrix.png, errors.csv
└── requirements.txt
```

## 11. How to Run

```bash
pip install -r requirements.txt
python src/make_dataset.py   # creates data/tickets.csv (290 tickets)
python src/train_eval.py     # cleans, splits, trains, evaluates
```

## 12. Results (test set, 44 tickets)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Keyword baseline | 0.932 | 0.932 |
| TF-IDF + Logistic Regression | 0.977 | 0.977 |

- 5-fold cross-validation macro F1: **0.975 ± 0.02**
- Lowest per-class recall: `billing` at 0.909 (all other classes 1.0)
- Only 1 misclassified test ticket (see `results/errors.csv`)

**Honest caveat:** the data is synthetic and built from templates, so these scores are optimistic. The success criteria are met on this dataset, but real tickets would be harder. The next improvement is to test on real or public support data.

---

*Author: Lahouel Manar · SWYNEX Technologies Internship*
