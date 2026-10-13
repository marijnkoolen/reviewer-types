"""Shared category taxonomy and column-name mapping across the gold, silver, and
test-set files, which each spell out labels differently."""

# short code (used in predictions files) -> full label name (used in gold annotation file)
CODE_TO_LABEL = {
    "aut": "Author",
    "cla": "Classification",
    "con": "Content",
    "con_nar": "Content--Narrative",
    "con_oth": "Content--Other",
    "con_quo": "Content--Quote",
    "con_the": "Content--Theme",
    "oth": "Other_works",
    "rea": "Reader_response",
    "rea_eva": "Reader_response--Evaluation_of_quality",
    "rea_fee": "Reader_response--Feelings",
    "rea_ide": "Reader_response--Identification_and_immersion",
    "rea_rea": "Reader_response--Reading_Context",
    "rea_rec": "Reader_response--Reception",
    "rea_ref": "Reader_response--Reflection",
    "rec": "Recommendations",
    "sty": "Style",
    "sty_con": "Style--Context",
    "sty_str": "Style--Structure",
    "sty_sty": "Style--Stylistic_features",
}
LABEL_TO_CODE = {v: k for k, v in CODE_TO_LABEL.items()}

MAIN_LABELS = ["Author", "Classification", "Content", "None", "Other_works",
               "Reader_response", "Recommendations", "Style"]

SUB_LABELS = [v for v in CODE_TO_LABEL.values() if "--" in v]

# sub-category -> parent main category, for enforcing the hierarchy constraint
# after correction/imputation (sub == 1 implies parent main == 1)
SUB_TO_MAIN = {v: v.split("--")[0] for v in SUB_LABELS}

ALL_CODES = list(CODE_TO_LABEL.keys())  # excludes 'None', which has no predictions column

# --- Two distinct exclusion lists. Do not conflate them. ---
#
# SPARSE_GOLD_LABELS: too few positive *gold* (human-annotated) examples for
# reliable quantitative analysis on their own. This is a statistical-power
# issue and applies everywhere -- gold and silver alike.
SPARSE_GOLD_LABELS = [
    "Style--Context",              # 11/1152 gold reviews (1.0%)
    "Content--Quote",              # 68/1152 gold reviews (5.9%)
    "Reader_response--Reception",  # 68/1152 gold reviews (5.9%)
]

# WEAK_MODEL_LABELS: the RobBERT/MaChAmp classifier tests poorly on these on
# the held-out test set (see results/correction/confusion_matrix.csv), independent of how
# prevalent they are. This is a *model reliability* issue, so it only applies
# to model-predicted (silver) data -- these labels remain fully valid on the
# gold set, where they are human-annotated ground truth, not predictions.
# Test-set F1: Reflection 0.40, Style--Structure 0.48, Content--Theme 0.52,
# Feelings 0.54 (all well below the 0.7-0.9 range of the other labels).
WEAK_MODEL_LABELS = [
    "Reader_response--Reflection",
    "Style--Structure",
    "Content--Theme",
    "Reader_response--Feelings",
]

# Genres the annotated (gold + test) data covers. RobBERT/MaChAmp was never
# trained or evaluated on Non-fiction or missing-genre reviews, so its error
# rates on them are unknown -- the confusion matrix (and any correction built
# on it) does not demonstrably apply outside this set. Exclude Non-fiction and
# missing-genre reviews from every silver analysis that relies on correction
# (raw/uncorrected silver numbers for them may still be reported separately,
# clearly labeled as unvalidated).
FICTION_GENRES = [
    "Children_fiction", "Fantasy_fiction", "Historical_fiction",
    "Literary_fiction", "Literary_thriller", "Other fiction",
    "Regional_fiction", "Romance", "Suspense", "Young_adult",
]
