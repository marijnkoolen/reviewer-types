"""Turn a parser's output into spaCy training data for its lemmatizer.

Premise: nl_core_news_lg already lemmatises with an EditTreeLemmatizer -- a
trained component, not a lookup table. Its out-of-domain failures are learned
suffix-edit rules misfiring on words it never saw (lotgenoat, opsteekken,
familienam, werkwijz). That is a data problem, not an architecture problem, so
the fix is to retrain that component on in-domain text lemmatised by a system
with a real lexicon behind it: Alpino for preference, Frog as the fallback.

Alignment: spaCy must see the tokenisation it will see at run time, so each line
is tokenised by spaCy itself and silver lemmas are attached only where spaCy's
character span matches the silver system's span exactly. Tokens without an exact
match keep an empty lemma, which spaCy treats as missing annotation and excludes
from the loss -- better than training on a guessed alignment.

Splitting is by review, never by sentence: sentences from one review share
vocabulary and style, so a sentence-level split leaks and flatters the dev score.

WHAT THIS CAN AND CANNOT BUY YOU
  * It cannot exceed the silver system's own accuracy; you are distilling
    Alpino/Frog into a faster model, not improving on it.
  * It inherits the silver system's lemma conventions. Alpino and Frog produce
    plain Dutch citation forms; Stanza produces UD Dutch Alpino conventions with
    underscores (op_steken). Do not mix silver sources in one training set.
  * It is worth doing precisely because the errors it fixes are concentrated in
    rare and OOV words, which is where this corpus's vocabulary lives.

Usage:
  python src/parsers/build_lemma_trainset.py --silver alpino
  python src/parsers/build_lemma_trainset.py --silver frog --dev-frac 0.15
Writes results/parser_comparison/lemma_train/{train,dev}.spacy and config.cfg
"""
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_fmt import OUT_DIR, normalise, read_tokens  # noqa: E402
from compare_util import discover_tables  # noqa: E402
from common_fmt import iter_lines  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver", default="alpino",
                    help="tool whose lemmas become the training target")
    ap.add_argument("--base-model", default="nl_core_news_lg",
                    help="pipeline supplying the tokenizer, and the frozen "
                         "components the new lemmatizer trains on top of")
    ap.add_argument("--dev-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(OUT_DIR / "lemma_train"))
    args = ap.parse_args()

    tables = discover_tables()
    if args.silver not in tables:
        raise SystemExit(
            f"no tokens_{args.silver}.tsv. Available: {', '.join(tables)}")

    # silver[(doc, line)][(start, end)] = lemma
    silver = defaultdict(dict)
    for row in read_tokens(tables[args.silver]):
        if row["start"] < 0 or not row["lemma"]:
            continue
        silver[(row["doc_id"], row["line_id"])][(row["start"], row["end"])] = \
            row["lemma"]

    lines = [(d, i, normalise(t)) for d, i, t in iter_lines()]
    docs_ids = sorted({d for d, _, _ in lines})
    rng = random.Random(args.seed)
    rng.shuffle(docs_ids)
    n_dev = max(1, int(len(docs_ids) * args.dev_frac))
    dev_docs = set(docs_ids[:n_dev])

    import spacy
    from spacy.tokens import DocBin

    nlp = spacy.load(args.base_model, exclude=["ner", "parser"])
    tokenizer = nlp.tokenizer
    # A sourced component carries no inline model config, so the listener width
    # cannot be interpolated from it -- read the literal value off the base
    # pipeline and bake it in.
    tok2vec_width = nlp.config["components"]["tok2vec"]["model"]["encode"]["width"]

    bins = {"train": DocBin(store_user_data=False),
            "dev": DocBin(store_user_data=False)}
    stats = {"train": [0, 0], "dev": [0, 0]}  # [tokens, labelled tokens]

    for doc_id, line_id, text in lines:
        gold = silver.get((doc_id, line_id))
        if not gold:
            continue
        doc = tokenizer(text)
        labelled = 0
        for tok in doc:
            span = (tok.idx, tok.idx + len(tok.text))
            lemma = gold.get(span)
            if lemma:
                tok.lemma_ = lemma
                labelled += 1
            else:
                tok.lemma_ = ""  # missing annotation: excluded from the loss
        if not labelled:
            continue
        split = "dev" if doc_id in dev_docs else "train"
        bins[split].add(doc)
        stats[split][0] += len(doc)
        stats[split][1] += labelled

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, db in bins.items():
        db.to_disk(out_dir / f"{split}.spacy")

    # The skeleton below only states what is non-default; spaCy's own
    # fill-config expands it to a complete, validated config. Writing a full
    # config by hand would silently rot against the installed spaCy version.
    cfg_path = out_dir / "config.cfg"
    cfg_path.write_text(_config(args.base_model, tok2vec_width),
                        encoding="utf-8")
    from spacy.cli.init_config import fill_config
    fill_config(cfg_path, cfg_path, silent=True)
    # fill_config re-adds default score weights for every component in the
    # pipeline, including the frozen tagger and morphologizer. Those score 0
    # here because the training data carries lemmas only, which makes the
    # reported SCORE meaningless and best-model selection depend on constants.
    # Lemma accuracy is the only thing being optimised, so say so.
    _force_lemma_only_scoring(cfg_path)

    print(f"silver source : {args.silver}")
    print(f"base model    : {args.base_model}")
    for split in ("train", "dev"):
        tok, lab = stats[split]
        print(f"{split:<14}: {len(bins[split].strings) and ''}"
              f"{sum(1 for _ in bins[split].get_docs(nlp.vocab))} docs, "
              f"{tok} tokens, {lab} labelled "
              f"({100 * lab / max(1, tok):.1f}%)")
    print(f"\nwrote {out_dir}/train.spacy, dev.spacy, config.cfg")
    print("\ntrain with:\n"
          f"  python -m spacy train {cfg_path} \\\n"
          f"    --paths.train {out_dir}/train.spacy \\\n"
          f"    --paths.dev {out_dir}/dev.spacy \\\n"
          f"    --output {out_dir}/model --gpu-id 0")
    print("\nThen evaluate the retrained lemmatizer the same way as everything "
          "else:\n"
          "  python src/parsers/run_spacy.py --model "
          f"{out_dir}/model/model-best --out "
          f"{OUT_DIR}/tokens_spacy_retrained.tsv\n"
          "  python src/parsers/compare_lemmas.py\n"
          "and check whether its unattested-lemma rate drops toward the silver "
          "system's.")


def _force_lemma_only_scoring(cfg_path):
    """Rewrite [training.score_weights] so only lemma_acc counts."""
    from thinc.api import Config
    # interpolate=False is essential: interpolating resolves ${paths.train}
    # to its null placeholder and writes that back, so the CLI overrides
    # would have nothing left to fill.
    cfg = Config().from_disk(cfg_path, interpolate=False)
    cfg["training"]["score_weights"] = {
        "lemma_acc": 1.0, "tag_acc": None, "pos_acc": None,
        "morph_acc": None, "tag_micro_p": None, "tag_micro_r": None,
        "tag_micro_f": None, "morph_per_feat": None,
    }
    cfg.to_disk(cfg_path)


def _config(base_model, tok2vec_width):
    """Config that trains only the lemmatizer, on top of frozen components
    sourced from the base pipeline.

    Freezing matters: the point is to replace the lemmatizer without disturbing
    the tagger and morphologizer, which are fine and which downstream analyses
    may already depend on.
    """
    return f"""# Retrain only the EditTreeLemmatizer of {base_model} on in-domain
# silver lemmas. Every other component is sourced from the base pipeline and
# frozen, so tagging and morphology are untouched.
[paths]
train = null
dev = null

[nlp]
lang = "nl"
pipeline = ["tok2vec","tagger","morphologizer","trainable_lemmatizer"]
batch_size = 128

[components]

[components.tok2vec]
source = "{base_model}"

[components.tagger]
source = "{base_model}"

[components.morphologizer]
source = "{base_model}"

[components.trainable_lemmatizer]
factory = "trainable_lemmatizer"
backoff = "orth"
min_tree_freq = 2
overwrite = true
top_k = 1

[components.trainable_lemmatizer.model]
@architectures = "spacy.Tagger.v2"
nO = null
normalize = false

[components.trainable_lemmatizer.model.tok2vec]
@architectures = "spacy.Tok2VecListener.v1"
width = {tok2vec_width}
upstream = "tok2vec"

[corpora]

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${{paths.train}}
gold_preproc = false
max_length = 0

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${{paths.dev}}
gold_preproc = false
max_length = 0

[training]
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"
seed = 0
frozen_components = ["tok2vec","tagger","morphologizer"]
annotating_components = ["tok2vec","tagger","morphologizer"]
max_epochs = 0
max_steps = 20000
patience = 3000
eval_frequency = 500

[training.optimizer]
@optimizers = "Adam.v1"

[training.batcher]
@batchers = "spacy.batch_by_words.v1"
discard_oversize = false
size = 2000
tolerance = 0.2

[training.logger]
@loggers = "spacy.ConsoleLogger.v1"
progress_bar = false

[training.score_weights]
lemma_acc = 1.0

[initialize]
vectors = "{base_model}"
"""


if __name__ == "__main__":
    main()
