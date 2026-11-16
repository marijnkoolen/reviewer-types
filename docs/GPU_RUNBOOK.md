# GPU runbook: open-model sentence classifier (UA-025 a)

Hardware (UA-028): 2 VMs (called VM1 and VM2 below), each with 2 A10 GPUs of 22 GB. You start the jobs; I can't reach the VMs.
Scripts: `src/classifier/gt_v2_llm.py` (train + evaluate), `gt_v2_llm_predict.py` (label a corpus),
`gt_v2_transformer.py` (encoder baseline). All were smoke-tested on the Mac (2026-11-16) with `Qwen/Qwen2.5-0.5B`, unquantised;
the 4-bit path (bitsandbytes) only runs on CUDA and is first tested by step 3.

## 1. Copy code and data to each VM (Run on the Mac, in the project root)
Replace `VM1` with the ssh host and `~/reviewer-types` with the target folder; repeat for VM2.

    rsync -aR src/annotate/review_meta.py src/classifier data/gt_v2/train_input.tsv.gz VM1:~/reviewer-types/

For corpus labelling (step 5) also copy the corpus (250 MB) and the metadata (13 MB):

    rsync -aR data/review_sentences.tsv.gz metadata/review_metadata.tsv.gz VM1:~/reviewer-types/

`data/gt_v2/train_input.tsv.gz` (from `gt_v2_export.py`) holds the 29,462 training sentences with labels, header and
split; on a machine without `data/gt_v2/labels.tsv` the scripts read it instead. These files hold review text: keep
them on the VMs, not in a public place.

## 2. Environment (Run on each VM, in `~/reviewer-types`)

    python3 -m venv venv && . venv/bin/activate
    pip install torch            # or the wheel for the VM's CUDA version
    pip install -r src/classifier/requirements_gpu.txt

## 3. Training: four runs at once, one per GPU (Run on each VM, in `~/reviewer-types`, venv active)
Run each command in its own tmux window (for example `tmux new -s qwen7b`, activate the venv, paste the line).
Each run: 4-bit base model + LoRA, 2 epochs, max 2,048 tokens per sequence (4,003 sequences, 2.1M tokens per epoch).
My estimate is 1–3 h per run on an A10; the log prints progress every 200 sequences.

VM1:

    CUDA_VISIBLE_DEVICES=0 python src/classifier/gt_v2_llm.py --model Qwen/Qwen2.5-7B --name llm_qwen25_7b 2>&1 | tee llm_qwen25_7b.log
    CUDA_VISIBLE_DEVICES=1 python src/classifier/gt_v2_llm.py --model utter-project/EuroLLM-9B --name llm_eurollm_9b 2>&1 | tee llm_eurollm_9b.log

VM2:

    CUDA_VISIBLE_DEVICES=0 python src/classifier/gt_v2_llm.py --model Qwen/Qwen2.5-14B --name llm_qwen25_14b --max_len 1536 2>&1 | tee llm_qwen25_14b.log
    CUDA_VISIBLE_DEVICES=1 python src/classifier/gt_v2_transformer.py --model FacebookAI/xlm-roberta-large --name xlmr_large --lr 1e-5 2>&1 | tee xlmr_large.log

- The model names are the versions I know; a newer version of the same family can be passed with `--model`.
- Gated models (2026-11-16: `utter-project/EuroLLM-9B` answers 401 "gated repo"): accept the terms on the model's Hugging Face page, create a read token (huggingface.co → Settings → Access Tokens), then on the VM run `hf auth login` (older versions: `huggingface-cli login`) and paste the token. The Qwen models are not gated.
- 14B is the largest that should fit in 22 GB at 4 bits; if it runs out of memory, lower `--max_len` to 1024.
- Out of memory on any run: add `--max_len 1024` (more reviews get split, but every sentence still sees its neighbours).
- `xlmr_large` is the baseline from UA-025 (b): the RobBERT setup with a multilingual large encoder.

## 4. Copy the results back (Run on the Mac, in the project root)

    rsync -a VM1:~/reviewer-types/results/gt_v2_classifier/ results/gt_v2_classifier/
    rsync -a VM2:~/reviewer-types/results/gt_v2_classifier/ results/gt_v2_classifier/

Each run folder has `summary.json`, `per_label.tsv`, `predictions.tsv.gz`, `train_log.json` and (LLM runs) `adapter/`
(LoRA weights + head, ~100–300 MB, not in git). I then compare the runs with RobBERT (test micro-F1 0.80, gold 0.77).

## 5. Label the Dutch corpus with the best run (Run on each VM, in `~/reviewer-types`)
The adapter folder of the chosen run must be on both VMs (copy `results/gt_v2_classifier/<run>/` across). Shard K of 4
takes every 4th review; with `<run>` = the chosen run name, each in its own tmux window:

    # VM1
    CUDA_VISIBLE_DEVICES=0 python src/classifier/gt_v2_llm_predict.py --run <run> --shard 0 --num_shards 4 2>&1 | tee predict_0.log
    CUDA_VISIBLE_DEVICES=1 python src/classifier/gt_v2_llm_predict.py --run <run> --shard 1 --num_shards 4 2>&1 | tee predict_1.log
    # VM2
    CUDA_VISIBLE_DEVICES=0 python src/classifier/gt_v2_llm_predict.py --run <run> --shard 2 --num_shards 4 2>&1 | tee predict_2.log
    CUDA_VISIBLE_DEVICES=1 python src/classifier/gt_v2_llm_predict.py --run <run> --shard 3 --num_shards 4 2>&1 | tee predict_3.log

Output: `results/corpus_labels/<run>/shard_KK_of_04/part_*.tsv.gz` (per sentence: probabilities and labels). A
stopped job resumes when started again with the same command (also after the tmux session is gone). Speed is printed per part of 2,000 reviews.
