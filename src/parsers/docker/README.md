# Alpino in Docker — install and run

Alpino ships as an x86_64 Linux binary linked against a specific glibc. The
image here installs it from the **official RUG distribution**
(`www.let.rug.nl/vannoord/alp/Alpino/versions/binary/`), not from a third-party
Docker Hub image, so the provenance and the version are both known.

The base image fixes the glibc version, so the host distribution does not
matter — this works the same on Ubuntu 20.04, 22.04, Debian or RHEL.

## Prerequisites

```bash
docker --version          # any recent Docker Engine
docker info >/dev/null && echo "daemon reachable"
id -nG | tr ' ' '\n' | grep -qx docker && echo "user in docker group"
df -h .                   # ~3 GB free: 424 MB download, ~1.8 GB image
nproc                     # shard count: one per core
free -g                   # see "Memory" below
```

If `docker info` fails with a permission error, add yourself to the `docker`
group and start a new login shell:

```bash
sudo usermod -aG docker "$USER"
newgrp docker          # or log out and back in
```

**Do this if you can.** Running the container under `sudo` makes every output
file root-owned, which also breaks resume on the next run: the state file is no
longer writable by you. See *Running when Docker needs sudo* below if group
membership is not an option on this VM.

## Build

```bash
make alpino-image
```

or directly:

```bash
docker build -f src/parsers/docker/Dockerfile.alpino \
  -t alpino-impfic:latest src/parsers/docker
```

Takes a few minutes, almost all of it the 424 MB tarball download. On Apple
Silicon add `--platform linux/amd64` (it then runs emulated, several times
slower — fine for testing, not for benchmarking).

To pin a different Alpino build, or to target a host that needs an older glibc:

```bash
docker build -f src/parsers/docker/Dockerfile.alpino -t alpino-impfic:latest \
  --build-arg ALPINO_TARBALL=Alpino-x86_64-Linux-glibc-2.23-git20250710.070258-sicstus.tar.gz \
  --build-arg BASE_IMAGE=ubuntu:16.04 \
  src/parsers/docker
```

## Verify

```bash
docker run --rm alpino-impfic:latest bash -c '
  printf "impfic-test-1|Dit is een test .\n" > /tmp/t.txt
  mkdir -p /tmp/xml
  "$ALPINO_HOME/bin/Alpino" -notk -veryfast -flag treebank /tmp/xml \
      end_hook=xml -parse < /tmp/t.txt >/dev/null 2>&1
  ls /tmp/xml'
```

Expect `impfic-test-1.xml`. That one command checks all four things that can go
wrong: the binary loads (it needs the X11/Tcl libraries even headless), the
locale is right, the grammar is present, and the sentence-id naming works.

## Run the benchmark

The runner uses only the Python standard library, so it runs **inside** the
image with no venv. Prefer one container with N processes over N containers:
less overhead, one image load, and a single place to watch.

```bash
docker run --rm \
  -v "$PWD":/work -w /work \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e SHARDS=22 \
  alpino-impfic:latest bash src/parsers/bench_alpino.sh
```

* `--user` keeps output files owned by you instead of root.
* `-e HOME=/tmp` is needed because that user has no home directory in the image.
* `-e LIMIT=200` first, to confirm the whole path works before the full run.
* `-e SPEED=-fast` or `-e SPEED=` trade throughput for accuracy
  (`-veryfast` is the default).
* `-e CHUNK=1000` sets the chunk size — see **Chunking and resume** below.

Results land in `results/alpino_tsv/`:

| file | contents |
|---|---|
| `alpino_tokens.shard<i>.tsv` | `review_id sent_num tok_id begin end word lemma pos postag rel head_tok_id frame` |
| `xml_shard<i>_chunk<k>.tar.gz` | full Alpino XML, with `--keep-xml` |
| `alpino_parsequality.shard<i>.tsv` | per sentence: `n_tokens`, Alpino's `cats` and `skips` |
| `alpino_bench.shard<i>.json` | timing for that shard |

| `alpino_state.shard<i>.json` | resume progress; delete it, or pass `--restart`, to start that shard over |

`skips > 0` means Alpino could not build one spanning analysis and fell back to
a sequence of fragments — worth filtering on before trusting a parse.

## Running when Docker needs sudo

`--user "$(id -u):$(id -g)"` is expanded by *your* shell before `sudo` runs, so
a direct `sudo docker run ...` already writes files owned by you. The problem is
`sudo make ...`: the recipe itself then runs as root, `id -u` reports 0, and
everything lands root-owned.

Pick one:

```bash
# 1. Best: no sudo at all.
sudo usermod -aG docker "$USER" && newgrp docker
make bench-alpino-docker SHARDS=22

# 2. Keep make unprivileged, sudo only the docker call.
make bench-alpino-docker SHARDS=22 DOCKER_CMD="sudo docker"

# 3. Whole thing under sudo still works: the Makefile prefers SUDO_UID/SUDO_GID,
#    which sudo sets to the invoking user.
sudo make bench-alpino-docker SHARDS=22
```

If an earlier run already produced root-owned files, take them back before
resuming — otherwise the run fails on the state file instead of resuming:

```bash
sudo chown -R "$USER" results/alpino_tsv
```

## Dependencies in the TSV

`rel` + `head_tok_id` form a CoNLL-shaped tree (`head_tok_id` 0 = ROOT). Alpino
cannot emit XML and dependency triples in one pass, so heads are reconstructed
from the XML by head percolation.

That reconstruction agrees with Alpino's own triples on **96.5% of tokens**
(830 tokens, `validate_heads.py`). The residual is concentrated in relative and
wh pronouns (~1.6%), which Alpino resolves via co-indexation to a trace inside
the clause, and a discourse long tail (~1.1%). Everything else -- clause
structure, coordination, complementation, punctuation -- matches.

Re-check it yourself after any change to the conversion:

```bash
make validate-heads LIMIT=200
```

If dependencies are central to your analysis rather than incidental, either run
a second pass with `end_hook=dependencies` (exact, but doubles Alpino time) or
use `--keep-xml` so the conversion can be redone later without re-parsing.

## Killing a run cleanly

Killing the driver is not enough. Shards are backgrounded, each spawns Alpino as
a child, and a kill aimed at an ancestor leaves those children running with
their memory. Several interrupted runs then accumulate and oversubscribe the
machine: the symptom is a collapsed per-shard rate with **no** swapping and
**no** iowait, and `vmstat` showing a run queue (`r`) several times the core
count while `us` sits at 99%.

The drivers now trap INT/TERM and signal their whole process group, so Ctrl-C
cleans up. A `kill -9` cannot be trapped, so after one, check for survivors:

```bash
# containerised runs: the container outlives a killed `docker run` client
docker ps --filter ancestor=alpino-impfic:latest
docker kill $(docker ps -q --filter ancestor=alpino-impfic:latest)

# native runs, or leftovers inside a container
pgrep -af "run_alpino_tsv.py|Alpino.bin" | head
pkill -f run_alpino_tsv.py; pkill -f Alpino.bin

# confirm the machine is idle again before relaunching
vmstat 1 3      # r should fall to ~0, us to ~0
```

Check `r` against `nproc` before blaming throughput on Alpino: if the run queue
is well above the core count, every process is simply timesharing.

## Chunking and resume

Each shard is processed in chunks of `CHUNK` sentences (default 1000) rather
than in one pass. Output is appended after every chunk, so rows appear while the
job runs instead of only at the end, and per-chunk progress is logged to
`results/alpino_tsv/_bench_shard<i>.log`:

    shard 3 chunk 7/32: 7000/31500 sentences, 118431 tokens, 412s (2.43 sent/s)

**To resume an interrupted run, re-run the identical command.** Each shard picks
up at its first unfinished chunk; shards that already completed do nothing.

Recovery is byte-exact rather than best-effort. After each chunk the output
files are flushed and `fsync`ed and their sizes recorded in the state file; on
restart the outputs are truncated back to those recorded sizes before appending.
A chunk that was half-written when the process died therefore leaves no partial
rows behind. This is tested: injecting torn rows into a killed run's output and
resuming reproduces a byte-identical file to an uninterrupted run.

Changing `--input`, `--shard`, `--n-shards`, `--chunk-size`, `--limit` or
`--speed` changes what a chunk *is*, so resuming across such a change would
interleave two different partitions of the data. The runner refuses, naming the
mismatch; pass `--restart` if starting over is what you meant.

Choosing `CHUNK`: it is the resume granularity, so an interruption costs at most
one chunk of work, but every chunk pays one Alpino startup of roughly five
seconds. At 1–3 sentences/s a 1000-sentence chunk runs 5–15 minutes and startup
is under 2% — a reasonable trade. Go smaller only if you expect frequent
interruptions.

If you prefer a native install to Docker, the same tarball unpacks anywhere and
the runners take `--alpino-home /path/to/Alpino`; the apt packages the
Dockerfile installs are then the host's responsibility.

## Memory

Each Alpino process holds roughly 1 GB resident, so 22 shards need ~22 GB plus
headroom. Check with `free -g` and lower `SHARDS` if the VM is tight.

This is the failure mode to watch for: under memory pressure shards do not fail
loudly, they thrash. On a Mac with Docker Desktop capped at 3.8 GB, eight shards
OOM-killed each other (exit 137) and throughput fell to ~0.05 sentences/s —
about 8× slower than the same work with enough memory. On Linux, Docker uses
host memory directly and there is no separate VM ceiling to raise, but the
arithmetic still applies.

## Notes

* **Alpino is CPU-only.** It makes no use of the A10s; the 22 cores are the
  resource that matters and the GPUs sit idle. Save those for Stanza, spaCy-trf
  or any lemmatizer retraining.
* **`--keep-xml` does not scale.** It keeps one `<review_id>-<sent_num>.xml` per
  sentence, which at full corpus size is millions of files per shard directory.
  The TSV is normally what you want.
* **Per-sentence timeout.** `--user-max` (default 60000 ms) makes Alpino return
  a partial parse rather than hang on a pathological sentence. Long sentences
  are where its runtime blows up, so this bounds the worst case.
