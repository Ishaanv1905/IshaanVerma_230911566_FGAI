"""Word-level autoregressive text generation on TinyStories.

Trains SimpleRNN, BiRNN, LSTM and GRU next-word predictors under identical
conditions, evaluates them, and generates text at several temperatures.

Usage:
    python train.py                                   # full run
    python train.py --epochs 1 --train_stories 500 --out results_test   # smoke test
"""
import argparse
import collections
import json
import math
import os
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--data", default="data/input.txt")
p.add_argument("--out", default="results")
p.add_argument("--train_stories", type=int, default=12000)
p.add_argument("--val_stories", type=int, default=1500)
p.add_argument("--test_stories", type=int, default=1500)
p.add_argument("--stride", type=int, default=4, help="step between consecutive training windows")
p.add_argument("--seq_len", type=int, default=24)
p.add_argument("--vocab_size", type=int, default=5000)
p.add_argument("--embed_dim", type=int, default=100)
p.add_argument("--hidden", type=int, default=128)
p.add_argument("--epochs", type=int, default=6)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--gen_words", type=int, default=60)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
torch.set_num_threads(os.cpu_count() or 2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UNK, EOS, NL = "<unk>", "<eos>", "<nl>"


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)


# ----------------------------------------------------------------------------
# 1. Data: load stories, split by story, tokenize
# ----------------------------------------------------------------------------
raw = open(args.data, encoding="utf-8").read()
stories = [s.strip() for s in raw.split("<|endoftext|>") if s.strip()]
rs = np.random.default_rng(args.seed)
order = rs.permutation(len(stories))
n_tr, n_va, n_te = args.train_stories, args.val_stories, args.test_stories
assert n_tr + n_va + n_te <= len(stories), "not enough stories for this split"
tr_s = [stories[i] for i in order[:n_tr]]
va_s = [stories[i] for i in order[n_tr:n_tr + n_va]]
te_s = [stories[i] for i in order[n_tr + n_va:n_tr + n_va + n_te]]

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)*|[.,!?;:\"-]|\n")
QUOTE_FIX = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "\u0092": "'",
                           "\u0093": '"', "\u0094": '"'})


def tokenize(text):
    text = text.translate(QUOTE_FIX).lower()
    toks = [NL if t == "\n" else t for t in TOKEN_RE.findall(text)]
    return toks


def tokenize_stories(ss):
    out = []
    for s in ss:
        out.extend(tokenize(s))
        out.append(EOS)
    return out


tr_tok, va_tok, te_tok = tokenize_stories(tr_s), tokenize_stories(va_s), tokenize_stories(te_s)

counts = collections.Counter(tr_tok)  # vocabulary comes from training stories only
itos = [UNK] + [w for w, _ in counts.most_common(args.vocab_size - 1)]
stoi = {w: i for i, w in enumerate(itos)}
V = len(itos)


def to_ids(toks):
    return np.array([stoi.get(t, 0) for t in toks], dtype=np.int64)


tr_ids, va_ids, te_ids = to_ids(tr_tok), to_ids(va_tok), to_ids(te_tok)


def make_xy(arr, L, stride):
    """Sliding window: input = L words, target = the next word."""
    n = len(arr) - L
    starts = np.arange(0, n, stride)
    idx = starts[:, None] + np.arange(L)[None, :]
    return torch.from_numpy(arr[idx]), torch.from_numpy(arr[starts + L])


Xtr, ytr = make_xy(tr_ids, args.seq_len, args.stride)
Xva, yva = make_xy(va_ids, args.seq_len, args.stride)
Xte, yte = make_xy(te_ids, args.seq_len, args.stride)

stats = {
    "stories_in_file": len(stories),
    "stories_train": len(tr_s), "stories_val": len(va_s), "stories_test": len(te_s),
    "characters_file": len(raw),
    "tokens_train": len(tr_tok), "tokens_val": len(va_tok), "tokens_test": len(te_tok),
    "unique_tokens_train": len(counts),
    "vocab_size_used": V,
    "unk_rate_train": float((tr_ids == 0).mean()),
    "unk_rate_val": float((va_ids == 0).mean()),
    "windows_train": len(Xtr), "windows_val": len(Xva), "windows_test": len(Xte),
    "top_15_tokens": counts.most_common(15),
}
print(json.dumps(stats, indent=2), flush=True)

train_bigrams = set(zip(tr_ids[:-1].tolist(), tr_ids[1:].tolist()))
# majority baseline: always predict the most frequent training target
most_common_target = int(torch.bincount(ytr).argmax())
baseline_acc = float((yva == most_common_target).float().mean())
stats["majority_baseline_top1_val"] = baseline_acc
print("majority-class baseline top-1 on val:", round(baseline_acc, 4), flush=True)


# ----------------------------------------------------------------------------
# 2. Models
# ----------------------------------------------------------------------------
class LM(nn.Module):
    """Embedding -> recurrent layer -> last-step state -> linear to vocab."""

    def __init__(self, kind, V, E, H):
        super().__init__()
        self.kind = kind
        self.emb = nn.Embedding(V, E)
        bi = kind == "BiRNN"
        cls = {"SimpleRNN": nn.RNN, "BiRNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[kind]
        kw = {"nonlinearity": "tanh"} if cls is nn.RNN else {}
        self.rnn = cls(E, H, batch_first=True, bidirectional=bi, **kw)
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(H * (2 if bi else 1), V)

    def forward(self, x):
        o, _ = self.rnn(self.drop(self.emb(x)))
        # Last time step. For BiRNN this is [forward state at last word ; backward state
        # at last word]; the backward pass only reads the given context window.
        return self.out(self.drop(o[:, -1, :]))


# ----------------------------------------------------------------------------
# 3. Training / evaluation
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, X, y, bs=1024):
    model.eval()
    lf = nn.CrossEntropyLoss(reduction="sum")
    tot, c1, c5 = 0.0, 0, 0
    for i in range(0, len(X), bs):
        xb, yb = X[i:i + bs].to(DEVICE), y[i:i + bs].to(DEVICE)
        lg = model(xb)
        tot += lf(lg, yb).item()
        top5 = lg.topk(5, dim=1).indices
        c1 += (top5[:, 0] == yb).sum().item()
        c5 += (top5 == yb[:, None]).any(dim=1).sum().item()
    n = len(X)
    loss = tot / n
    return {"loss": loss, "ppl": math.exp(loss), "top1": c1 / n, "top5": c5 / n}


def train_model(kind):
    set_seed(args.seed)  # identical init seed + identical batch order for every model
    model = LM(kind, V, args.embed_dim, args.hidden).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
    lf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(args.seed)
    hist, best, best_state = [], float("inf"), None
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        run, nb = 0.0, 0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            xb, yb = Xtr[b].to(DEVICE), ytr[b].to(DEVICE)
            opt.zero_grad()
            loss = lf(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item()
            nb += 1
        sched.step()
        v = evaluate(model, Xva, yva)
        row = {"epoch": ep, "train_loss": run / nb, "val_loss": v["loss"],
               "val_ppl": v["ppl"], "val_top1": v["top1"], "val_top5": v["top5"],
               "elapsed_s": time.time() - t0}
        hist.append(row)
        print(f"[{kind}] ep{ep} train {row['train_loss']:.3f} val {v['loss']:.3f} "
              f"ppl {v['ppl']:.1f} top1 {v['top1']:.3f} top5 {v['top5']:.3f} "
              f"t={row['elapsed_s']:.0f}s", flush=True)
        if v["loss"] < best:
            best = v["loss"]
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
    train_time = time.time() - t0
    model.load_state_dict(best_state)  # best-validation checkpoint
    val, test = evaluate(model, Xva, yva), evaluate(model, Xte, yte)
    final = {"params": n_params, "train_time_s": train_time,
             "time_per_epoch_s": train_time / args.epochs,
             "best_epoch": int(np.argmin([h["val_loss"] for h in hist])) + 1,
             **{f"val_{k}": x for k, x in val.items()},
             **{f"test_{k}": x for k, x in test.items()}}
    return model, hist, final


# ----------------------------------------------------------------------------
# 4. Generation + text metrics
# ----------------------------------------------------------------------------
def detok(words):
    s = ""
    for w in words:
        if w in (NL, EOS):
            s = s.rstrip(" ") + ("\n" if w == NL else "\n[end of story]\n")
        elif w in ".,!?;:":
            s = s.rstrip(" ") + w + " "
        elif w == "-":
            s = s.rstrip(" ") + "-"
        else:
            s += w + " "
    return s.strip()


@torch.no_grad()
def generate(model, seed_text, n_words, temperature, rng):
    model.eval()
    seed = tokenize(seed_text)
    ctx = [stoi.get(t, 0) for t in seed][-args.seq_len:]
    while len(ctx) < args.seq_len:
        ctx = [stoi[EOS]] + ctx  # pad the left with an end-of-story marker = "a story starts here"
    out = []
    for _ in range(n_words):
        x = torch.tensor([ctx[-args.seq_len:]], device=DEVICE)
        logits = model(x)[0]
        logits[0] = -1e9  # never emit <unk>
        probs = torch.softmax(logits / temperature, dim=-1).cpu().numpy().astype("float64")
        probs /= probs.sum()
        nxt = int(rng.choice(len(probs), p=probs))
        ctx.append(nxt)
        out.append(itos[nxt])
    return out


def text_metrics(words):
    w = [x for x in words if x not in (NL, EOS)]
    n = max(len(w), 1)
    d1 = len(set(w)) / n
    bi = list(zip(w[:-1], w[1:]))
    tri = list(zip(w[:-2], w[1:-1], w[2:]))
    d2 = len(set(bi)) / max(len(bi), 1)
    rep3 = 1 - len(set(tri)) / max(len(tri), 1)  # share of trigrams that repeat an earlier one
    run = best = 1
    for a, b in zip(w[:-1], w[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    ids_ = [stoi.get(x, 0) for x in w]
    valid = np.mean([(a, b) in train_bigrams for a, b in zip(ids_[:-1], ids_[1:])]) if len(ids_) > 1 else 0
    return {"distinct1": d1, "distinct2": d2, "repeat_trigram": rep3,
            "max_word_run": best, "bigram_validity": float(valid)}


SEEDS = [
    "once upon a time, there was a little girl named",
    "one day, a boy named tom went to the park and",
    "the dog was very happy because",
    "she looked at the big red ball and said,",
    "mom and dad took the family to the",
]
TEMPS = [0.5, 1.0, 1.5]
KINDS = ["SimpleRNN", "BiRNN", "LSTM", "GRU"]

if __name__ == "__main__":
    results, histories, samples, gen_rows = {}, {}, [], []
    for kind in KINDS:
        model, hist, final = train_model(kind)
        results[kind], histories[kind] = final, hist
        for si, seed_text in enumerate(SEEDS):
            for T in TEMPS:
                rng = np.random.default_rng(1000 + si)  # same sampling stream per seed
                words = generate(model, seed_text, args.gen_words, T, rng)
                m = text_metrics(words)
                m.update({"model": kind, "seed_id": si, "temperature": T})
                gen_rows.append(m)
                samples.append((kind, seed_text, T, detok(words)))
        torch.save(model.state_dict(), os.path.join(args.out, f"{kind}.pt"))

    json.dump({"dataset": stats, "config": vars(args), "results": results},
              open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    json.dump(gen_rows, open(os.path.join(args.out, "generation_metrics.json"), "w"), indent=2)

    with open(os.path.join(args.out, "history.csv"), "w") as f:
        f.write("model,epoch,train_loss,val_loss,val_ppl,val_top1,val_top5,elapsed_s\n")
        for k, h in histories.items():
            for r in h:
                f.write(f"{k},{r['epoch']},{r['train_loss']:.4f},{r['val_loss']:.4f},"
                        f"{r['val_ppl']:.2f},{r['val_top1']:.4f},{r['val_top5']:.4f},{r['elapsed_s']:.1f}\n")

    with open(os.path.join(args.out, "generated_samples.md"), "w") as f:
        f.write(f"# Generated text samples\n\nSame {len(SEEDS)} seeds and same sampling stream for every model, "
                f"{args.gen_words} words each.\n")
        for si, seed_text in enumerate(SEEDS):
            f.write(f"\n## Seed {si + 1}: `{seed_text}`\n")
            for T in TEMPS:
                f.write(f"\n### Temperature {T}\n")
                for kind, s, t, txt in samples:
                    if s == seed_text and t == T:
                        f.write(f"\n**{kind}**\n\n```\n{seed_text} | {txt}\n```\n")

    agg = {}
    for r in gen_rows:
        agg.setdefault((r["model"], r["temperature"]), []).append(r)
    agg_rows = []
    for (k, T), rs_ in agg.items():
        agg_rows.append({"model": k, "temperature": T,
                         **{m: float(np.mean([x[m] for x in rs_]))
                            for m in ["distinct1", "distinct2", "repeat_trigram",
                                      "max_word_run", "bigram_validity"]}})
    json.dump(agg_rows, open(os.path.join(args.out, "generation_summary.json"), "w"), indent=2)

    # ---- plots
    colors = {"SimpleRNN": "#d95f02", "BiRNN": "#7570b3", "LSTM": "#1b9e77", "GRU": "#e7298a"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for k, h in histories.items():
        ep = [r["epoch"] for r in h]
        ax[0].plot(ep, [r["train_loss"] for r in h], color=colors[k], label=k)
        ax[1].plot(ep, [r["val_loss"] for r in h], color=colors[k], label=k)
        ax[2].plot(ep, [r["val_top1"] for r in h], color=colors[k], label=k)
    for a, t in zip(ax, ["Training loss", "Validation loss", "Validation top-1 accuracy"]):
        a.set_title(t); a.set_xlabel("epoch"); a.grid(alpha=.3)
    ax[0].legend()
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "training_curves.png"), dpi=130); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ks = list(results)
    ax[0].bar(ks, [results[k]["val_ppl"] for k in ks], color=[colors[k] for k in ks]); ax[0].set_title("Validation perplexity (lower is better)")
    ax[1].bar(ks, [results[k]["val_top1"] for k in ks], color=[colors[k] for k in ks]); ax[1].set_title("Validation top-1 accuracy")
    ax[2].bar(ks, [results[k]["train_time_s"] for k in ks], color=[colors[k] for k in ks]); ax[2].set_title("Training time (s)")
    for a in ax: a.grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "model_comparison.png"), dpi=130); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for k in KINDS:
        rows = sorted([r for r in agg_rows if r["model"] == k], key=lambda r: r["temperature"])
        for a, m, t in zip(ax, ["distinct2", "repeat_trigram", "bigram_validity"],
                           ["Diversity (distinct-2)", "Trigram repetition", "Bigram validity (coherence proxy)"]):
            a.plot([r["temperature"] for r in rows], [r[m] for r in rows], "o-", color=colors[k], label=k)
            a.set_title(t); a.set_xlabel("temperature"); a.grid(alpha=.3)
    ax[0].legend()
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "temperature_effects.png"), dpi=130); plt.close()
    print("DONE", flush=True)
