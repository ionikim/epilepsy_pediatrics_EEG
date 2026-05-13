import time
import numpy as np
import matplotlib.pyplot as plt


def compute_nmi(a, b):
    ca, cb = np.unique(a), np.unique(b)
    pa = np.array([(a == x).mean() for x in ca])
    pb = np.array([(b == x).mean() for x in cb])
    P  = np.zeros((len(ca), len(cb)))
    for i, x in enumerate(ca):
        for j, y in enumerate(cb):
            P[i, j] = ((a == x) & (b == y)).mean()
    mi = sum(P[i, j] * np.log(P[i, j] / (pa[i] * pb[j]))
             for i in range(len(ca)) for j in range(len(cb)) if P[i, j] > 0)
    ha = -sum(p * np.log(p) for p in pa if p > 0)
    hb = -sum(p * np.log(p) for p in pb if p > 0)
    return 2 * mi / (ha + hb) if ha + hb > 0 else 1.0


def sweet_spot_experiment(run_fn, terminations, n_runs=5):
    cpu_times, nmi_scores = [], []
    for t in terminations:
        runs = []
        t0 = time.perf_counter()
        for r in range(n_runs):
            runs.append(run_fn(t, seed=r))
        cpu_times.append(time.perf_counter() - t0)
        nmis = [compute_nmi(runs[0], x) for x in runs[1:]]
        nmi_scores.append(np.mean(nmis))
    return np.array(cpu_times), np.array(nmi_scores)


def plot_sweet_spot(x, cpu, nmi, title, xlabel, out):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("CPU time [s]", color="crimson")
    ax1.plot(x, cpu, color="crimson", marker="o")
    ax1.tick_params(axis="y", labelcolor="crimson")

    ax2 = ax1.twinx()
    ax2.set_ylabel("NMI (stability)", color="seagreen")
    ax2.plot(x, nmi, color="seagreen", marker="s")
    ax2.tick_params(axis="y", labelcolor="seagreen")
    ax2.set_ylim(0, 1.05)

    idx = np.argmax(np.gradient(nmi) / (np.gradient(cpu) + 1e-9))
    ax1.axvline(x[idx], linestyle="--", color="black", alpha=0.6)
    ax1.text(x[idx], cpu[idx], " sweet spot", va="bottom")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.show()
