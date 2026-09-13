import numpy as np
import matplotlib.pyplot as plt

RUT = "21.620.910-9"  # Maximiliano Palma
d = [int(c) for c in RUT if c.isdigit()][:8]

# El enunciado usa d3 para el radio de las antenas 1 y 2; d6 no se utiliza.
ANTENAS = [((d[0] * 10, d[1] * 10), d[2] * 2 + 15),
           ((d[3] * 10, d[4] * 10), d[2] * 2 + 15),
           ((d[6] * 10, d[7] * 10), 25)]

LADO = 100.0
AREA_TOTAL = 10_000.0
ESCENARIOS = [10_000, 100_000, 1_000_000, 10_000_000]

print("RUT utilizado:", RUT, "-> d1..d8 =", d)
for i, ((cx, cy), r) in enumerate(ANTENAS, 1):
    print(f"  Antena {i}: centro=({cx}, {cy})  radio={r} km")


def simular(N, seed=42, chunk=1_000_000):
    rng = np.random.default_rng(seed)
    conteos = np.zeros(4, dtype=np.int64)
    muestra = None
    for i in range(0, N, chunk):
        m = min(chunk, N - i)
        x = rng.uniform(0, LADO, m)
        y = rng.uniform(0, LADO, m)
        cob = sum(((x - cx) ** 2 + (y - cy) ** 2 <= r * r) for (cx, cy), r in ANTENAS)
        conteos += np.bincount(cob, minlength=4)
        if muestra is None:
            muestra = (x[:6000], y[:6000], cob[:6000])
    return conteos, muestra


filas, muestras = [], {}
for N in ESCENARIOS:
    c, mu = simular(N)
    assert c.sum() == N
    muestras[N] = mu
    p_util = (c[1] + c[2]) / N
    p_int = c[3] / N
    p_sin = c[0] / N
    err = AREA_TOTAL * np.sqrt(p_util * (1 - p_util) / N)
    filas.append((N, AREA_TOTAL * p_util, err, AREA_TOTAL * p_int, AREA_TOTAL * p_sin))

print(f"\n{'N':>12} {'Area util':>12} {'+/- (1 EE)':>11} {'Interferida':>13} {'Sin senal':>12}")
for N, a_u, e, a_i, a_s in filas:
    assert abs(a_u + a_i + a_s - AREA_TOTAL) < 1e-6
    print(f"{N:>12,} {a_u:>12.2f} {e:>11.2f} {a_i:>13.2f} {a_s:>12.2f}")

ref = filas[-1][1]
print(f"\nReferencia (N=10M): {ref:.2f} km2")
print(f"{'N':>12} {'Error abs.':>12} {'Error rel.':>11} {'EE teorico':>12} {'Razon EE':>10}")
previo = None
for N, a_u, e, _, _ in filas:
    razon = f"{e / previo:.3f}" if previo else "-"
    print(f"{N:>12,} {abs(a_u - ref):>12.2f} {abs(a_u - ref) / ref:>10.4%} {e:>12.2f} {razon:>10}")
    previo = e

# Paleta Okabe-Ito: separable en daltonismo (el par verde/naranja habitual no lo es).
# El tamano refuerza al color: la clase dominante "sin senal" va pequena y tenue.
CLASES = [("Sin senal (0 antenas)", '#0072b2', 2, 0.30),
          ("Servicio util (1 o 2 antenas)", '#009e73', 7, 0.90),
          ("Interferido (3 antenas)", '#d55e00', 7, 0.90)]

fig, axes = plt.subplots(2, 2, figsize=(12, 12.8))
for ax, N in zip(axes.ravel(), ESCENARIOS):
    x, y, cob = muestras[N]
    clase = np.where(cob == 0, 0, np.where(cob == 3, 2, 1))
    for k, (etq, col, s, al) in enumerate(CLASES):
        m = clase == k
        ax.scatter(x[m], y[m], c=col, s=s, alpha=al, linewidths=0,
                   label=etq if ax is axes[0, 0] else None)
    for (cx, cy), r in ANTENAS:
        ax.add_patch(plt.Circle((cx, cy), r, fill=False, ec='#1a1a1a', lw=1.6))
        ax.plot(cx, cy, marker='^', color='#1a1a1a', ms=10, mec='white', mew=1.3)
    ax.set_xlim(0, LADO); ax.set_ylim(0, LADO); ax.set_aspect('equal')
    a_u = [f[1] for f in filas if f[0] == N][0]
    ax.set_title(f"N = {N:,}   |   area util = {a_u:.1f} km2", fontsize=11)
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.grid(alpha=0.15, lw=0.5)
    ax.set_axisbelow(True)

fig.legend(loc='lower center', ncol=3, frameon=False, markerscale=3.2,
           fontsize=10, bbox_to_anchor=(0.5, 0.005))
fig.suptitle("Monte Carlo - cobertura de antenas (submuestra de 6.000 puntos por escenario)",
             fontsize=13)
fig.tight_layout(rect=[0, 0.035, 1, 1])
fig.savefig("p3_cobertura.png", dpi=110)
print("\nGrafico guardado en p3_cobertura.png")
