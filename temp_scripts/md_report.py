"""Run fit_all_models on a sample and print a markdown + LaTeX block
with the per-model parameter estimates and the plot reference.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fit_all import fit_all_models

# Map parameter names to LaTeX symbols.
TEX = {
    "beta"   : r"\beta",
    "gamma"  : r"\gamma",
    "sigma"  : r"\sigma",
    "alpha"  : r"\alpha",
    "beta1"  : r"\beta_1",
    "beta2"  : r"\beta_2",
    "beta3"  : r"\beta_3",
    "beta4"  : r"\beta_4",
    "lambda1": r"\lambda_1",
    "lambda2": r"\lambda_2",
    "mu1"    : r"\mu_1",
    "mu2"    : r"\mu_2",
    "mu3"    : r"\mu_3",
    "mu_e"   : r"\mu_e",
    "E0"     : r"E_0",
    "D0"     : r"D_0",
}


def fmt_num(x: float) -> str:
    """Compact scientific/decimal format."""
    if x == 0.0:
        return "0"
    ax = abs(x)
    if 1e-3 <= ax < 1e4:
        return f"{x:.4g}"
    return f"{x:.3e}".replace("e+0", "e+").replace("e-0", "e-")


def md_report(path: str) -> str:
    result = fit_all_models(path)
    sample = result["sample"]
    N      = int(sample["params"]["population"])
    n      = sample.get("n_points") or len(sample["time"])
    title  = (sample["params"].get("event")
              or sample["params"].get("country")
              or os.path.basename(path))
    plot   = result["plot_path"].replace("\\", "/")
    fname  = os.path.basename(path)

    out: list[str] = []
    out.append(f"### {fname}  ({title}, $N={N:,}$, {n} points)")
    out.append("")
    out.append(f"![All models fit to {fname}]({plot})")
    out.append("")
    out.append("**Fit quality (sorted best to worst):**")
    out.append("")
    out.append("| Rank | Model | RMSE | RMSE / peak | cond($J^\\top J$) |")
    out.append("|---|---|---|---|---|")
    peak = max(sample["compartments"]["I"])
    for i, r in enumerate(result["results"], start=1):
        cond_str = "—" if not isinstance(r["jac_cond"], float) or r["jac_cond"] != r["jac_cond"] \
                   else f"{r['jac_cond']:.2e}"
        out.append(f"| {i} | **{r['name']}** | {r['rmse']:,.2f} | "
                   f"{100*r['rmse']/peak:.2f}% | {cond_str} |")
    out.append("")

    out.append("**Parameter estimates** $\\hat\\theta_k \\pm \\sigma_k$ "
               "(Theorem 38; $\\sigma_k=\\sqrt{(A^{-1})_{kk}\\,(\\beta-\\Phi(\\hat\\theta))}$ "
               "with $A=J^\\top J$, $\\beta=\\Phi(\\hat\\theta)\\,n/(n-p)$):")
    out.append("")

    for r in result["results"]:
        rate_rows = [(n, v, s) for n, v, s in
                     zip(r["param_names"], r["theta"], r["se"])
                     if n in r["rates"]]
        latent_rows = [(n, v, s) for n, v, s in
                       zip(r["param_names"], r["theta"], r["se"])
                       if n not in r["rates"]]

        out.append(f"**{r['name']}** — RMSE = {r['rmse']:,.2f}")
        out.append("")
        out.append("| Параметр | Оцінка $\\hat\\theta_k$ | Похибка $\\sigma_k$ |")
        out.append("|---|---|---|")
        for name, val, se in rate_rows:
            tex = TEX.get(name, name)
            out.append(f"| ${tex}$ | ${fmt_num(val)}$ | ${fmt_num(se)}$ |")
        if latent_rows:
            out.append("")
            latents = ", ".join(
                f"${TEX.get(name, name)} = {fmt_num(val)} \\pm {fmt_num(se)}$"
                for name, val, se in latent_rows
            )
            out.append(f"Латентні початкові умови (nuisance-параметри): {latents}")
        out.append("")

    return "\n".join(out)


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "samples/flu1978_school.json",
        "samples/COVID_Germany_2020.json",
    ]
    blocks = [md_report(p) for p in paths]
    out_path = "temp_scripts/md_report.out"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(blocks))
    print(f"\nWrote markdown report -> {out_path}")
