"""
analyse.py - statistical analysis for the GM active travel equity project

Looks at the relationship between active travel infrastructure, deprivation,
and travel mode share across GM LSOAs. Runs OLS regression and produces
summary stats and diagnostic outputs.

Usage:
    python src/analyse.py [--data data/processed/gm_lsoa_analysis.csv]
                          [--out-dir outputs]
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

warnings.filterwarnings("ignore")


def describe_by_quintile(df: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics for key variables grouped by IMD quintile.

    Parameters
    ----------
    df : pandas.DataFrame
        Analysis dataset with imd_quintile, infra_density_m_per_km2,
        active_travel_pct, and imd_score columns.

    Returns
    -------
    pandas.DataFrame
        Mean, median, and SD of infrastructure density and active travel
        mode share for each quintile (1 = most deprived).
    """
    stats = (
        df.groupby("imd_quintile")
        .agg(
            n=("lsoa_code", "count"),
            infra_density_mean=("infra_density_m_per_km2", "mean"),
            infra_density_median=("infra_density_m_per_km2", "median"),
            infra_density_sd=("infra_density_m_per_km2", "std"),
            active_travel_mean=("active_travel_pct", "mean"),
            active_travel_median=("active_travel_pct", "median"),
            active_travel_sd=("active_travel_pct", "std"),
            imd_score_mean=("imd_score", "mean"),
        )
        .round(2)
        .reset_index()
    )
    stats["imd_quintile_label"] = stats["imd_quintile"].map({
        1: "Q1 (most deprived)",
        2: "Q2",
        3: "Q3",
        4: "Q4",
        5: "Q5 (least deprived)",
    })
    return stats


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson and Spearman correlations for the main variables.

    Parameters
    ----------
    df : pandas.DataFrame
        Analysis dataset with numeric variables of interest.

    Returns
    -------
    pandas.DataFrame
        One row per variable pair with Pearson r, Spearman rho, and p-values.
    """
    from scipy.stats import spearmanr, pearsonr

    variables = {
        "imd_score": "IMD Score",
        "infra_density_m_per_km2": "Infra Density (m/km²)",
        "active_travel_pct": "Active Travel (%)",
        "cycling_pct": "Cycling (%)",
        "walking_pct": "Walking (%)",
    }

    subset = df[list(variables.keys())].dropna()
    rows = []

    for x_key, x_label in variables.items():
        for y_key, y_label in variables.items():
            if x_key >= y_key:
                continue
            x = subset[x_key]
            y = subset[y_key]
            pr_res = pearsonr(x, y)
            sr_res = spearmanr(x, y)
            # scipy >= 1.9 returns a result object; older versions return a tuple
            pr = float(pr_res.statistic if hasattr(pr_res, "statistic") else pr_res[0])
            pp = float(pr_res.pvalue if hasattr(pr_res, "pvalue") else pr_res[1])
            sr = float(sr_res.statistic if hasattr(sr_res, "statistic") else sr_res[0])
            sp = float(sr_res.pvalue if hasattr(sr_res, "pvalue") else sr_res[1])
            rows.append({
                "Variable 1": x_label,
                "Variable 2": y_label,
                "Pearson r": round(pr, 3),
                "Pearson p": round(pp, 4),
                "Spearman rho": round(sr, 3),
                "Spearman p": round(sp, 4),
            })

    return pd.DataFrame(rows)


def run_ols_regression(df: pd.DataFrame):
    """OLS regression with borough fixed effects.

    Model: active_travel_pct ~ imd_score + log_infra_density + C(borough)

    Borough fixed effects control for things like topography, bus network
    coverage, and other borough-level factors that might confound the
    deprivation-travel relationship.

    Parameters
    ----------
    df : pandas.DataFrame
        Analysis dataset with active_travel_pct, imd_score,
        log_infra_density, and borough columns.

    Returns
    -------
    object
        Fitted statsmodels RegressionResultsWrapper with HC3 SEs.
    """
    model_df = df[["active_travel_pct", "imd_score", "log_infra_density", "borough"]].dropna().copy()

    print(f"\nOLS regression on {len(model_df)} LSOAs (listwise deletion).")

    formula = "active_travel_pct ~ imd_score + log_infra_density + C(borough)"
    results = smf.ols(formula, data=model_df).fit(cov_type="HC3")
    return results


def run_ols_no_fe(df: pd.DataFrame):
    """Baseline OLS without borough fixed effects.

    Parameters
    ----------
    df : pandas.DataFrame
        Analysis dataset.

    Returns
    -------
    object
        Fitted statsmodels RegressionResultsWrapper.
    """
    model_df = df[["active_travel_pct", "imd_score", "log_infra_density"]].dropna().copy()
    return smf.ols("active_travel_pct ~ imd_score + log_infra_density", data=model_df).fit(cov_type="HC3")


def regression_diagnostics(results) -> dict:
    """Run diagnostic tests on the fitted regression model.

    Parameters
    ----------
    results : statsmodels RegressionResults

    Returns
    -------
    dict
        Keys: breusch_pagan_stat, breusch_pagan_p, durbin_watson, vif.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    resid = results.resid

    bp_lm, bp_p, _, _ = het_breuschpagan(resid, results.model.exog)
    dw = durbin_watson(resid)

    try:
        exog = results.model.exog
        names = results.model.exog_names
        vif_vals = {
            names[i]: round(float(variance_inflation_factor(exog, i)), 2)
            for i in range(len(names))
            if names[i] != "Intercept" and not names[i].startswith("C(borough)")
        }
    except Exception:
        vif_vals = {}

    return {
        "breusch_pagan_stat": round(float(bp_lm), 4),
        "breusch_pagan_p": round(float(bp_p), 4),
        "durbin_watson": round(float(dw), 4),
        "vif": vif_vals,
    }


def format_regression_summary(results_fe, results_base, diag: dict) -> str:
    """Format regression results as a readable plain-text summary.

    Parameters
    ----------
    results_fe : statsmodels RegressionResults
        Borough fixed-effects model.
    results_base : statsmodels RegressionResults
        Baseline model (no fixed effects).
    diag : dict
        Diagnostics from regression_diagnostics().

    Returns
    -------
    str
        Formatted text.
    """
    lines = [
        "=" * 72,
        "GM ACTIVE TRAVEL EQUITY - OLS REGRESSION SUMMARY",
        "=" * 72,
        "",
        "DEPENDENT VARIABLE: Active travel mode share (% of workers)",
        "",
        "-- BASELINE MODEL (no borough fixed effects) --",
        results_base.summary().as_text(),
        "",
        "-- FULL MODEL (borough fixed effects, HC3 SEs) --",
        results_fe.summary().as_text(),
        "",
        "-- KEY COEFFICIENTS --",
    ]

    fe_params = results_fe.params
    fe_ci = results_fe.conf_int()
    fe_pval = results_fe.pvalues

    for term in ["imd_score", "log_infra_density"]:
        if term in fe_params:
            lines.append(
                f"  {term:30s}  coef={fe_params[term]:+.4f}  "
                f"95% CI [{fe_ci.loc[term, 0]:+.4f}, {fe_ci.loc[term, 1]:+.4f}]  "
                f"p={fe_pval[term]:.4f}"
            )

    lines += [
        "",
        "-- MODEL FIT --",
        f"  R2           : {results_fe.rsquared:.4f}",
        f"  Adj. R2      : {results_fe.rsquared_adj:.4f}",
        f"  F-statistic  : {results_fe.fvalue:.2f}  (p={results_fe.f_pvalue:.4e})",
        f"  N            : {int(results_fe.nobs)}",
        "",
        "-- DIAGNOSTICS --",
        f"  Breusch-Pagan (LM): stat={diag['breusch_pagan_stat']}  "
        f"p={diag['breusch_pagan_p']}",
        f"  Durbin-Watson      : {diag['durbin_watson']}",
    ]

    if diag["vif"]:
        lines.append("  VIF:")
        for term, vif in diag["vif"].items():
            lines.append(f"    {term:30s}  {vif}")

    lines += ["", "=" * 72]
    return "\n".join(lines)


def main(data_path: str = "data/processed/gm_lsoa_analysis.csv",
         out_dir: str = "outputs") -> None:
    """Run the full statistical analysis pipeline.

    Parameters
    ----------
    data_path : str
        Path to the processed analysis CSV from clean.py.
    out_dir : str
        Directory to write output files.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} LSOAs from {data_path}")

    print("\n--- Summary statistics by IMD quintile ---")
    quintile_stats = describe_by_quintile(df)
    print(quintile_stats.to_string(index=False))
    quintile_stats.to_csv(out / "quintile_summary.csv", index=False)

    print("\n--- Correlation matrix ---")
    corr = compute_correlations(df)
    print(corr.to_string(index=False))
    corr.to_csv(out / "correlations.csv", index=False)

    results_fe = run_ols_regression(df)
    results_base = run_ols_no_fe(df)
    diag = regression_diagnostics(results_fe)

    summary_text = format_regression_summary(results_fe, results_base, diag)
    print(summary_text)

    summary_path = out / "regression_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Statistical analysis for GM active travel equity."
    )
    parser.add_argument(
        "--data", default="data/processed/gm_lsoa_analysis.csv",
    )
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()
    main(data_path=args.data, out_dir=args.out_dir)
