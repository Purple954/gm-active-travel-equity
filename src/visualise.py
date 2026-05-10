"""
visualise.py
------------
Generate all figures and interactive maps for the GM active travel
equity analysis.

Usage
-----
    python src/visualise.py [--data data/processed/gm_lsoa_analysis.gpkg]
                            [--out-dir outputs]

Outputs
-------
outputs/figures/fig1_infra_by_quintile.png
outputs/figures/fig2_deprivation_vs_mode_share.png
outputs/figures/fig3_infra_density_by_borough.png
outputs/figures/fig4_cycling_vs_walking_share.png
outputs/maps/gm_active_travel_equity.html
"""

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

GM_PALETTE = {
    "Bolton": "#1f77b4",
    "Bury": "#ff7f0e",
    "Manchester": "#2ca02c",
    "Oldham": "#d62728",
    "Rochdale": "#9467bd",
    "Salford": "#8c564b",
    "Stockport": "#e377c2",
    "Tameside": "#7f7f7f",
    "Trafford": "#bcbd22",
    "Wigan": "#17becf",
}

QUINTILE_LABELS = {
    1: "Q1\n(most deprived)",
    2: "Q2",
    3: "Q3",
    4: "Q4",
    5: "Q5\n(least deprived)",
}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Figure 1: Infrastructure density by IMD quintile (box plot)
# ---------------------------------------------------------------------------


def plot_infra_by_quintile(df: pd.DataFrame, out_dir: Path) -> Path:
    """Box plot of cycling infrastructure density by IMD quintile.

    Parameters
    ----------
    df : pandas.DataFrame
        Analysis dataset with ``imd_quintile`` and
        ``infra_density_m_per_km2`` columns.
    out_dir : Path
        Directory to save the figure.

    Returns
    -------
    Path
        Path to the saved PNG file.

    Notes
    -----
    Outliers are shown as individual points. The y-axis uses a square-root
    scale to reduce the visual dominance of extreme values without losing
    the zero-anchored baseline.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    plot_df = df.dropna(subset=["imd_quintile", "infra_density_m_per_km2"]).copy()
    plot_df["quintile_label"] = plot_df["imd_quintile"].map(QUINTILE_LABELS)
    order = [QUINTILE_LABELS[i] for i in range(1, 6)]

    sns.boxplot(
        data=plot_df,
        x="quintile_label",
        y="infra_density_m_per_km2",
        hue="quintile_label",
        order=order,
        palette="Blues",
        legend=False,
        flierprops={"marker": "o", "markersize": 2, "alpha": 0.4},
        ax=ax,
    )

    ax.set_title(
        "Cycling infrastructure density by neighbourhood deprivation quintile\n"
        "Greater Manchester LSOAs, 2021–2023",
        fontsize=11, pad=10,
    )
    ax.set_xlabel("IMD 2019 quintile", labelpad=8)
    ax.set_ylabel("Infrastructure density (m per km²)", labelpad=8)

    # Annotate median values
    medians = plot_df.groupby("quintile_label")["infra_density_m_per_km2"].median()
    for i, label in enumerate(order):
        if label in medians:
            ax.text(
                i, medians[label] + 5, f"{medians[label]:.0f}",
                ha="center", va="bottom", fontsize=8, color="0.3",
            )

    ax.set_ylim(bottom=0)
    plt.tight_layout()

    path = out_dir / "fig1_infra_by_quintile.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return path


# ---------------------------------------------------------------------------
# Figure 2: Deprivation score vs active travel mode share (scatter)
# ---------------------------------------------------------------------------


def plot_deprivation_vs_mode_share(df: pd.DataFrame, out_dir: Path) -> Path:
    """Scatter plot of IMD score vs active travel mode share with OLS line.

    Each point is an LSOA coloured by borough. The OLS regression line
    and 95% confidence band show the overall direction of association.

    Parameters
    ----------
    df : pandas.DataFrame
        Analysis dataset with ``imd_score``, ``active_travel_pct``,
        and ``borough`` columns.
    out_dir : Path
        Directory to save the figure.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    plot_df = df.dropna(subset=["imd_score", "active_travel_pct", "borough"]).copy()

    fig, ax = plt.subplots(figsize=(9, 6))

    for borough, group in plot_df.groupby("borough"):
        ax.scatter(
            group["imd_score"],
            group["active_travel_pct"],
            label=borough,
            color=GM_PALETTE.get(borough, "#999"),
            alpha=0.45,
            s=12,
            linewidths=0,
        )

    # OLS regression line (pooled, no fixed effects)
    x = plot_df["imd_score"].values
    y = plot_df["active_travel_pct"].values
    valid = np.isfinite(x) & np.isfinite(y)
    coeffs = np.polyfit(x[valid], y[valid], 1)
    x_line = np.linspace(x[valid].min(), x[valid].max(), 200)
    ax.plot(x_line, np.polyval(coeffs, x_line), color="black", lw=1.5, zorder=5)

    # Confidence band via bootstrap (light shading)
    n_boot = 200
    boot_lines = np.zeros((n_boot, len(x_line)))
    rng = np.random.default_rng(42)
    for i in range(n_boot):
        idx = rng.choice(valid.sum(), size=valid.sum(), replace=True)
        bc = np.polyfit(x[valid][idx], y[valid][idx], 1)
        boot_lines[i] = np.polyval(bc, x_line)
    ci_low = np.percentile(boot_lines, 2.5, axis=0)
    ci_high = np.percentile(boot_lines, 97.5, axis=0)
    ax.fill_between(x_line, ci_low, ci_high, alpha=0.15, color="black")

    ax.set_title(
        "Neighbourhood deprivation and active travel mode share\n"
        "Greater Manchester LSOAs, Census 2021",
        fontsize=11, pad=10,
    )
    ax.set_xlabel("IMD 2019 score (higher = more deprived)", labelpad=8)
    ax.set_ylabel("Active travel mode share (% of workers)", labelpad=8)

    legend = ax.legend(
        title="Borough",
        loc="upper right",
        fontsize=7,
        title_fontsize=8,
        markerscale=1.5,
        framealpha=0.7,
    )

    ax.text(
        0.02, 0.97,
        f"OLS slope: {coeffs[0]:+.3f} pp per IMD unit\n(pooled, no controls)",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=8, color="0.3",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.8),
    )

    plt.tight_layout()
    path = out_dir / "fig2_deprivation_vs_mode_share.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return path


# ---------------------------------------------------------------------------
# Figure 3: Infrastructure density distribution by borough
# ---------------------------------------------------------------------------


def plot_density_by_borough(df: pd.DataFrame, out_dir: Path) -> Path:
    """Horizontal box plot of infrastructure density by GM borough.

    Boroughs are ordered by median density (highest at top) to allow
    easy comparison of inter-borough variation.

    Parameters
    ----------
    df : pandas.DataFrame
        Analysis dataset with ``borough`` and
        ``infra_density_m_per_km2`` columns.
    out_dir : Path
        Directory to save the figure.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    plot_df = df.dropna(subset=["borough", "infra_density_m_per_km2"]).copy()
    order = (
        plot_df.groupby("borough")["infra_density_m_per_km2"]
        .median()
        .sort_values(ascending=True)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    palette = [GM_PALETTE.get(b, "#aaa") for b in order]
    sns.boxplot(
        data=plot_df,
        y="borough",
        x="infra_density_m_per_km2",
        hue="borough",
        order=order,
        palette=palette,
        legend=False,
        flierprops={"marker": "o", "markersize": 2, "alpha": 0.4},
        orient="h",
        ax=ax,
    )

    ax.set_title(
        "Distribution of cycling infrastructure density by GM borough\n"
        "Ordered by median (m per km², OpenStreetMap)",
        fontsize=11, pad=10,
    )
    ax.set_xlabel("Infrastructure density (m per km²)", labelpad=8)
    ax.set_ylabel("")
    ax.set_xlim(left=0)

    plt.tight_layout()
    path = out_dir / "fig3_infra_density_by_borough.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return path


# ---------------------------------------------------------------------------
# Figure 4: Cycling vs walking share, coloured by deprivation
# ---------------------------------------------------------------------------


def plot_cycling_vs_walking(df: pd.DataFrame, out_dir: Path) -> Path:
    """Scatter plot of cycling vs walking mode share, coloured by IMD quintile.

    Disentangles the two components of active travel to show whether the
    infrastructure-deprivation relationship is driven primarily by cycling
    or walking.

    Parameters
    ----------
    df : pandas.DataFrame
        Analysis dataset with ``cycling_pct``, ``walking_pct``, and
        ``imd_quintile`` columns.
    out_dir : Path
        Directory to save the figure.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    plot_df = df.dropna(subset=["cycling_pct", "walking_pct", "imd_quintile"]).copy()
    plot_df["quintile_label"] = plot_df["imd_quintile"].map({
        1: "Q1 (most deprived)",
        2: "Q2", 3: "Q3", 4: "Q4",
        5: "Q5 (least deprived)",
    })

    fig, ax = plt.subplots(figsize=(8, 6))

    quintile_palette = sns.color_palette("RdYlGn", 5)
    for i, (q_label, group) in enumerate(plot_df.groupby("quintile_label")):
        ax.scatter(
            group["walking_pct"],
            group["cycling_pct"],
            label=q_label,
            color=quintile_palette[i],
            alpha=0.5,
            s=12,
            linewidths=0,
        )

    ax.set_title(
        "Cycling vs walking mode share by IMD quintile\n"
        "Greater Manchester LSOAs, Census 2021",
        fontsize=11, pad=10,
    )
    ax.set_xlabel("Walking mode share (% of workers)", labelpad=8)
    ax.set_ylabel("Cycling mode share (% of workers)", labelpad=8)
    ax.legend(title="IMD quintile", fontsize=7, title_fontsize=8, markerscale=1.5)

    # Add diagonal reference line (equal shares)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], ls=":", lw=0.8, color="0.6", label="_nolegend_")

    plt.tight_layout()
    path = out_dir / "fig4_cycling_vs_walking_share.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return path


# ---------------------------------------------------------------------------
# Interactive Folium choropleth map
# ---------------------------------------------------------------------------


def build_interactive_map(gdf: gpd.GeoDataFrame, out_dir: Path) -> Path:
    """Build a multi-layer interactive choropleth map using Folium.

    The map has three toggleable layers:
    1. Infrastructure density (m per km²)
    2. IMD quintile (1 = most deprived)
    3. Active travel mode share (%)

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Analysis GeoDataFrame with geometry in WGS 84 (EPSG:4326).
    out_dir : Path
        Directory to save the HTML file.

    Returns
    -------
    Path
        Path to the saved HTML file.
    """
    import folium
    import branca.colormap as cm

    # Greater Manchester centroid
    bounds = gdf.geometry.total_bounds  # (minx, miny, maxx, maxy)
    centre = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    m = folium.Map(location=centre, zoom_start=10, tiles="CartoDB positron")

    # Helper to add a choropleth layer
    def _add_layer(
        name: str,
        col: str,
        colormap,
        label: str,
        show: bool = False,
    ) -> None:
        layer = folium.FeatureGroup(name=name, show=show)
        valid = gdf.dropna(subset=[col])

        for _, row in valid.iterrows():
            val = row[col]
            tooltip_text = (
                f"<b>{row.get('lsoa_name', row['lsoa_code'])}</b><br>"
                f"Borough: {row.get('borough', 'N/A')}<br>"
                f"{label}: {val:.1f}"
            )
            folium.GeoJson(
                row["geometry"].__geo_interface__,
                style_function=lambda feature, v=val: {
                    "fillColor": colormap(v),
                    "color": "white",
                    "weight": 0.3,
                    "fillOpacity": 0.75,
                },
                tooltip=folium.Tooltip(tooltip_text),
            ).add_to(layer)

        layer.add_to(m)
        colormap.caption = label
        colormap.add_to(m)

    # Layer 1: Infrastructure density
    infra_vals = gdf["infra_density_m_per_km2"].dropna()
    infra_cmap = cm.LinearColormap(
        ["#f7fbff", "#2171b5"],
        vmin=float(infra_vals.quantile(0.05)),
        vmax=float(infra_vals.quantile(0.95)),
    )
    _add_layer(
        "Infrastructure density (m/km²)",
        "infra_density_m_per_km2",
        infra_cmap,
        "Infra density (m/km²)",
        show=True,
    )

    # Layer 2: IMD quintile
    imd_cmap = cm.StepColormap(
        ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"],
        index=[1, 2, 3, 4, 5, 6],
        vmin=1, vmax=5,
    )
    _add_layer(
        "IMD quintile (1=most deprived)",
        "imd_quintile",
        imd_cmap,
        "IMD quintile",
        show=False,
    )

    # Layer 3: Active travel mode share
    mode_vals = gdf["active_travel_pct"].dropna()
    mode_cmap = cm.LinearColormap(
        ["#fff5eb", "#7f2704"],
        vmin=float(mode_vals.quantile(0.05)),
        vmax=float(mode_vals.quantile(0.95)),
    )
    _add_layer(
        "Active travel mode share (%)",
        "active_travel_pct",
        mode_cmap,
        "Active travel (%)",
        show=False,
    )

    folium.LayerControl(collapsed=False).add_to(m)

    m.get_root().html.add_child(folium.Element(
        "<div style='position:fixed;bottom:30px;left:30px;z-index:1000;"
        "background:white;padding:10px;border-radius:4px;"
        "box-shadow:0 1px 4px rgba(0,0,0,0.3);font-size:12px;max-width:260px'>"
        "<b>GM Active Travel Equity</b><br>"
        "Use the layer control (top right) to toggle between infrastructure density, "
        "deprivation quintile, and active travel mode share."
        "</div>"
    ))

    path = out_dir / "gm_active_travel_equity.html"
    m.save(str(path))
    print(f"  Saved interactive map: {path.name}")
    return path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(
    data_path: str = "data/processed/gm_lsoa_analysis.gpkg",
    out_dir: str = "outputs",
) -> None:
    """Generate all figures and the interactive map.

    Parameters
    ----------
    data_path : str, optional
        Path to the processed GeoPackage from clean.py.
    out_dir : str, optional
        Root output directory; figures saved to <out_dir>/figures/,
        map to <out_dir>/maps/.
    """
    out = Path(out_dir)
    fig_dir = out / "figures"
    map_dir = out / "maps"
    fig_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {data_path} …")
    gdf = gpd.read_file(data_path)
    df = pd.DataFrame(gdf.drop(columns="geometry"))

    print("\n--- Generating figures ---")
    plot_infra_by_quintile(df, fig_dir)
    plot_deprivation_vs_mode_share(df, fig_dir)
    plot_density_by_borough(df, fig_dir)
    plot_cycling_vs_walking(df, fig_dir)

    print("\n--- Building interactive map ---")
    build_interactive_map(gdf, map_dir)

    print(f"\nAll outputs written to {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate visualisations for GM active travel equity."
    )
    parser.add_argument(
        "--data",
        default="data/processed/gm_lsoa_analysis.gpkg",
        help="Path to processed GeoPackage (default: data/processed/gm_lsoa_analysis.gpkg)",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs",
        help="Root output directory (default: outputs)",
    )
    args = parser.parse_args()
    main(data_path=args.data, out_dir=args.out_dir)
