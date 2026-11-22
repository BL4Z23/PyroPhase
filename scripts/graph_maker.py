"""
Graph maker: plots two lines against a shared x axis using seaborn.

Expected CSV columns (header names are configurable below):
    PERCENTAGE COVERAGE  -> x axis
    SPANNED PERCENTAGE   -> line 1
    COMPLETE PERCENTAGE  -> line 2

Usage:
    python graph_maker.py data.csv
    python graph_maker.py data.csv -o chart.png --title "Coverage report"
    python graph_maker.py data.csv --smooth 15 --linewidth 5
    python graph_maker.py --demo
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Change these if your CSV uses different header names
X_COL = "percentage_coverage"
SPANNED_COL = "spanned_percentage"
COMPLETE_COL = "complete_percentage"

# Dark-mode palette
BG = "#0F1117"
FG = "#E6E8EB"
GRID = "#2A2F3A"
COLORS = {"Spanned %": "#2ECC71", "Complete %": "#E5383B"}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in (X_COL, SPANNED_COL, COMPLETE_COL) if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing column(s): {missing}. Found: {list(df.columns)}")
    return df[[X_COL, SPANNED_COL, COMPLETE_COL]].sort_values(X_COL).reset_index(drop=True)


def demo_data() -> pd.DataFrame:
    x = np.linspace(0, 100, 200)
    return pd.DataFrame(
        {
            X_COL: x,
            SPANNED_COL: 100 / (1 + np.exp(-(x - 65) / 5)),
            COMPLETE_COL: 100 / (1 + np.exp(-(x - 90) / 2.5)),
        }
    )


def smooth(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Centred rolling mean, kept within 0-100 and non-decreasing."""
    if window <= 1:
        return df
    out = df.copy()
    for col in (SPANNED_COL, COMPLETE_COL):
        s = out[col].rolling(window, center=True, min_periods=1).mean()
        s = s.clip(0, 100)
        out[col] = np.maximum.accumulate(s.to_numpy())
    return out


def make_graph(df, output, title, show, linewidth):
    sns.set_theme(
        style="darkgrid",
        context="talk",
        font_scale=1.1,
        rc={
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.edgecolor": FG,
            "axes.labelcolor": FG,
            "text.color": FG,
            "xtick.color": FG,
            "ytick.color": FG,
            "grid.color": GRID,
        },
    )

    long_df = df.melt(
        id_vars=X_COL,
        value_vars=[SPANNED_COL, COMPLETE_COL],
        var_name="Series",
        value_name="Percentage",
    )
    long_df["Series"] = long_df["Series"].map(
        {SPANNED_COL: "Spanned %", COMPLETE_COL: "Complete %"}
    )

    fig, ax = plt.subplots(figsize=(11, 6.5))
    sns.lineplot(
        data=long_df,
        x=X_COL,
        y="Percentage",
        hue="Series",
        palette=COLORS,
        linewidth=linewidth,
        estimator=None,
        errorbar=None,
        solid_capstyle="round",
        solid_joinstyle="round",
        ax=ax,
    )

    ax.set_title(title, pad=16, fontweight="bold")
    ax.set_xlabel("Percentage coverage (%)", fontweight="bold")
    ax.set_ylabel("Percentage (%)", fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.grid(True, color=GRID, linewidth=1)
    legend = ax.legend(title=None, loc="upper left", frameon=True, framealpha=1)
    legend.get_frame().set_facecolor(BG)
    legend.get_frame().set_edgecolor(GRID)
    for text in legend.get_texts():
        text.set_color(FG)
    for spine in ax.spines.values():
        spine.set_color(FG)
        spine.set_linewidth(1.5)
    ax.tick_params(width=1.5, color=FG)

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    print(f"Saved chart to {output}")
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Seaborn two-line graph maker")
    parser.add_argument("csv", nargs="?", help="Path to CSV file")
    parser.add_argument("-o", "--output", default="graph.png", help="Output image")
    parser.add_argument("--title", default="Spanned vs Complete by Coverage")
    parser.add_argument("--smooth", type=int, default=15,
                        help="Smoothing window in rows (1 = off, default 15)")
    parser.add_argument("--linewidth", type=float, default=4.5)
    parser.add_argument("--demo", action="store_true", help="Use sample data")
    parser.add_argument("--no-show", action="store_true", help="Don't open a window")
    args = parser.parse_args()

    if args.demo:
        df = demo_data()
    elif args.csv:
        df = load_data(args.csv)
    else:
        parser.error("Provide a CSV path or use --demo")

    df = smooth(df, args.smooth)
    make_graph(df, args.output, args.title, not args.no_show, args.linewidth)


if __name__ == "__main__":
    main()
