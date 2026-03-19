import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PAPER = "#f7f4ee"
AXIS_BG = "#fbf9f5"
GRID = "#ddd7cc"
TEXT = "#2c2823"
EDGE = "#2d261f"

SERIES_ORDER = [
    "Synthetic TTS",
    "Real Audio person1",
    "Real Audio person2",
]

SERIES_COLORS = {
    "Synthetic TTS": "#dd7f00",
    "Real Audio person1": "#1c7f77",
    "Real Audio person2": "#3fc4b7",
}

METRIC_SPECS = [
    ("pass_row_rate", "Overall"),
    ("tool_use_correct_pass_rate", "Tool"),
    ("instruction_following_pass_rate", "Instruction"),
    ("kb_grounding_pass_rate", "KB"),
    ("ambiguity_handling_pass_rate", "Ambiguity"),
    ("state_tracking_pass_rate", "State"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a warm editorial grouped bar chart for the conversation-bench "
            "audio repeat batch. The chart always excludes turn_taking and always "
            "includes repeat-to-repeat error bars."
        )
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        required=True,
        help="Path to the batch results directory containing all_runs.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path. Defaults to <batch-dir>/metrics_avg_barplot.png.",
    )
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Avenir Next.ttc",
                "/System/Library/Fonts/HelveticaNeue.ttc",
                "/System/Library/Fonts/Helvetica.ttc",
            ]
        )
    else:
        candidates.extend(
            [
                "/System/Library/Fonts/Avenir Next.ttc",
                "/System/Library/Fonts/Avenir.ttc",
                "/System/Library/Fonts/HelveticaNeue.ttc",
                "/System/Library/Fonts/Helvetica.ttc",
                "/Library/Fonts/Arial Unicode.ttf",
            ]
        )

    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_centered(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    width, height = text_size(draw, text, font)
    draw.text((x - width / 2, y - height / 2), text, font=font, fill=fill)


def build_metric_rows(all_runs_df: pd.DataFrame) -> list[dict]:
    rows = []
    for label in SERIES_ORDER:
        series_df = all_runs_df[all_runs_df["audio_label"] == label].copy()
        if series_df.empty:
            raise ValueError(f"Missing runs for audio label: {label}")

        values = []
        for column_name, display_name in METRIC_SPECS:
            numeric = pd.to_numeric(series_df[column_name], errors="coerce").dropna()
            if numeric.empty:
                mean_value = 0.0
                std_value = 0.0
            else:
                mean_value = float(numeric.mean()) * 100.0
                std_value = float(numeric.std(ddof=0)) * 100.0 if len(numeric) > 1 else 0.0

            values.append(
                {
                    "key": column_name,
                    "label": display_name,
                    "mean_percent": mean_value,
                    "std_percent": std_value,
                }
            )

        failed_turns_text = "; ".join(
            f"r{int(row.repeat)}:{row.failed_turns}"
            for row in series_df.sort_values("repeat").itertuples()
        )
        rows.append(
            {
                "label": label,
                "values": values,
                "failed_turns_text": failed_turns_text,
            }
        )
    return rows


def render_chart(rows: list[dict], output_path: Path) -> None:
    metrics = [metric["label"] for metric in rows[0]["values"]]

    width = 2160
    height = 1380
    margin_left = 180
    margin_right = 90
    margin_top = 225
    margin_bottom = 255
    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    title_font = load_font(54)
    subtitle_font = load_font(24)
    axis_font = load_font(24)
    tick_font = load_font(22)
    label_font = load_font(18)
    legend_font = load_font(22)
    small_font = load_font(17)

    image = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        [(plot_left, plot_top), (plot_right, plot_bottom)],
        radius=26,
        fill=AXIS_BG,
        outline=GRID,
        width=2,
    )

    draw.text(
        (plot_left, 55),
        "conversation bench metrics by audio source",
        font=title_font,
        fill=TEXT,
    )
    draw.text(
        (plot_left, 118),
        "gpt-realtime-1.5, avg of 3 repeats, rehydrated, no vad, skip-turn-taking",
        font=subtitle_font,
        fill="#5e564d",
    )

    legend_y = 168
    legend_x = plot_left
    for row in rows:
        color = SERIES_COLORS[row["label"]]
        draw.rounded_rectangle(
            [(legend_x, legend_y), (legend_x + 28, legend_y + 18)],
            radius=4,
            fill=color,
            outline=EDGE,
            width=2,
        )
        draw.text((legend_x + 40, legend_y - 6), row["label"], font=legend_font, fill=TEXT)
        legend_x += 360

    for pct_value in range(0, 101, 20):
        y = plot_bottom - (pct_value / 100.0) * plot_height
        draw.line([(plot_left, y), (plot_right, y)], fill=GRID, width=2)
        label = f"{pct_value}%"
        label_width, label_height = text_size(draw, label, tick_font)
        draw.text(
            (plot_left - label_width - 18, y - label_height / 2),
            label,
            font=tick_font,
            fill=TEXT,
        )

    draw.line([(plot_left, plot_top), (plot_left, plot_bottom)], fill=EDGE, width=3)
    draw.line([(plot_left, plot_bottom), (plot_right, plot_bottom)], fill=EDGE, width=3)

    group_count = len(metrics)
    group_width = plot_width / group_count
    bar_width = 68
    inner_gap = 18
    error_bar_cap = 10

    for metric_index, metric_label in enumerate(metrics):
        group_center = plot_left + group_width * metric_index + group_width / 2
        offsets = [-(bar_width + inner_gap), 0, bar_width + inner_gap]
        for row, offset in zip(rows, offsets):
            metric_row = row["values"][metric_index]
            mean_value = metric_row["mean_percent"]
            std_value = metric_row["std_percent"]
            lower_value = max(0.0, mean_value - std_value)
            upper_value = min(100.0, mean_value + std_value)

            x0 = group_center + offset - bar_width / 2
            x1 = x0 + bar_width
            y1 = plot_bottom
            y0 = plot_bottom - (mean_value / 100.0) * plot_height
            draw.rounded_rectangle(
                [(x0, y0), (x1, y1)],
                radius=10,
                fill=SERIES_COLORS[row["label"]],
                outline=EDGE,
                width=3,
            )

            err_low = plot_bottom - (lower_value / 100.0) * plot_height
            err_high = plot_bottom - (upper_value / 100.0) * plot_height
            center_x = (x0 + x1) / 2
            draw.line([(center_x, err_high), (center_x, err_low)], fill=EDGE, width=3)
            draw.line(
                [(center_x - error_bar_cap, err_high), (center_x + error_bar_cap, err_high)],
                fill=EDGE,
                width=3,
            )
            draw.line(
                [(center_x - error_bar_cap, err_low), (center_x + error_bar_cap, err_low)],
                fill=EDGE,
                width=3,
            )

            value_text = f"{mean_value:.1f}%"
            draw_centered(draw, center_x, y0 - 18, value_text, label_font, TEXT)
            error_text = f"+/-{std_value:.1f}"
            draw_centered(draw, center_x, y1 + 22, error_text, label_font, "#5e564d")

        label_y = plot_bottom + 64
        label_width, label_height = text_size(draw, metric_label, axis_font)
        draw.text(
            (group_center - label_width / 2, label_y),
            metric_label,
            font=axis_font,
            fill=TEXT,
        )

    y_label = "pass rate"
    temp = Image.new("RGBA", (220, 60), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp)
    temp_draw.text((0, 0), y_label, font=axis_font, fill=TEXT)
    rotated = temp.rotate(90, expand=True)
    image.paste(
        rotated,
        (70, plot_top + plot_height // 2 - rotated.height // 2),
        rotated,
    )

    notes = [
        "error bars show +/-1 repeat-to-repeat standard deviation across the 3 runs",
        "turn-taking is intentionally omitted because these runs were judged with --skip-turn-taking",
        "api/runtime salvage turns: synthetic r1 turn 63, person2 r1 turn 2, person1 r2 turn 52",
    ]
    notes_y = height - 128
    for note in notes:
        draw.text((plot_left, notes_y), note, font=small_font, fill="#5e564d")
        notes_y += 26

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def main() -> None:
    args = parse_args()
    batch_dir = args.batch_dir.expanduser().resolve()
    all_runs_path = batch_dir / "all_runs.csv"
    if not all_runs_path.exists():
        raise FileNotFoundError(f"Missing all_runs.csv in {batch_dir}")

    all_runs_df = pd.read_csv(all_runs_path)
    rows = build_metric_rows(all_runs_df)

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (batch_dir / "metrics_avg_barplot.png").resolve()
    )
    render_chart(rows, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
