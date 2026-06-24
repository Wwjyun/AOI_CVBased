from __future__ import annotations

import json
import re
import sys
import traceback
import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from tkinter import BooleanVar, StringVar, Tk, filedialog, messagebox, ttk

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    raise SystemExit("This tool needs Pillow. Install it with: pip install Pillow") from exc


CANVAS_SIZE = (1000, 760)
PLOT_BOX = (82, 92, 940, 650)
COLORS = {
    "bg": "#f7f8fa",
    "plot": "#ffffff",
    "border": "#c9ced6",
    "grid": "#e4e7ec",
    "text": "#1d2433",
    "muted": "#667085",
    "PASS": "#20a464",
    "NG": "#dc3545",
    "ERROR": "#f59e0b",
    "UNKNOWN": "#7a8699",
}


@dataclass(frozen=True)
class ScatterPoint:
    tile_id: str
    x: float
    y: float
    status: str
    defect_count: int


@dataclass(frozen=True)
class ScatterRecord:
    image_name: str
    source_path: Path
    width: float
    height: float
    points: list[ScatterPoint]


def load_scatter_records(json_path: Path) -> list[ScatterRecord]:
    with json_path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)

    if isinstance(data, dict) and isinstance(data.get("items"), list):
        records = []
        for item in data["items"]:
            detail = item.get("detail", {}) if isinstance(item, dict) else {}
            if isinstance(detail, dict) and isinstance(detail.get("tiles"), list):
                records.append(_record_from_result(detail, json_path, item.get("image_name", "")))
        return [record for record in records if record.points]

    if isinstance(data, dict) and isinstance(data.get("tiles"), list):
        record = _record_from_result(data, json_path, data.get("image_name", ""))
        return [record] if record.points else []

    return []


def load_csv_records(csv_path: Path) -> list[ScatterRecord]:
    groups: dict[str, list[ScatterPoint]] = {}
    max_bounds: dict[str, tuple[float, float]] = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_name = str(row.get("image_name", "") or csv_path.stem)
            bbox = _parse_bbox(row.get("bbox_global"))
            if bbox is None:
                continue
            x, y, width, height = bbox
            point = ScatterPoint(
                tile_id=str(row.get("tile_id", "-") or "-"),
                x=x + width / 2.0,
                y=y + height / 2.0,
                status="NG",
                defect_count=1,
            )
            groups.setdefault(image_name, []).append(point)
            current_width, current_height = max_bounds.get(image_name, (1.0, 1.0))
            max_bounds[image_name] = (
                max(current_width, x + width),
                max(current_height, y + height),
            )

    records = []
    for image_name, points in groups.items():
        width, height = max_bounds.get(image_name, (1.0, 1.0))
        records.append(
            ScatterRecord(
                image_name=image_name,
                source_path=csv_path,
                width=max(width, 1.0),
                height=max(height, 1.0),
                points=points,
            )
        )
    return records


def _record_from_result(result: dict, source_path: Path, fallback_name: str = "") -> ScatterRecord:
    points: list[ScatterPoint] = []
    max_right = 0.0
    max_bottom = 0.0
    final_result = str(result.get("final_result", "") or "")

    for tile_result in result.get("tiles", []) or []:
        tile = tile_result.get("tile", {}) or {}
        x = _float_value(tile.get("x"))
        y = _float_value(tile.get("y"))
        width = _float_value(tile.get("width"))
        height = _float_value(tile.get("height"))
        max_right = max(max_right, x + width)
        max_bottom = max(max_bottom, y + height)

        detectors = tile_result.get("detectors", []) or []
        defect_count = sum(len(detector.get("defects", []) or []) for detector in detectors)
        status = str(tile_result.get("result", "") or "PASS").upper()
        if final_result.upper() == "ERROR":
            status = "ERROR"

        points.append(
            ScatterPoint(
                tile_id=str(tile.get("tile_id", "-")),
                x=x + width / 2.0,
                y=y + height / 2.0,
                status=status if status in {"PASS", "NG", "ERROR"} else "UNKNOWN",
                defect_count=defect_count,
            )
        )

    image_name = str(result.get("image_name", "") or fallback_name or source_path.stem)
    return ScatterRecord(
        image_name=image_name,
        source_path=source_path,
        width=max(max_right, 1.0),
        height=max(max_bottom, 1.0),
        points=points,
    )


def export_record(record: ScatterRecord, output_dir: Path, suffix: str = "") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", CANVAS_SIZE, COLORS["bg"])
    draw = ImageDraw.Draw(image)
    title_font = _font(24, bold=True)
    label_font = _font(15)
    small_font = _font(12)

    title = record.image_name
    draw.text((40, 26), title, fill=COLORS["text"], font=title_font)
    draw.text(
        (40, 58),
        f"source: {record.source_path.name} | points: {len(record.points)} | NG points: {_count_status(record, 'NG')}",
        fill=COLORS["muted"],
        font=small_font,
    )

    left, top, right, bottom = PLOT_BOX
    draw.rectangle(PLOT_BOX, fill=COLORS["plot"], outline=COLORS["border"], width=2)
    for index in range(1, 5):
        x = left + (right - left) * index / 5
        y = top + (bottom - top) * index / 5
        draw.line((x, top, x, bottom), fill=COLORS["grid"], width=1)
        draw.line((left, y, right, y), fill=COLORS["grid"], width=1)

    for point in record.points:
        px = left + point.x / record.width * (right - left)
        py = top + point.y / record.height * (bottom - top)
        radius = 6 + min(12, max(0, point.defect_count) * 2)
        color = COLORS.get(point.status, COLORS["UNKNOWN"])
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color, outline="#ffffff", width=2)

    draw.text(((left + right) / 2 - 20, bottom + 30), "tile x", fill=COLORS["muted"], font=label_font)
    draw.text((28, (top + bottom) / 2), "tile y", fill=COLORS["muted"], font=label_font)
    _draw_legend(draw, right - 270, top + 16, small_font)

    safe_name = _safe_filename(Path(record.image_name).stem or record.source_path.stem)
    if suffix:
        safe_name = f"{safe_name}_{suffix}"
    output_path = _unique_path(output_dir / f"{safe_name}_scatter.png")
    image.save(output_path)
    return output_path


def export_folder(input_dir: Path, output_dir: Path, recursive: bool) -> tuple[int, list[str]]:
    json_pattern = "**/*.json" if recursive else "*.json"
    csv_pattern = "**/*.csv" if recursive else "*.csv"
    report_paths = sorted([*input_dir.glob(json_pattern), *input_dir.glob(csv_pattern)])
    exported = 0
    errors: list[str] = []

    for report_path in report_paths:
        try:
            if report_path.suffix.lower() == ".json":
                records = load_scatter_records(report_path)
            elif report_path.suffix.lower() == ".csv":
                records = load_csv_records(report_path)
            else:
                records = []
            for index, record in enumerate(records):
                suffix = report_path.stem if len(records) == 1 else f"{report_path.stem}_{index + 1}"
                export_record(record, output_dir, suffix=suffix)
                exported += 1
        except Exception as exc:
            errors.append(f"{report_path}: {exc}")

    return exported, errors


class ScatterExportApp:
    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("AOI Scatter Plot Exporter")
        self.root.geometry("680x360")
        self.input_dir = StringVar()
        self.output_dir = StringVar()
        self.recursive = BooleanVar(value=True)
        self.status = StringVar(value="Select a folder that contains AOI JSON or CSV reports.")
        self._build()

    def run(self) -> None:
        self.root.mainloop()

    def _build(self) -> None:
        frame = ttk.Frame(self.root, padding=18)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Report folder").grid(row=0, column=0, sticky="w", pady=8)
        ttk.Entry(frame, textvariable=self.input_dir).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(frame, text="Browse", command=self._choose_input).grid(row=0, column=2, sticky="ew")

        ttk.Label(frame, text="Output folder").grid(row=1, column=0, sticky="w", pady=8)
        ttk.Entry(frame, textvariable=self.output_dir).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(frame, text="Browse", command=self._choose_output).grid(row=1, column=2, sticky="ew")

        ttk.Checkbutton(frame, text="Include subfolders", variable=self.recursive).grid(
            row=2, column=1, sticky="w", pady=8
        )

        ttk.Button(frame, text="Export Scatter Plots", command=self._export).grid(
            row=3, column=1, sticky="ew", padx=8, pady=18
        )

        status_box = ttk.Label(frame, textvariable=self.status, wraplength=620, foreground="#344054")
        status_box.grid(row=4, column=0, columnspan=3, sticky="ew", pady=8)

    def _choose_input(self) -> None:
        folder = filedialog.askdirectory(title="Select AOI report folder")
        if not folder:
            return
        self.input_dir.set(folder)
        if not self.output_dir.get():
            self.output_dir.set(str(Path(folder) / "scatter_plots"))

    def _choose_output(self) -> None:
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.output_dir.set(folder)

    def _export(self) -> None:
        input_dir = Path(self.input_dir.get())
        output_dir = Path(self.output_dir.get() or input_dir / "scatter_plots")
        if not input_dir.is_dir():
            messagebox.showerror("Folder not found", "Please select a valid JSON report folder.")
            return

        self.status.set("Exporting...")
        self.root.update_idletasks()
        try:
            exported, errors = export_folder(input_dir, output_dir, self.recursive.get())
        except Exception:
            messagebox.showerror("Export failed", traceback.format_exc())
            self.status.set("Export failed.")
            return

        if exported == 0:
            self.status.set("No scatter plots exported. No AOI JSON tile data or CSV defect data was found.")
            messagebox.showwarning("No data", self.status.get())
            return

        message = f"Exported {exported} scatter plot(s) to:\n{output_dir}"
        if errors:
            message += f"\n\nSkipped {len(errors)} file(s). First error:\n{errors[0]}"
        self.status.set(message)
        messagebox.showinfo("Export complete", message)


def _draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int, font: ImageFont.ImageFont) -> None:
    for index, status in enumerate(("PASS", "NG", "ERROR")):
        item_x = x + index * 86
        color = COLORS[status]
        draw.ellipse((item_x, y, item_x + 12, y + 12), fill=color, outline="#ffffff", width=1)
        draw.text((item_x + 18, y - 2), status, fill=COLORS["text"], font=font)


def _count_status(record: ScatterRecord, status: str) -> int:
    return sum(1 for point in record.points if point.status == status)


def _parse_bbox(value: object) -> tuple[float, float, float, float] | None:
    if value in (None, ""):
        return None
    if isinstance(value, (list, tuple)):
        raw = value
    else:
        try:
            raw = ast.literal_eval(str(value))
        except (SyntaxError, ValueError):
            return None
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    return (
        _float_value(raw[0]),
        _float_value(raw[1]),
        _float_value(raw[2]),
        _float_value(raw[3]),
    )


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value).strip(" ._")
    return cleaned[:120] or "scatter"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 10000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Too many duplicate output files for {path}")


def _float_value(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "arialbd.ttf" if bold else "arial.ttf",
        "msjhbd.ttc" if bold else "msjh.ttc",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    if argv:
        input_dir = Path(argv[0])
        output_dir = Path(argv[1]) if len(argv) > 1 else input_dir / "scatter_plots"
        exported, errors = export_folder(input_dir, output_dir, recursive=True)
        print(f"Exported {exported} scatter plot(s) to {output_dir}")
        if errors:
            print(f"Skipped {len(errors)} file(s). First error: {errors[0]}")
        return 0 if exported else 1

    ScatterExportApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
