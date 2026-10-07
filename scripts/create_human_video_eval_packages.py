"""Create human evaluation distribution packages for GSM8K video quality scoring.

The script reads the 200-row video comparison table, groups samples into
10-question batches, copies videos/problem images into evaluator-friendly
folders, creates a prefilled Excel scoring sheet, and writes one zip per batch.

It does not modify the original experiment results.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import zipfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "data/experiments/gsm8k/video_runs/full_vs_simple_pilot_200"
DEFAULT_COMPARISON_CSV = EXPERIMENT_DIR / "comparison.csv"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "human_eval_packages"
DEFAULT_BATCH_SIZE = 10
DEFAULT_SEED = 20260630

RUBRIC_HTML = REPO_ROOT / "视频质量评分准则.html"
RUBRIC_PDF_CANDIDATES = [
    REPO_ROOT / "视频质量评分准则.pdf",
    REPO_ROOT / "视频质量人工评分准则.pdf",
]

SCORING_COLUMNS = [
    "题号",
    "题目图片",
    "视频A",
    "视频B",
    "A_公式来源解释清晰度",
    "A_讲解步骤完整性",
    "A_画面可读性",
    "A_音画同步",
    "A_旁白内容清晰自然度",
    "A_自学适用性",
    "A_可选备注",
    "B_公式来源解释清晰度",
    "B_讲解步骤完整性",
    "B_画面可读性",
    "B_音画同步",
    "B_旁白内容清晰自然度",
    "B_自学适用性",
    "B_可选备注",
    "整体备注",
    "batch_id",
    "selection_rank",
    "sample_id",
]

SCORE_COLUMN_INDEXES = [5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create 20 human video evaluation zip packages.")
    parser.add_argument("--comparison-csv", type=Path, default=DEFAULT_COMPARISON_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true", help="Delete existing output package folders before creating.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned batches without writing.")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["selection_rank"]))
    return rows


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def validate_inputs(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 200:
        raise SystemExit(f"Expected 200 comparison rows, found {len(rows)}.")
    for row in rows:
        if not bool_value(row.get("full_system_video_generated")):
            raise SystemExit(f"Missing Full System video for {row['sample_id']}.")
        if not bool_value(row.get("plain_simple_video_generated")):
            raise SystemExit(f"Missing Plain Simple video for {row['sample_id']}.")
        for key in ["image_path", "full_system_video_path", "plain_simple_video_path"]:
            path = resolve_repo_path(row[key])
            if not path.exists():
                raise SystemExit(f"Missing file for {row['sample_id']} {key}: {path}")
    if not RUBRIC_HTML.exists():
        raise SystemExit(f"Missing rubric HTML: {RUBRIC_HTML}")
    if not rubric_pdf_path().exists():
        raise SystemExit("Missing rubric PDF. Expected 视频质量评分准则.pdf or 视频质量人工评分准则.pdf in repo root.")


def rubric_pdf_path() -> Path:
    for path in RUBRIC_PDF_CANDIDATES:
        if path.exists():
            return path
    return RUBRIC_PDF_CANDIDATES[0]


def balanced_ab_assignments(rows: list[dict[str, Any]], seed: int) -> dict[str, str]:
    """Return sample_id -> system assigned to video A.

    Balanced 100/100 across 200 rows while staying deterministic.
    """

    labels = ["full_system"] * (len(rows) // 2) + ["plain_simple"] * (len(rows) - len(rows) // 2)
    rng = random.Random(seed)
    rng.shuffle(labels)
    return {row["sample_id"]: label for row, label in zip(rows, labels, strict=True)}


def batch_name(batch_index: int, batch_rows: list[dict[str, Any]]) -> str:
    first_rank = int(batch_rows[0]["selection_rank"])
    last_rank = int(batch_rows[-1]["selection_rank"])
    return f"batch_{batch_index:03d}_rank_{first_rank:03d}_{last_rank:03d}"


def score_sheet_name(batch_id: str) -> str:
    return f"视频质量评分表_{batch_id}.xlsx"


def readme_text(score_sheet_file: str) -> str:
    return "\n".join(
        [
            "打分前请阅读",
            "",
            "1. 请先解压整个压缩包，不要直接在压缩包预览界面中观看视频或填写表格。",
            "2. 本文件夹包含 10 道题。每道题对应两个讲解视频：A 和 B。请完整观看同一道题的两个视频后，再根据评分标准分别填写对应指标分数。",
            "3. 请打开“视频质量评分准则.html”或“视频质量评分准则.pdf”查看评分规则。两个文件内容完全一致，选择其中一个查看即可。",
            "4. 对每道题，请先查看 problems 文件夹中的题目图片，再观看 videos 文件夹中对应的 Qxx_A.mp4 和 Qxx_B.mp4。打分表 Excel 中已经预填了题目图片和视频链接，可直接点击打开。",
            f"5. 请在“{score_sheet_file}”中填写两个视频的各项评分。评分范围为 1-5 分。",
            "6. 备注不是必填项，可以不写；如果发现明显问题，也可以简单记录。",
            f"7. 完成后请保存 Excel 表格。返回填写后的“{score_sheet_file}”即可，其他视频、图片和评分准则文件无需返回。",
            "",
            "万分感谢您的帮助与支持！",
            "",
        ]
    )


def create_batch(
    *,
    batch_dir: Path,
    batch_rows: list[dict[str, Any]],
    assignments: dict[str, str],
    admin_mapping_rows: list[dict[str, Any]],
    package_summary_rows: list[dict[str, Any]],
) -> None:
    videos_dir = batch_dir / "videos"
    problems_dir = batch_dir / "problems"
    videos_dir.mkdir(parents=True, exist_ok=True)
    problems_dir.mkdir(parents=True, exist_ok=True)
    score_sheet_file = score_sheet_name(batch_dir.name)

    shutil.copy2(RUBRIC_HTML, batch_dir / "视频质量评分准则.html")
    shutil.copy2(rubric_pdf_path(), batch_dir / "视频质量评分准则.pdf")
    (batch_dir / "打分前请阅读.txt").write_text(readme_text(score_sheet_file), encoding="utf-8")

    excel_rows: list[list[str]] = []
    for index, row in enumerate(batch_rows, start=1):
        qid = f"Q{index:02d}"
        sample_id = row["sample_id"]
        a_system = assignments[sample_id]
        b_system = "plain_simple" if a_system == "full_system" else "full_system"

        image_target = f"problems/{qid}.png"
        video_a_target = f"videos/{qid}_A.mp4"
        video_b_target = f"videos/{qid}_B.mp4"

        shutil.copy2(resolve_repo_path(row["image_path"]), batch_dir / image_target)
        shutil.copy2(resolve_repo_path(video_path_for(row, a_system)), batch_dir / video_a_target)
        shutil.copy2(resolve_repo_path(video_path_for(row, b_system)), batch_dir / video_b_target)

        excel_rows.append([qid, image_target, video_a_target, video_b_target, *[""] * 15, batch_dir.name, row["selection_rank"], sample_id])
        admin_mapping_rows.append(
            {
                "batch_id": batch_dir.name,
                "question_no": qid,
                "selection_rank": row["selection_rank"],
                "sample_id": sample_id,
                "video_a_system": a_system,
                "video_b_system": b_system,
                "video_a_file": video_a_target,
                "video_b_file": video_b_target,
                "problem_image": image_target,
                "full_system_video_path": row["full_system_video_path"],
                "plain_simple_video_path": row["plain_simple_video_path"],
                "source_image_path": row["image_path"],
            }
        )

    package_summary_rows.append(
        {
            "batch_id": batch_dir.name,
            "rank_start": batch_rows[0]["selection_rank"],
            "rank_end": batch_rows[-1]["selection_rank"],
            "sample_count": len(batch_rows),
            "zip_file": f"{batch_dir.name}.zip",
            "score_sheet_file": score_sheet_file,
        }
    )
    write_scoring_xlsx(batch_dir / score_sheet_file, excel_rows)


def video_path_for(row: dict[str, Any], system: str) -> str:
    if system == "full_system":
        return row["full_system_video_path"]
    if system == "plain_simple":
        return row["plain_simple_video_path"]
    raise ValueError(f"Unknown system: {system}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def zip_batch(batch_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(batch_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(batch_dir.parent))


def xlsx_col(index: int) -> str:
    result = ""
    while index:
        index, rem = divmod(index - 1, 26)
        result = chr(65 + rem) + result
    return result


def cell(ref: str, value: Any = "", *, style: int | None = None, hyperlink: str | None = None) -> str:
    style_attr = f' s="{style}"' if style is not None else ""
    if hyperlink:
        style_attr = ' s="2"'
    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"{style_attr}><is><t>{text}</t></is></c>'


def write_scoring_xlsx(path: Path, rows: list[list[str]]) -> None:
    sheet_rows: list[str] = []
    header_cells = [cell(f"{xlsx_col(i)}1", value, style=1) for i, value in enumerate(SCORING_COLUMNS, start=1)]
    sheet_rows.append(f'<row r="1">{"".join(header_cells)}</row>')

    hyperlinks: list[tuple[str, str, str]] = []
    rels: list[str] = []
    rel_id = 1
    for r_idx, row_values in enumerate(rows, start=2):
        row_cells: list[str] = []
        for c_idx, value in enumerate(row_values, start=1):
            ref = f"{xlsx_col(c_idx)}{r_idx}"
            if c_idx in {2, 3, 4}:
                rid = f"rId{rel_id}"
                rel_id += 1
                row_cells.append(cell(ref, value, hyperlink=value))
                hyperlinks.append((ref, value, rid))
                rels.append(
                    f'<Relationship Id="{rid}" '
                    'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" '
                    f'Target="{escape(value)}" TargetMode="External"/>'
                )
            else:
                row_cells.append(cell(ref, value))
        sheet_rows.append(f'<row r="{r_idx}">{"".join(row_cells)}</row>')

    data_validations = []
    for col_idx in SCORE_COLUMN_INDEXES:
        col = xlsx_col(col_idx)
        data_validations.append(
            f'<dataValidation type="list" allowBlank="1" showErrorMessage="1" '
            f'sqref="{col}2:{col}{len(rows) + 1}"><formula1>"1,2,3,4,5"</formula1></dataValidation>'
        )

    hyperlink_xml = "".join(
        f'<hyperlink ref="{ref}" r:id="{rid}" display="{escape(display)}"/>' for ref, display, rid in hyperlinks
    )
    hidden_meta_columns = "".join(
        f'<col min="{index}" max="{index}" width="0" hidden="1" customWidth="1"/>'
        for index in range(20, 23)
    )
    worksheet = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>
  <cols>
    <col min="1" max="1" width="10" customWidth="1"/>
    <col min="2" max="4" width="24" customWidth="1"/>
    <col min="5" max="10" width="18" customWidth="1"/>
    <col min="11" max="11" width="28" customWidth="1"/>
    <col min="12" max="17" width="18" customWidth="1"/>
    <col min="18" max="19" width="28" customWidth="1"/>
    {hidden_meta_columns}
  </cols>
  <sheetData>{''.join(sheet_rows)}</sheetData>
  <dataValidations count="{len(data_validations)}">{''.join(data_validations)}</dataValidations>
  <hyperlinks>{hyperlink_xml}</hyperlinks>
</worksheet>
'''

    sheet_rels = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{''.join(rels)}</Relationships>
'''
    workbook = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="评分表" sheetId="1" r:id="rId1"/></sheets>
</workbook>
'''
    workbook_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
'''
    root_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>
'''
    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>
'''
    styles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="3">
    <font><sz val="11"/><name val="Microsoft YaHei"/></font>
    <font><b/><sz val="11"/><name val="Microsoft YaHei"/></font>
    <font><u/><color rgb="FF0563C1"/><sz val="11"/><name val="Microsoft YaHei"/></font>
  </fonts>
  <fills count="3">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFEAF2F8"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border><left style="thin"/><right style="thin"/><top style="thin"/><bottom style="thin"/><diagonal/></border>
  </borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="3">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1"/>
    <xf numFmtId="0" fontId="2" fillId="0" borderId="1" xfId="0" applyFont="1" applyBorder="1"/>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>
'''

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as xlsx:
        xlsx.writestr("[Content_Types].xml", content_types)
        xlsx.writestr("_rels/.rels", root_rels)
        xlsx.writestr("xl/workbook.xml", workbook)
        xlsx.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        xlsx.writestr("xl/worksheets/sheet1.xml", worksheet)
        xlsx.writestr("xl/worksheets/_rels/sheet1.xml.rels", sheet_rels)
        xlsx.writestr("xl/styles.xml", styles)


def main() -> int:
    args = parse_args()
    rows = read_rows(args.comparison_csv)
    validate_inputs(rows)

    if args.dry_run:
        for index in range(0, len(rows), args.batch_size):
            batch_rows = rows[index : index + args.batch_size]
            print(batch_name(index // args.batch_size + 1, batch_rows))
        return 0

    output_dir = args.output_dir
    unzipped_dir = output_dir / "01_unzipped_batches"
    zip_dir = output_dir / "02_zip_for_distribution"
    returned_dir = output_dir / "03_returned_excels"
    admin_dir = output_dir / "admin"

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    elif output_dir.exists():
        raise SystemExit(f"Output directory already exists. Use --overwrite to recreate: {output_dir}")

    unzipped_dir.mkdir(parents=True, exist_ok=True)
    zip_dir.mkdir(parents=True, exist_ok=True)
    returned_dir.mkdir(parents=True, exist_ok=True)
    admin_dir.mkdir(parents=True, exist_ok=True)

    assignments = balanced_ab_assignments(rows, args.seed)
    admin_mapping_rows: list[dict[str, Any]] = []
    package_summary_rows: list[dict[str, Any]] = []

    for batch_index, start in enumerate(range(0, len(rows), args.batch_size), start=1):
        batch_rows = rows[start : start + args.batch_size]
        name = batch_name(batch_index, batch_rows)
        batch_dir = unzipped_dir / name
        print(f"Creating {name}")
        create_batch(
            batch_dir=batch_dir,
            batch_rows=batch_rows,
            assignments=assignments,
            admin_mapping_rows=admin_mapping_rows,
            package_summary_rows=package_summary_rows,
        )
        zip_batch(batch_dir, zip_dir / f"{name}.zip")

    write_csv(admin_dir / "admin_mapping.csv", admin_mapping_rows)
    write_csv(admin_dir / "package_summary.csv", package_summary_rows)
    (admin_dir / "generation_config.json").write_text(
        json.dumps(
            {
                "comparison_csv": str(args.comparison_csv),
                "batch_size": args.batch_size,
                "seed": args.seed,
                "sample_count": len(rows),
                "batch_count": len(package_summary_rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Done: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
