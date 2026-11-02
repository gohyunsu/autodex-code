# Precision insertion thesis draft

This directory contains a formal thesis draft based on the physical success
episode recorded on 2026-10-30. Incomplete evaluation is identified as
preliminary experiments and future work inside the thesis structure.

## Outputs

- `precision_insertion_thesis_ko.md`: editable source
- `precision_insertion_thesis_ko.pdf`: A4 review PDF
- `precision_insertion_thesis_ko.docx`: editable Word export
- `precision_insertion_thesis_ko.html`: self-contained PDF-rendering input
- `assets/`: report figures and the linked closed-loop GIF
- `assets/source/`: presentation-style composites used only to derive the
  camera-specific evidence crops

Before submission, replace `[성명 확인 필요]` on the cover with the advisor's
full Korean name and confirm the submission and degree dates.

## Physical evidence

- Attempt:
  `/mnt/paradex2/hyunsu/precision_insertion/runs/zerodex_capture4/precision-insertion/inspire/precision_key_cylinder_r15_h80/attempt_3c8ff1b830d347a2b8c45aa7ae1a5363`
- Session:
  `/mnt/paradex2/hyunsu/precision_insertion/runs/zerodex_capture4/precision-insertion/inspire/precision_key_cylinder_r15_h80/_pipeline_runs/20261030_222800_220645_219060`
- Canonical visualization:
  `/mnt/paradex2/hyunsu/precision_insertion/runs/zerodex_capture4/2026/10/30/20261030_222800_220645_219060/visualizations/closed_loop_multiview_without_25305462`

Camera `25305462` is omitted only from figures for this episode because it was
physically disturbed after calibration. The report does not define a permanent
camera blacklist.

## Rebuild

```bash
cd /home/hyunsu/autodex-code/demo/precision-insertion/reports/thesis_draft_20261102
python3 make_figures.py

pandoc precision_insertion_thesis_ko.md \
  --from markdown+tex_math_dollars+tex_math_single_backslash \
  --toc --number-sections --standalone --mathml \
  --css report.css --self-contained --resource-path=. \
  -o precision_insertion_thesis_ko.html

pandoc precision_insertion_thesis_ko.md \
  --from markdown+tex_math_dollars+tex_math_single_backslash \
  --toc --number-sections --resource-path=. \
  -o precision_insertion_thesis_ko.docx

google-chrome --headless --disable-gpu --no-sandbox \
  --disable-dev-shm-usage --no-pdf-header-footer \
  --print-to-pdf="$PWD/precision_insertion_thesis_ko.pdf" \
  "file://$PWD/precision_insertion_thesis_ko.html"
```

