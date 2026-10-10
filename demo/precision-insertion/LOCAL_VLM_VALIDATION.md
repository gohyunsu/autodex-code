# Local VLM mode: initial validation (2026-10-10)

The independent demo can select `mode="local"` with
`load_vlm_backend(...)`, using ZeroDex's existing `main.vlm_base.BaseVLM`.
`probe_vlm.py --backend local` runs lift or insertion-visual classification
on saved image pairs only. No stock AutoDex execution files were changed.

## Environment and smoke test

- GPU: RTX 3090, 24 GB; CUDA visible to PyTorch 2.4.1+cu121 outside the
  restricted shell sandbox.
- Isolated environment: `~/.venvs/precision-vlm`, created with
  `--system-site-packages` to reuse that PyTorch. Installed the versions in
  `requirements-local-vlm.txt`, matching the checked-out ZeroDex inference
  dependencies. The production `autodex_bodex` environment was not modified.
- Model: `Qwen/Qwen3-VL-2B-Instruct`; downloaded to the standard Hugging Face
  cache (~4.0 GB) and loaded on GPU. No Gemini credential is needed.
- Two actual model calls succeeded using *presentation renders*, not
  synchronized AutoDex camera trial frames. Lift returned a parseable `held`;
  insertion visual returned a parseable `normal_appearance`. Those labels
  **must not be treated as correct physical outcomes**: the images were not
  a matched real before/after experiment, and the insertion response itself
  said no insertion was visible. The test establishes model loading, image
  transport and schema parsing only.
- The first model call wrapped JSON in a Markdown fence and used image
  numbers rather than camera IDs. Parsing now accepts only a *complete* JSON
  fence while retaining the exact raw response; the prompt explicitly names
  valid camera IDs. Incorrect/absent evidence IDs still fail closed.
- All demo tests: `278 passed` on the AutoDex Python environment.

## Remaining validation before live use

Collect time-synchronized, phase-paired **raw AutoDex camera images** with
independent lift and insertion labels. Evaluate false-positive `held`, jam,
and insertion-appearance rates against those labels, including occlusion and
slip cases. A VLM label never substitutes for the separate depth/force and
safety gates. The point/axis grounding route additionally needs native-size
undistorted images, calibration, held-out pixel-error estimates, and measured
multi-view metric accuracy. The current demo has no commissioned live camera
capture/robot executor, so this local mode is not a complete autonomous
insertion loop.
