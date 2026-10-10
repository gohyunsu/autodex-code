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

## Follow-up GPU check (2026-10-11)

The downloaded Qwen3-VL-2B model still loads on the RTX 3090. In a saved-render
lift probe, a 256-token response repeated its explanation, ended before closing
the JSON, and was correctly mapped to `unobservable` with a parse error. A
separate saved-render probe with 512 output tokens returned parseable JSON. The
saved-image probe and benchmark now default to 512 tokens, matching the backend
default. Neither render pair is synchronized robot evidence, so this check
establishes execution and parsing only, not label accuracy. Even at 512 tokens,
malformed responses may occur and must remain fail-closed.

An additional offline-cache RTX 3090 replay on 2026-10-11 loaded the same
model and completed inference in 5.99 s. Its two presentation panels were
*not* synchronized before/after frames. The model repeated a contradictory
explanation until the 512-token limit and left its JSON unfinished; the
checkpoint therefore recorded `parse_error` and `class=unobservable`. This
confirms that local execution is available but a syntactically valid,
evidence-grounded answer is not guaranteed. Do not interpret this probe as a
grasp-success or model-accuracy measurement.

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
