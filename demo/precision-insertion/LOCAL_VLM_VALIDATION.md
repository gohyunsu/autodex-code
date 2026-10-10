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
The metric tip/axis parsers now also accept a *complete* JSON Markdown fence,
matching the semantic parser; truncated JSON or extra prose still yields
null landmarks. This is a format compatibility fix, not a pixel-accuracy
validation.

## Compact-response replay (2026-10-11)

A repeat of the saved-render lift probe with the cached Qwen3-VL-2B model
initially produced an unclosed JSON string: its free-form evidence repeated
until the 512-token limit. The shared closed-set prompt now asks for one
compact JSON object and at most 12 evidence words. With the same two
presentation images, the model returned parseable JSON in 0.84 s of inference
on the RTX 3090. The output class was `held`, but these images are not a
synchronized physical lift pair; **this is format validation only, not an
accuracy result or a trusted grasp label**. Malformed output still maps to
`unobservable` rather than being repaired or used to authorize motion.

## Saved-render prompt regression (2026-10-11)

An insertion-appearance smoke test initially returned the *literal* pipe-
separated choice list from the prompt as its `visual_class`. The parser
correctly rejected this and returned `unobservable`. The closed-set prompts
now list alternatives outside the JSON field instead of presenting a
`label_a|label_b` value template. With the same cached Qwen3-VL-2B model and
the same two presentation renders, the response was valid JSON and parsed as
`normal_appearance` in 1.01 s of model inference. This demonstrates improved
format adherence only: its evidence said the key was *above* the socket, so
the label is not a trustworthy insertion result. The render pair is neither
time-synchronized AutoDex camera footage nor an independently labeled trial.
Malformed answers continue to map to `unobservable`.

The saved-image lift probe also loaded the model offline and returned a
parseable `held` response in 0.83 s. Its images are different presentation
scenes, not a before/after grasp, so this is likewise only a transport and
schema test. All physical task outcomes still require independent evidence.

## Remaining validation before live use

On 2026-10-11 the cached Qwen3-VL-2B model was loaded again with
`HF_HUB_OFFLINE=1` on the RTX 3090. The current compact prompt returned a
parseable `held` JSON response in about 1.03 s of model inference. The two
inputs were distinct **presentation renders**, not a synchronized grasp
trial; the label is therefore a transport/format smoke-test result only.
The source hashes and raw response are saved in
`/tmp/precision-local-vlm-smoke-current-20261011.json` on this workstation.

Collect time-synchronized, phase-paired **raw AutoDex camera images** with
independent lift and insertion labels. Evaluate false-positive `held`, jam,
and insertion-appearance rates against those labels, including occlusion and
slip cases. A VLM label never substitutes for the separate depth/force and
safety gates. The point/axis grounding route additionally needs native-size
undistorted images, calibration, held-out pixel-error estimates, and measured
multi-view metric accuracy. The current demo has no commissioned live camera
capture/robot executor, so this local mode is not a complete autonomous
insertion loop.

## Current local-mode check (2026-10-11)

Retested the existing `--backend local` path with offline-cached
`Qwen/Qwen3-VL-2B-Instruct` on the RTX 3090. The lift and insertion-visual
saved-image probes both loaded and produced parseable JSON (about 1.03 s and
0.96 s of model inference, respectively). These inputs were presentation
renders, not synchronized robot-camera trials. In the insertion probe, the
model selected `normal_appearance` while its own evidence said the key was
*above* the socket. This is a concrete example of why successful parsing is
not correctness and why the VLM cannot alone certify 20 mm insertion. Raw
reports on this workstation are
`/tmp/precision-local-vlm-probe-user-20261011.json` and
`/tmp/precision-local-vlm-insertion-probe-user-20261011.json`.
