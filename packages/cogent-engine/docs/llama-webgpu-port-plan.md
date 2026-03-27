# WebGPU File Review: `constant/dev` vs `origin/master`

This note is based on direct file-to-file comparison only. Commit history was intentionally ignored for this pass. The truth source here is the branch snapshots under `ggml/src/ggml-webgpu`.

## Scope

- Compared branches:
  - `constant/dev`
  - `origin/master`
- Compared directory:
  - `ggml/src/ggml-webgpu`
- Important rule:
  - `ggml-webgpu-shader-lib.hpp` is generated, but it still reflects real branch behavior. It should not be hand-edited, yet it must be reviewed as evidence of actual shader/pipeline differences.

## Tree Status

### Unchanged

- `pre_wgsl.hpp`

### Modified

- `CMakeLists.txt`
- `ggml-webgpu-shader-lib.hpp`
- `ggml-webgpu.cpp`
- `wgsl-shaders/argmax.wgsl`
- `wgsl-shaders/argsort.wgsl`
- `wgsl-shaders/argsort_merge.wgsl`
- `wgsl-shaders/binary.wgsl`
- `wgsl-shaders/common_decls.tmpl`
- `wgsl-shaders/concat.wgsl`
- `wgsl-shaders/cpy.tmpl.wgsl`
- `wgsl-shaders/cumsum.wgsl`
- `wgsl-shaders/embed_wgsl.py`
- `wgsl-shaders/flash_attn.wgsl`
- `wgsl-shaders/get_rows.wgsl`
- `wgsl-shaders/glu.tmpl.wgsl`
- `wgsl-shaders/memset.wgsl`
- `wgsl-shaders/mul_mat.wgsl`
- `wgsl-shaders/mul_mat_decls.tmpl`
- `wgsl-shaders/mul_mat_reg_tile.wgsl`
- `wgsl-shaders/mul_mat_subgroup_matrix.wgsl`
- `wgsl-shaders/mul_mat_vec.wgsl`
- `wgsl-shaders/pad.wgsl`
- `wgsl-shaders/repeat.wgsl`
- `wgsl-shaders/rope.tmpl.wgsl`
- `wgsl-shaders/scale.wgsl`
- `wgsl-shaders/set.wgsl`
- `wgsl-shaders/set_rows.wgsl`
- `wgsl-shaders/soft_max.tmpl.wgsl`
- `wgsl-shaders/solve_tri.wgsl`
- `wgsl-shaders/ssm_conv.wgsl`
- `wgsl-shaders/sum_rows.wgsl`
- `wgsl-shaders/unary.wgsl`

### Added On `origin/master`

- `wgsl-shaders/bin_op.tmpl.wgsl`
- `wgsl-shaders/binary_head.tmpl`
- `wgsl-shaders/conv2d.wgsl`
- `wgsl-shaders/diag.wgsl`
- `wgsl-shaders/get_rows.tmpl.wgsl`
- `wgsl-shaders/l2_norm.wgsl`
- `wgsl-shaders/mul_mat.tmpl.wgsl`
- `wgsl-shaders/mul_mat_decls_compat.tmpl`
- `wgsl-shaders/mul_mat_reg_tile.tmpl.wgsl`
- `wgsl-shaders/mul_mat_subgroup_matrix.tmpl.wgsl`
- `wgsl-shaders/mul_mat_vec.tmpl.wgsl`
- `wgsl-shaders/ssm_scan_128.wgsl`
- `wgsl-shaders/ssm_scan_256.wgsl`
- `wgsl-shaders/ssm_scan_64.wgsl`
- `wgsl-shaders/template_decls_compat.tmpl`
- `wgsl-shaders/tri.wgsl`

### Removed On `origin/master`

- `wgsl-shaders/gated_delta_net.wgsl`

### Renamed On `origin/master`

- `wgsl-shaders/row_norm.wgsl` -> `wgsl-shaders/rms_norm.wgsl`

## Why The Generated Header Matters

`ggml-webgpu-shader-lib.hpp` is not noise here. It exposes real behavior differences between the branches.

### Pipeline Getter Differences

Getters present on `origin/master` but not on `constant/dev`:

- `get_cpy_pipeline`
- `get_glu_pipeline`
- `get_rope_pipeline`
- `get_soft_max_pipeline`
- `get_rms_norm_pipeline`
- `get_l2_norm_pipeline`
- `get_tri_pipeline`
- `get_diag_pipeline`
- `get_conv_2d_pipeline`

Getters present on `constant/dev` but not on `origin/master`:

- `get_gated_delta_net_pipeline`
- `get_row_norm_pipeline`

### Embedded Shader Symbol Differences

`origin/master` adds embedded shader families for:

- `cpy_*`
- `glu_*`
- `rope_*`
- `soft_max_*`
- `rms_norm`
- `l2_norm`
- `tri`
- `diag`
- `conv2d`
- many `mul_mat_*` and `mul_mat_vec_*` f32 / mixed / `_al` variants
- many `add/sub/mul/div` `_al` variants for binary ops

`constant/dev` keeps embedded symbols that `origin/master` no longer has:

- `wgsl_gated_delta_net`
- `wgsl_row_norm`

That means the generated header is confirming real implementation divergence, not merely a rebuild artifact.

## `ggml-webgpu.cpp`: Direct Implementation Differences

### 1. Runtime Capability Model

`origin/master` adds explicit optional-f16 handling:

- `has_f16_support` in capability state
- shader paths that check `has_f16_support`
- device setup that no longer treats `ShaderF16` as unconditional in the same way as `constant/dev`

This change is wired through the encode paths and shader selection logic. It is not just build glue.

### 2. Submission / Wait Path

`origin/master` replaces the older submission tracking with new future-based helpers:

- new structs:
  - `webgpu_submission_futures`
  - `webgpu_pool_bufs`
  - `webgpu_transient_buf_pool`
- new helper:
  - `ggml_backend_webgpu_wait_future`

`constant/dev` still has the older wait/submission helpers:

- `webgpu_submission`
- `ggml_backend_webgpu_handle_wait_status`
- `ggml_backend_webgpu_erase_completed_futures`
- `ggml_backend_webgpu_wait_profile_futures`

This is a real runtime behavior difference in queue completion and resource lifetime handling.

### 3. Alias Handling And Buffer Safety

`origin/master` adds explicit alias helpers that do not exist on `constant/dev`:

- `ggml_webgpu_handle_alias`
- `ggml_webgpu_handle_binary_src_alias`
- `ggml_webgpu_share_buffer`

These helpers are then threaded through many encode paths, including:

- binary ops
- copy/contiguous paths
- `GET_ROWS`
- `MUL_MAT`
- `FLASH_ATTN_EXT`
- `CONCAT`
- `CONV_2D`
- `SSM_CONV`
- `SSM_SCAN`

### 4. Packed Quant Tensor Path

`origin/master` adds a packed quant cache/update path that `constant/dev` does not have:

- `webgpu_packed_tensor`
- `ggml_webgpu_block_stride_bytes`
- `ggml_webgpu_get_packed_tensor`
- `ggml_webgpu_update_packed_tensor`
- `ggml_webgpu_update_packed_view_source`

That path is consumed by:

- `ggml_webgpu_get_rows`
- `ggml_webgpu_mul_mat`
- tensor upload/update code

### 5. Operator Helpers Added/Removed

Added on `origin/master`:

- `ggml_webgpu_binary_op_shader_lib`
- `ggml_webgpu_conv_2d`
- `ggml_webgpu_diag`
- `ggml_webgpu_l2_norm`
- `ggml_webgpu_rms_norm`
- `ggml_webgpu_ssm_scan`
- `ggml_webgpu_tri`
- device lost / uncaptured error callbacks

Only on `constant/dev`:

- `ggml_webgpu_gated_delta_net`
- `ggml_webgpu_row_norm`
- `ggml_webgpu_init_cpy_pipeline`
- `ggml_webgpu_init_glu_pipeline`
- `ggml_webgpu_init_rope_pipeline`
- `ggml_webgpu_init_soft_max_pipeline`

### 6. Operator Switch Differences

Operator cases present on `origin/master` but not on `constant/dev`:

- `GGML_OP_CONV_2D`
- `GGML_OP_SSM_SCAN`

Operator case present on `constant/dev` but not on `origin/master`:

- `GGML_OP_GATED_DELTA_NET`

This is the cleanest operator-level summary of the two branches.

## Build And Shader Generation Files

### `CMakeLists.txt`

`origin/master` changes build/generation behavior in these concrete ways:

- `file(GLOB ...)` becomes `CONFIGURE_DEPENDS`
- new `WGSL_TEMPLATE_FILES` glob for `*.tmpl`
- template files are added to shader-header regeneration dependencies
- C++20 is enabled
- `CXX_SCAN_FOR_MODULES` is disabled for the `ggml-webgpu` target

### `wgsl-shaders/embed_wgsl.py`

The script keeps the same overall role, but `origin/master` adjusts it for the expanded template set:

- cleaner `#decl(...)` parsing
- extra output-name branching for:
  - `SRC0_TYPE` / `SRC1_TYPE`
  - `SRC_TYPE` / `DST_TYPE`
- this matches the new generated shader families in:
  - `cpy`
  - `mul_mat`
  - `mul_mat_vec`
  - `binary`

## Shader File Families: What Actually Changed

### A. Shared Declarations And Binary Ops

Files:

- `wgsl-shaders/common_decls.tmpl`
- `wgsl-shaders/binary.wgsl`
- `wgsl-shaders/bin_op.tmpl.wgsl`
- `wgsl-shaders/binary_head.tmpl`

What changed:

- `common_decls.tmpl` grows from simple quant declarations into a much larger shared layer:
  - `ENABLE_F16`
  - `F16_TO_F32_HELPER`
  - `F16_STORAGE_HELPERS`
  - atomic `u32` load/store helpers for emulated f16 storage
  - expanded packed `_A` quant struct layouts and tables
- `binary.wgsl` becomes a much thinner kernel body
- `binary_head.tmpl` now carries the stride-based source indexing logic
- `bin_op.tmpl.wgsl` adds a new generated family for:
  - add/sub/mul/div
  - f32 variants
  - f16 variants
  - inplace variants
  - self variants
  - `_al` atomic/u32 fallback variants

This is a major reorganization, not a cosmetic rewrite.

### B. Copy, Row Access, And Set-Row Paths

Files:

- `wgsl-shaders/cpy.tmpl.wgsl`
- `wgsl-shaders/get_rows.wgsl`
- `wgsl-shaders/get_rows.tmpl.wgsl`
- `wgsl-shaders/set_rows.wgsl`

What changed:

- `cpy.tmpl.wgsl` adds explicit copy families for:
  - `f32 -> f32`
  - `f32 -> f16`
  - `f16 -> f16`
  - `f16 -> f32`
  - `_al` variants that write or read f16 through `atomic<u32>`
- `get_rows.tmpl.wgsl` is a large new generator for:
  - f32
  - f16
  - packed f16 `_al`
  - q/k quant types
  - iq quant types
  - per-type dequant logic using the expanded declarations
- `get_rows.wgsl` is reshaped around the generated template family
- `set_rows.wgsl` changes storage assumptions:
  - source rows are fixed as `f32`
  - destination can be `f32`, `f16`, or atomic/u32-backed f16
  - `stride_src0` and `stride_dst0` are added

### C. Matrix Multiply Family

Files:

- `wgsl-shaders/mul_mat.wgsl`
- `wgsl-shaders/mul_mat.tmpl.wgsl`
- `wgsl-shaders/mul_mat_decls.tmpl`
- `wgsl-shaders/mul_mat_decls_compat.tmpl`
- `wgsl-shaders/mul_mat_reg_tile.wgsl`
- `wgsl-shaders/mul_mat_reg_tile.tmpl.wgsl`
- `wgsl-shaders/mul_mat_subgroup_matrix.wgsl`
- `wgsl-shaders/mul_mat_subgroup_matrix.tmpl.wgsl`
- `wgsl-shaders/mul_mat_vec.wgsl`
- `wgsl-shaders/mul_mat_vec.tmpl.wgsl`

What changed:

- the old handwritten kernels are pushed into generated template families
- `origin/master` adds broad variant coverage for:
  - `f32 x f32`
  - `f16 x f16`
  - `f16 x f32`
  - packed `_al` f16/u32 forms
  - q-types
  - iq-types
- `mul_mat_decls_compat.tmpl` and the `_compat` declarations exist specifically to support packed/compat layouts
- vector, register-tile, and subgroup-matrix paths all gain `_al` and f32/mixed variants
- `mul_mat_subgroup_matrix.wgsl` also has a small signature cleanup in `main`

This is the biggest shader-family divergence in the directory.

### D. Activations, Norms, And Attention Helpers

Files:

- `wgsl-shaders/glu.tmpl.wgsl`
- `wgsl-shaders/rope.tmpl.wgsl`
- `wgsl-shaders/soft_max.tmpl.wgsl`
- `wgsl-shaders/row_norm.wgsl`
- `wgsl-shaders/rms_norm.wgsl`
- `wgsl-shaders/l2_norm.wgsl`
- `wgsl-shaders/unary.wgsl`
- `wgsl-shaders/scale.wgsl`
- `wgsl-shaders/flash_attn.wgsl`

What changed:

- `glu.tmpl.wgsl` adds:
  - f32 variants
  - f16 variants
  - `_al` atomic/u32-backed f16 output variants
  - explicit `SAVE_DST` vs `SAVE_DST_ATOMIC` paths
- `rope.tmpl.wgsl` adds:
  - f32 variants
  - f16 variants
  - `_al` packed-f16/u32 variants
  - inplace and non-inplace atomic write paths
  - FF and no-FF binding families
- `soft_max.tmpl.wgsl` adds:
  - f16 mask support via normal f16
  - f16 mask support via `_al` u32-backed fallback
  - sink/inplace combinations for both
- `row_norm.wgsl` is replaced by `rms_norm.wgsl`
- `l2_norm.wgsl` is added as its own dedicated shader
- `unary.wgsl` adds `TYPE_F16_AL` and routes loads/stores through atomic helpers
- `flash_attn.wgsl` changes workgroup indexing and f16 packing details
- `scale.wgsl` only changes lightly compared with the above

### E. Concatenation, Reduction, Sorting, And Utility Shaders

Files:

- `wgsl-shaders/concat.wgsl`
- `wgsl-shaders/argmax.wgsl`
- `wgsl-shaders/argsort.wgsl`
- `wgsl-shaders/argsort_merge.wgsl`
- `wgsl-shaders/cumsum.wgsl`
- `wgsl-shaders/memset.wgsl`
- `wgsl-shaders/sum_rows.wgsl`
- `wgsl-shaders/repeat.wgsl`
- `wgsl-shaders/pad.wgsl`

What changed:

- `concat.wgsl` now supports:
  - f32 input/output
  - f16 input/output
  - `_al` atomic/u32-backed f16 input/output
  - helper-based load/store instead of direct typed arrays only
- `argmax.wgsl`, `cumsum.wgsl`, `memset.wgsl`, and `sum_rows.wgsl` now use `num_workgroups` / global-linear indexing patterns
- `sum_rows.wgsl` also adds `stride_src0`
- `argsort.wgsl` and `argsort_merge.wgsl` are still the same operator family, but they are touched in the same pass as the wider dispatch/indexing cleanup
- `repeat.wgsl` drops the old mixed typing pattern and becomes much more fixed to `f32`
- `pad.wgsl` changes only lightly

### F. Model-Specific Operator Shaders

Files:

- `wgsl-shaders/conv2d.wgsl`
- `wgsl-shaders/ssm_conv.wgsl`
- `wgsl-shaders/ssm_scan_64.wgsl`
- `wgsl-shaders/ssm_scan_128.wgsl`
- `wgsl-shaders/ssm_scan_256.wgsl`
- `wgsl-shaders/tri.wgsl`
- `wgsl-shaders/diag.wgsl`
- `wgsl-shaders/solve_tri.wgsl`
- `wgsl-shaders/gated_delta_net.wgsl`

What changed:

- `conv2d.wgsl` is new and supports:
  - f32 paths
  - f16 paths
  - `_al` u32-backed f16 input / weight / output paths
- `ssm_conv.wgsl` is rewritten to the same typed load/store model as `conv2d`
- `ssm_scan_64.wgsl`, `ssm_scan_128.wgsl`, and `ssm_scan_256.wgsl` are entirely new
- `tri.wgsl` and `diag.wgsl` are entirely new
- `solve_tri.wgsl` is materially simplified:
  - old shared-memory batch structure is removed
  - indexing is rewritten around explicit base pointers and per-axis strides
- `gated_delta_net.wgsl` exists only on `constant/dev`

## Direct Truths To Carry Forward

These are the file-based facts that matter most before planning any minimal port:

1. `origin/master` is not just "the same code with different generated headers".
   - The generated header changes because the shader sources and pipeline getters really changed.

2. The biggest concrete themes on `origin/master` are:
   - logical layout correctness for broadcasted, overlapping, and view-backed tensors
   - destination-indexed writes where source-indexed writes are not correct
   - extra stride plumbing for operators that cannot assume contiguous `src`/`dst`
   - optional `ShaderF16` handling
   - explicit alias handling
   - atomic/u32-backed f16 fallback paths
   - packed quant tensor/update support
   - expanded f32 / mixed / packed shader families
   - `CONV_2D` and `SSM_SCAN`

3. The biggest concrete themes only on `constant/dev` are:
   - `GATED_DELTA_NET`
   - the older row-norm path
   - the older wait/submission model

4. The easiest way to make the port messy is to mix layout bugs, alias bugs, and no-f16 fallback work into the same patch.
   - In particular, no-`ShaderF16` work should not be used to discover or explain layout bugs.

5. Any minimal port plan must treat these as separate questions:
   - logical tensor layout correctness
   - runtime alias/dealias safety
   - runtime plumbing: capabilities, waits, diagnostics
   - precision capability and no-f16 fallback
   - packed quant support
   - operator coverage: `CONV_2D`, `SSM_SCAN`, `GATED_DELTA_NET`

## Reset Minimal-Change Reading Of The Diff

If the goal is to keep the `constant/dev` structure and progressively recover the behavior of `origin/master`, then the port order should start with correctness foundations, not precision fallback.

## Working Rules

1. Fix logical tensor layout behavior before adding no-f16 fallback.
2. Prefer validating layout fixes with `f32` or native-f16 paths first, so precision fallback is not masking indexing bugs.
3. Only add packed-u32 f16 emulation to an operator after that operator is already correct for broadcasting / overlap / view-like layouts.
4. Keep alias temp-buffer handling separate from shader indexing fixes.
5. Generated headers should be regenerated, but planning should remain source-file driven.

## Reset Feature-First Port Order

Each feature should be small enough to validate on its own. The point is to recover behavior step by step while keeping the control structure of `constant/dev`.

## Migration Status On `constant/dev` (current workspace)

This section tracks what has actually been migrated so far in this branch. It is intentionally narrower than the full plan.

### Done

- Feature 1 for the intended first slice:
  - `binary` now matches the `origin/master` layout model for the native-f16/f32 path:
    - writes are destination-indexed
    - exact destination alias modes are used (`src0 == dst`, `src1 == dst`)
    - broadcast/view indexing follows the `origin/master` shader path
  - `SET_ROWS` now has the layout/base correctness fixes:
    - copies from `base` when `base != dst`
    - scalar path uses `stride_src0` and `stride_dst0`
    - the current migrated shader path is intentionally scalar-only; the earlier local vec4 route was dropped when syncing to the fallback-capable upstream implementation
  - `ROPE` now threads `stride_src00` and `stride_dst0` so view-backed first-dimension layouts are handled correctly
- Feature 4:
  - `has_f16_support` is tracked
  - `ShaderF16` is requested conditionally
  - device creation now fails cleanly instead of asserting
- Feature 5:
  - no-`ShaderF16` fallback is now wired across the intended existing-shader slice:
    - `binary`
    - `SET_ROWS`
    - `CPY`
    - `ROPE`
    - `GLU`
    - `SOFT_MAX`
    - `GET_ROWS`
    - `UNARY`, `CLAMP`, `LOG`, `SQR`, `SQRT`, `SIN`, `COS`
    - `MUL_MAT` via the imported `origin/master` `mul_mat`, `mul_mat_vec`, `mul_mat_reg_tile`, and quantized legacy `_al` shader families
  - `common_decls.tmpl` now includes the shared packed-`u32` f16 helpers and `_T_A` quant layouts required by the upstream no-f16 shaders
  - shader/pipeline selection now routes onto the correct `_al` or A-layout variants when `has_f16_support == false`
  - backend support gating was relaxed for the same operator set so operator advertisement matches the actual no-f16 paths
  - actual `ShaderF16` capability detection is restored; the backend no longer hard-forces `has_f16_support = false`
- Feature 6:
  - packed quant upload/cache foundation is implemented:
    - `webgpu_packed_tensor`
    - `ggml_webgpu_block_stride_bytes`
    - `ggml_webgpu_get_packed_tensor`
    - `ggml_webgpu_update_packed_tensor`
    - `ggml_webgpu_update_packed_view_source`
  - upload/update now refreshes the packed cache for base tensors and compatible views
  - `GET_ROWS` uses packed quant buffers in no-`ShaderF16` mode
  - `MUL_MAT` now consumes packed quant source buffers in no-`ShaderF16` mode when the imported A-layout shaders require them
  - this is a foundation-complete claim, not a full parity/validation claim for every quantized consumer and view pattern
- Build/runtime stability cleanups tied to the above ports:
  - browser-safe future waiting and event pumping on Emscripten
  - eager-init native-f16 shaders are preprocessed before pipeline creation
  - `REPEAT(i16)` is no longer advertised on WebGPU because there is no valid no-`ShaderF16` path for it

### Partially done

- Feature 2:
  - runtime alias/de-alias handling is implemented for `binary`
  - temp-buffer staging and pre-copy commands are not yet generalized to the wider operator set
- Feature 3:
  - device lost callback
  - uncaptured error callback
  - last-submit labeling
  - browser-safe future waiting and event pumping on Emscripten
  - buffer-pool waits now process browser events instead of deadlocking the page
  - this is still a minimal local port, not the full `origin/master` submission/transient-buffer model
- Feature 7:
  - the `MUL_MAT` family has been synced enough for the imported no-`ShaderF16` paths to run:
    - `mul_mat`
    - `mul_mat_vec`
    - `mul_mat_reg_tile`
    - quantized legacy `_al` matrix kernels
  - no-`ShaderF16` routing now keeps unsupported quantized fast mat-vec cases off the emulation fast path and falls back to the legacy `_al` kernels instead
  - this is still not a claim of full `origin/master` matrix-family parity across all tuning, subgroup, and performance paths

### Not done

- Features 8 through 10 remain unported

### Explicit non-claim

- The x/y workgroup shaping work is not a separate migrated feature from `origin/master` in this branch.
- `compute_2d_workgroups(...)` is existing dispatch-shape hygiene used by the current `mul_mat` family, and binary dispatch still uses a bounded 2D launch shape.
- That should not be read as broader operator parity work by itself.

### Feature 1: Layout correctness foundation

Intent:

- recover the `origin/master` behavior around broadcasting, overlap, and view-backed tensors before touching no-f16 emulation
- make existing operators read and write according to logical `dst` layout rather than assuming source-like indexing

What belongs here:

- destination-indexed writes where `dst_i` is the correct logical write coordinate
- extra stride plumbing where `src` and `dst` first-dimension indexing differs
- view/non-contiguous destination handling
- overlap semantics at the shader/indexing level, but not temp-buffer alias helpers yet

Suggested internal order:

1. `binary`
2. `SET_ROWS`
3. `ROPE`
4. any smaller utility shader only if a failing case shows the same layout issue

What does not belong here:

- atomic/u32 f16 helpers
- `ShaderF16` capability changes
- packed quant cache/update work
- runtime alias temp-buffer helpers

Validation:

- focus first on broadcasted shapes
- then inplace / overlap / self-write cases
- then non-contiguous / view-backed source and destination cases
- prefer `f32` or native-f16 runs first to isolate layout from precision fallback

### Feature 2: Alias / de-alias runtime safety

What belongs here:

- explicit runtime helpers for overlapping buffers
- temp-buffer copy/share behavior for true alias cases
- source/destination overlap handling that cannot be solved by indexing alone

What does not belong here:

- no-f16 shader fallback
- packed quant support
- new operators

Validation:

- specifically test overlapping src/dst and self-overlap cases
- separate these results from layout-only fixes above

### Feature 3: Device diagnostics and wait stability

Intent:

- make runtime failures visible before changing execution behavior
- recover more robust queue completion behavior without mixing in shader-family work

What belongs here:

- device lost callback
- uncaptured error callback
- last-submit labeling for diagnostics
- future-based wait helper
- submit/wait bookkeeping conversion
- inflight completion handling

What does not belong here:

- alias handling
- packed quant support
- shader-file additions

Validation:

- logs become actionable when device errors occur
- repeated graph execution should not deadlock
- normal runs should complete with no timeout/assert regression

### Feature 4: Optional f16 capability negotiation

Intent:

- stop treating `ShaderF16` as an unconditional requirement during device setup
- record whether native f16 is actually available

What belongs here:

- `has_f16_support` capability tracking
- conditional `ShaderF16` feature request
- no operator changes

What does not belong here:

- no-f16 shader fallback
- packed quant logic
- layout fixes already covered above

Validation:

- adapters without `ShaderF16` should initialize instead of hard-failing
- adapters with `ShaderF16` should behave as before

### Feature 5: Existing-shader f16 fallback using `u32`

Intent:

- add no-`ShaderF16` fallback only to operators whose layout behavior is already correct

Suggested internal order:

1. `binary`
2. `SET_ROWS`
3. `CPY`
4. `ROPE`
5. `GLU`
6. `SOFT_MAX`

Important note:

- shader edits here should be as local as possible
- this stage should not be used to "discover" layout bugs; those belong to Feature 1
- keep template/generator refactors out unless they are strictly necessary

Validation:

- compare the same operator cases with and without native f16
- only use no-f16 runs after the layout cases already pass

Current branch status:

- completed for the intended existing-shader slice listed above
- the fallback now depends on the shared packed-`u32` helpers in `common_decls.tmpl`
- eager-init native-f16 shader creation also had to be preprocessed correctly to avoid raw `#ifdef` WGSL parse failures

### Feature 6: Packed quant upload/cache behavior and row access

Intent:

recover the quantized upload/update behavior needed by real inference paths

What belongs here:

- packed tensor cache/update helpers
- upload-time packed view update
- `GET_ROWS` quantized and packed paths

What does not belong here:

- new operators
- broad `MUL_MAT` family expansion

Validation:

- quantized row-access cases
- real-case inference that depends on packed quant uploads

Current branch status:

- foundation completed:
  - upload-time packed cache refresh
  - packed-view source propagation
  - `GET_ROWS` packed quant path
- `MUL_MAT` also now consumes packed quant source buffers in no-`ShaderF16` mode where the imported A-layout shaders require them
- this feature now directly underpins the no-f16 `GET_ROWS` and `MUL_MAT` paths
- this should not be read as exhaustive parity validation for every quantized consumer or every view/update edge case

### Feature 7: `MUL_MAT` family parity

Intent:

- recover the matrix path after layout, alias, capability, and basic f16 fallback are already understood

What belongs here:

- `mul_mat`
- `mul_mat_vec`
- `mul_mat_reg_tile`
- subgroup matrix path only after the base paths are correct

Validation:

- start with the smallest non-quant cases
- then mixed precision
- then quantized inputs

Current branch status:

- partially done
- the matrix-family shader sources/templates and selection logic have been ported far enough to support the current no-`ShaderF16` fallback path
- unsupported quantized fast mat-vec cases in no-`ShaderF16` mode now fall back to the legacy `_al` kernels instead of aborting in `get_mul_mat_vec_pipeline(...)`
- broader parity and tuning equivalence with `origin/master` should still be treated as open work

### Feature 8: Remaining existing operator parity

Intent:

- recover any remaining existing operators not already handled in the earlier focused stages

Examples:

- `RMS_NORM`
- `L2_NORM`
- `TRI`
- `DIAG`
- smaller utility shaders touched by the same cleanup

Validation:

- add only when the real graph or focused tests require them

### Feature 9: New operator coverage

Operators:

- `CONV_2D`
- `SSM_SCAN`

Intent:

- add operators that exist on `origin/master` but not on `constant/dev`

Validation:

- only add them when the real graph needs them, or when all earlier parity work is complete

### Feature 10: Branch divergence resolution

Main divergence:

- `constant/dev` has `GATED_DELTA_NET`
- `origin/master` has `SSM_SCAN` and no `GATED_DELTA_NET`

Intent:

- resolve this explicitly based on the actual model/operator graph, not by assuming one branch is universally right

Validation:

- confirm which path the target models actually emit

## Recommended first feature

Start with **Feature 1: Layout correctness foundation**.

Reason:

- it is the missing planning theme from the earlier version
- `origin/master` repeatedly bakes layout correctness into indexing, stride plumbing, and write semantics
- no-f16 fallback work is easier to reason about once these operators are already correct for broadcast / overlap / view-like cases

## Validation Reminder

This document is a branch diff review, not a correctness proof. Before porting any feature:

- test the exact failing real-case graph
- confirm which operator path is missing or broken
- keep generated headers regenerated, but review the source shader/template changes instead
