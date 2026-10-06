# P100 first-logits diagnostic twins

These sources instrument the API serve decode loop for an isolated correctness check. The control parent is
`6f32ec0`; the native-VMAD candidate parent is `2d17ce0`. Both receive the identical `generate.cpp` patch.
Performance binaries remain separate and unchanged.

`serve.server` launches the engine with `--serve`. Its decode loop previously skipped the standalone
`STRATA_DUMP_FIRST_LOGITS` hook. With that environment variable set, the first decode verifier window now writes
row zero of the complete target vocabulary as raw float32 values before commit. Each API request has its own
first window. Later windows do not overwrite the file. Copy, open, write, or close failures emit `ERR` and fail
the request. Without the variable, the diagnostic does no copies or file I/O.

The verifier forwards vocabulary size and logits copying through a layer split to the stage holding the head.
The existing `STRATA_LOGPOS` diagnostic remains limited to eligible teacher-forced prompt windows.

Run the CPU/source check with `uv run --no-project python tests/test_first_logits_diagnostic.py`.
It compiles the exact new diagnostic block, checks its serve-loop placement and raw output, and uses Linux
`/dev/full` to check real write and close failures. It does not prove GPU copying or model logit equality.
Build and validate diagnostic control/candidate engines separately before drawing that conclusion.
