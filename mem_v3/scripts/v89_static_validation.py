#!/usr/bin/env python3
from __future__ import annotations
import ast, pathlib, re, sys
root = pathlib.Path(__file__).resolve().parents[1]
errors=[]
controller=root/'scripts'/'v89_sustained_controller.py'
run=root/'scripts'/'run_v89_sustained_control.sh'
monitor=root/'scripts'/'monitor_v89_human.sh'
for p in [controller, run, monitor, root/'runtime'/'real_dataset.py', root/'runtime'/'lm_model.py']:
    if not p.exists(): errors.append(f'missing:{p.relative_to(root)}')
for p in [controller, root/'runtime'/'real_dataset.py', root/'runtime'/'lm_model.py', root/'runtime'/'deepspeed_runner.py']:
    try: ast.parse(p.read_text())
    except Exception as e: errors.append(f'python_parse_failed:{p.relative_to(root)}:{e}')
s=controller.read_text()
rs=run.read_text() if run.exists() else ''
ms=monitor.read_text() if monitor.exists() else ''
if 'VERSION = "v89.0.0"' not in s: errors.append('controller_version_not_v89')
if 'CONFIRM = "I_UNDERSTAND_V89_RECOVERY_CONTROL"' not in s: errors.append('controller_confirm_token_not_v89')
if ('I_UNDERSTAND_' + chr(86) + '80') in s or ('I_UNDERSTAND_' + chr(86) + '80') in rs: errors.append('stale_confirm_token_present')
if 'python scripts/v89_sustained_controller.py' not in rs: errors.append('run_script_not_calling_v89_controller')
if 'MEM_V89_ENABLE_API' not in rs: errors.append('run_script_missing_v89_api_flag')
if 'OPENAI_API_KEY' in rs: errors.append('run_script_requires_openai_key')
if 'v89_mandatory_floor_escape' not in s: errors.append('mandatory_floor_escape_missing')
if 'v89_api_keep_overridden_by_floor_escape' not in s: errors.append('api_keep_override_event_missing')
if 'productive_lane_switch_blocked_v89_midband' not in s: errors.append('v89_midband_guard_event_missing')
if 'v89_midband_switch_blocked' not in s: errors.append('v89_midband_guard_function_missing')
if 'below_lane_min_tokens_plus_optimizer_pressure' not in s: errors.append('override_reason_missing')
if 'V89 mandatory floor escape' not in s: errors.append('api_prompt_floor_escape_missing')
if '"--llm", "--api-executive-mode"' not in s: errors.append('controller_missing_llm_api_flags')
models=(root/'domain'/'models.py').read_text()
if 'VERSION = "v89.0.0"' not in models: errors.append('domain_models_version_not_v89')
if 'CONFIRMATION_TOKEN = "I_UNDERSTAND_V89_RECOVERY_CONTROL"' not in models: errors.append('domain_models_confirm_token_not_v89')
if ('I_UNDERSTAND_' + chr(86) + '80') in models: errors.append('domain_models_stale_confirm_token_present')
orch=(root/'core'/'orchestrator.py').read_text()
if 'new_evidence_dir("v89")' not in orch: errors.append('orchestrator_evidence_prefix_not_v89')
if 'v89_adaptive_benchmark_session.json' not in orch: errors.append('orchestrator_session_filename_not_v89')
if 'v89_sustained_control_events.jsonl' not in ms: errors.append('monitor_not_reading_v89_events')
for lane in ['aggressive_seq256_zero0_gacc4','fast_seq256_zero0_gacc4','safe_seq256']:
    if lane not in s: errors.append(f'lane_missing:{lane}')
for forbidden in ['zero0_gacc5','zero0_gacc3','safe_seq512','fire_seq512']:
    if forbidden in re.sub(r'No gacc3/gacc5 probes, no seq512 probes, no API lane creativity.', '', s):
        errors.append(f'forbidden_lane_reference:{forbidden}')
rd=(root/'runtime'/'real_dataset.py').read_text()
if 'def prewarm' not in rd: errors.append('dataset_prewarm_missing')
if 'MEM_DATASET_CACHE_MODE' not in rd or 'memmap' not in rd: errors.append('dataset_memmap_missing')
lm=(root/'runtime'/'lm_model.py').read_text()
if 'register_buffer("causal_mask"' not in lm: errors.append('static_causal_mask_missing')
if errors:
    print('V89_STATIC_VALIDATION: FAIL')
    for e in errors: print(' -', e)
    sys.exit(1)
print('V89_STATIC_VALIDATION: PASS')
