import json

for variant in ['edgesched_migration', 'symmetric_migration']:
    raw_path = f'/home/leslie/mvp_core/paper/experiment_data/03_raw_iterations/migration/{variant}_iter1.json'
    name_map = {
        'edgesched_migration': 'edgesched_aggregated',
        'symmetric_migration': 'symmetric_aggregated',
        'fast_down_naive_up': 'fast_down_naive_up_aggregated',
        'naive_down_optimized_up': 'naive_down_optimized_up_aggregated',
    }
    agg_path = f'/home/leslie/mvp_core/paper/experiment_data/01_paper_main/migration/{name_map[variant]}.json'
    try:
        with open(raw_path) as f:
            raw = json.load(f)
        with open(agg_path) as f:
            agg = json.load(f)
    except FileNotFoundError as e:
        print(f'❌ {variant}: 文件不存在 - {e}')
        continue

    print(f'=== {variant} ===')
    all_match = True
    for k in ['migration_stutter_ms', 'coexistence_cpu_ttft_ms', 'restore_time_ms', 'llm_gpu_ttft_ms']:
        raw_val = raw.get(k)
        agg_vals = agg['metrics'].get(k, {}).get('values', [])
        if agg_vals:
            match = [raw_val] == agg_vals
        else:
            match = raw_val == 0.0
        if not match:
            all_match = False
        print(f'  {k}: raw={raw_val}, agg={agg_vals if agg_vals else "(filtered, val=0)"}, match={match}')
    print(f'  {"✅ 一致" if all_match else "❌ 不一致"}')
    print()
