import json

with open('/home/leslie/mvp_core/paper/experiment_data/03_raw_iterations/migration/edgesched_migration_iter1.json') as f:
    raw = json.load(f)
with open('/home/leslie/mvp_core/paper/experiment_data/01_paper_main/migration/edgesched_aggregated.json') as f:
    agg = json.load(f)

print('=== raw iter1 ===')
for k in ['migration_stutter_ms', 'coexistence_cpu_ttft_ms', 'restore_time_ms', 'llm_gpu_ttft_ms']:
    print(f'  {k}: {raw.get(k)}')

print('=== aggregated ===')
for k, v in agg.get('metrics', {}).items():
    print(f'  {k}: mean={v["mean"]:.2f}, values={v["values"]}')

print('=== 验证 ===')
all_match = True
for k in ['migration_stutter_ms', 'coexistence_cpu_ttft_ms', 'restore_time_ms', 'llm_gpu_ttft_ms']:
    raw_val = raw.get(k)
    agg_vals = agg['metrics'].get(k, {}).get('values', [])
    match = [raw_val] == agg_vals
    if not match:
        all_match = False
    print(f'  {k}: raw={raw_val}, agg={agg_vals}, match={match}')

print()
print('✅ 一致!' if all_match else '❌ 不一致!')
