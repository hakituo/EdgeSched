import json
import sys

def check_consistency():
    for i in [1, 2]:
        with open(f'/home/leslie/mvp_core/paper/experiment_data/03_raw_iterations/migration/edgesched_migration_iter{i}.json') as f:
            d = json.load(f)
        print(f'=== iter{i} ===')
        for k in ['migration_stutter_ms', 'coexistence_cpu_ttft_ms', 'restore_time_ms', 'restore_window_max_gap_ms', 'llm_gpu_ttft_ms']:
            print(f'  {k}: {d.get(k, "N/A")}')

    with open('/home/leslie/mvp_core/paper/experiment_data/01_paper_main/migration/edgesched_aggregated.json') as f:
        agg = json.load(f)
    print('=== aggregated ===')
    for k, v in agg.get('metrics', {}).items():
        print(f'  {k}: mean={v["mean"]:.2f}, values={v["values"]}')

    print('=== 验证 ===')
    all_match = True
    for k in ['migration_stutter_ms', 'coexistence_cpu_ttft_ms', 'restore_time_ms', 'restore_window_max_gap_ms', 'llm_gpu_ttft_ms']:
        iter_vals = []
        for i in [1, 2]:
            with open(f'/home/leslie/mvp_core/paper/experiment_data/03_raw_iterations/migration/edgesched_migration_iter{i}.json') as f:
                d = json.load(f)
            iter_vals.append(d.get(k))
        if k in agg['metrics']:
            agg_vals = agg['metrics'][k]['values']
            match = iter_vals == agg_vals
        else:
            # 值为0的指标可能被过滤
            match = all(v == 0.0 for v in iter_vals)
            agg_vals = '(filtered, all zeros)'
        if not match:
            all_match = False
        print(f'  {k}: iter={iter_vals}, agg={agg_vals}, match={match}')

    if all_match:
        print('\n✅ 所有指标一致！raw iter 和 aggregated 完全对齐。')
    else:
        print('\n❌ 存在不一致！')
    return all_match

if __name__ == '__main__':
    ok = check_consistency()
    sys.exit(0 if ok else 1)
