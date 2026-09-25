from figures import (
    plot_migration_timeline, plot_migration_comparison, plot_ablation_comparison, plot_resource_migration,
    plot_crash_recovery, plot_stress_test,
    plot_cold_start_comparison, plot_lru_thrashing, plot_baseline_comparison,
    plot_real_workload_trace,
    FIG_MIGRATION, FIG_ROBUSTNESS, FIG_BASELINE, FIG_AUXILIARY,
)

FIG_MIGRATION.mkdir(parents=True, exist_ok=True)
FIG_ROBUSTNESS.mkdir(parents=True, exist_ok=True)
FIG_BASELINE.mkdir(parents=True, exist_ok=True)
FIG_AUXILIARY.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("EdgeSched Paper Figure Generation")
    print("=" * 60)

    print("\n[1/9] Migration Timeline (Symmetric vs Asymmetric)...")
    plot_migration_timeline()

    print("\n[2/9] Migration Comparison (Grouped Bar)...")
    plot_migration_comparison()

    print("\n[3/9] Ablation Comparison...")
    plot_ablation_comparison()

    print("\n[4/9] Resource Migration (EdgeSched N=5)...")
    plot_resource_migration()

    print("\n[5/9] Crash Recovery...")
    plot_crash_recovery()

    print("\n[6/9] Cold Start Comparison...")
    plot_cold_start_comparison()

    print("\n[7/9] LRU Thrashing...")
    plot_lru_thrashing()

    print("\n[8/9] Real Workload Trace...")
    plot_real_workload_trace()

    print("\n[9/9] Baseline Comparison + Stress Test...")
    plot_baseline_comparison()
    plot_stress_test()

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
