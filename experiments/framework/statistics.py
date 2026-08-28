"""
统计分析模块

提供统一的统计分析功能，包括：
- 描述性统计
- 置信区间计算
- 统计显著性检验
"""

import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LatencyStats:
    """延迟统计数据结构"""
    
    n: int
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_dev_ms: float = 0.0
    ci_lower_ms: Optional[float] = None
    ci_upper_ms: Optional[float] = None
    confidence: float = 0.95
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "n": self.n,
            "avg_ms": self.avg_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "std_dev_ms": self.std_dev_ms,
        }
        if self.ci_lower_ms is not None and self.ci_upper_ms is not None:
            result["ci_lower_ms"] = self.ci_lower_ms
            result["ci_upper_ms"] = self.ci_upper_ms
            result["confidence"] = self.confidence
        return result


class StatisticsAnalyzer:
    """统计分析器"""
    
    def __init__(self, confidence: float = 0.95):
        """
        初始化统计分析器
        
        Args:
            confidence: 置信水平，默认 0.95
        """
        self.confidence = confidence
    
    def calc_percentile(self, sorted_values: List[float], p: float) -> float:
        """
        计算百分位数
        
        Args:
            sorted_values: 已排序的值列表
            p: 百分位 (0.0 - 1.0)
            
        Returns:
            float: 百分位数值
        """
        n = len(sorted_values)
        if n == 0:
            return 0.0
        if n == 1:
            return sorted_values[0]
        
        idx = (n - 1) * p
        low = int(idx)
        high = min(n - 1, low + 1)
        weight = idx - low
        
        return sorted_values[low] * (1 - weight) + sorted_values[high] * weight
    
    def calc_mean(self, values: List[float]) -> float:
        """计算均值"""
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    def calc_std_dev(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        mean = self.calc_mean(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)
    
    def calc_confidence_interval(
        self, 
        values: List[float], 
        confidence: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        计算置信区间
        
        Args:
            values: 值列表
            confidence: 置信水平，默认使用实例配置
            
        Returns:
            Tuple[float, float]: (下界, 上界)
        """
        if len(values) < 2:
            mean = self.calc_mean(values)
            return (mean, mean)
        
        conf = confidence or self.confidence
        mean = self.calc_mean(values)
        std_err = self.calc_std_dev(values) / math.sqrt(len(values))
        
        try:
            from scipy import stats
            h = std_err * stats.t.ppf((1 + conf) / 2, len(values) - 1)
        except ImportError:
            z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(conf, 1.96)
            h = std_err * z
        
        return (mean - h, mean + h)
    
    def analyze_latencies(
        self, 
        latencies_seconds: List[float],
        include_ci: bool = True
    ) -> LatencyStats:
        """
        分析延迟数据
        
        Args:
            latencies_seconds: 延迟数据（秒）
            include_ci: 是否包含置信区间
            
        Returns:
            LatencyStats: 延迟统计结果
        """
        if not latencies_seconds:
            return LatencyStats(
                n=0, avg_ms=0.0, p50_ms=0.0, p95_ms=0.0, p99_ms=0.0,
                min_ms=0.0, max_ms=0.0, std_dev_ms=0.0
            )
        
        sorted_latencies = sorted(latencies_seconds)
        n = len(sorted_latencies)
        
        avg = self.calc_mean(latencies_seconds)
        std_dev = self.calc_std_dev(latencies_seconds)
        
        ci_lower, ci_upper = None, None
        if include_ci and n >= 2:
            ci_lower, ci_upper = self.calc_confidence_interval(latencies_seconds)
        
        return LatencyStats(
            n=n,
            avg_ms=avg * 1000,
            p50_ms=self.calc_percentile(sorted_latencies, 0.50) * 1000,
            p95_ms=self.calc_percentile(sorted_latencies, 0.95) * 1000,
            p99_ms=self.calc_percentile(sorted_latencies, 0.99) * 1000,
            min_ms=min(latencies_seconds) * 1000,
            max_ms=max(latencies_seconds) * 1000,
            std_dev_ms=std_dev * 1000,
            ci_lower_ms=ci_lower * 1000 if ci_lower is not None else None,
            ci_upper_ms=ci_upper * 1000 if ci_upper is not None else None,
            confidence=self.confidence if include_ci else 0.95,
        )
    
    def calc_degradation_percent(
        self, 
        baseline: float, 
        stress: float
    ) -> float:
        """
        计算退化百分比
        
        Args:
            baseline: 基线值
            stress: 压力值
            
        Returns:
            float: 退化百分比（正数表示退化）
        """
        if baseline <= 0:
            if stress > 0:
                return float('inf')
            return 0.0
        return ((stress - baseline) / baseline) * 100
    
    def calc_tail_latency_ratio(
        self, 
        avg_ms: float, 
        p99_ms: float
    ) -> float:
        """
        计算尾延迟比
        
        Args:
            avg_ms: 平均延迟
            p99_ms: P99 延迟
            
        Returns:
            float: 尾延迟比
        """
        if avg_ms <= 0:
            return 1.0
        return p99_ms / avg_ms


def report_with_confidence(
    samples: List[float], 
    confidence: float = 0.95
) -> Dict[str, Any]:
    """
    生成带置信区间的统计报告
    
    Args:
        samples: 样本数据
        confidence: 置信水平
        
    Returns:
        Dict: 统计报告
    """
    analyzer = StatisticsAnalyzer(confidence)
    stats = analyzer.analyze_latencies(samples, include_ci=True)
    return stats.to_dict()


def compare_distributions(
    baseline: List[float],
    treatment: List[float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    比较两个分布的统计显著性
    
    Args:
        baseline: 基线数据
        treatment: 处理组数据
        alpha: 显著性水平
        
    Returns:
        Dict: 比较结果
    """
    analyzer = StatisticsAnalyzer()
    
    baseline_stats = analyzer.analyze_latencies(baseline)
    treatment_stats = analyzer.analyze_latencies(treatment)
    
    result = {
        "baseline": baseline_stats.to_dict(),
        "treatment": treatment_stats.to_dict(),
        "improvement": {
            "avg_ms": baseline_stats.avg_ms - treatment_stats.avg_ms,
            "p99_ms": baseline_stats.p99_ms - treatment_stats.p99_ms,
        },
        "improvement_percent": {
            "avg_ms": analyzer.calc_degradation_percent(
                baseline_stats.avg_ms, treatment_stats.avg_ms
            ),
            "p99_ms": analyzer.calc_degradation_percent(
                baseline_stats.p99_ms, treatment_stats.p99_ms
            ),
        },
    }
    
    try:
        from scipy import stats
        if len(baseline) >= 2 and len(treatment) >= 2:
            stat, p_value = stats.mannwhitneyu(baseline, treatment, alternative='two-sided')
            result["statistical_test"] = {
                "test": "mann-whitney-u",
                "statistic": float(stat),
                "p_value": float(p_value),
                "significant": p_value < alpha,
            }
    except (ImportError, ValueError):
        pass
    
    return result
