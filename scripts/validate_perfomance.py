#!/usr/bin/env python
# scripts/validate_performance.py
"""Validate that performance metrics are statistically meaningful"""

import asyncio
import numpy as np
from scipy import stats
from typing import Dict
from app.db.database import AsyncSessionLocal
from app.db.models import CLVEntry, Prediction
from app.services.statistical_significance import StatisticalSignificance


async def validate_performance():
    """Validate that performance is statistically significant"""
    async with AsyncSessionLocal() as db:
        # Get all CLV entries
        result = await db.execute(select(CLVEntry.clv).where(CLVEntry.clv.isnot(None)))
        clv_values = [row[0] for row in result.all()]

        print("=" * 60)
        print("VIT NETWORK - PERFORMANCE VALIDATION")
        print("=" * 60)

        print(f"\n📊 Sample Size: {len(clv_values)} bets")

        if len(clv_values) < 30:
            print(f"⚠️ INSUFFICIENT DATA: Need 30+ bets for statistical significance")
            print(f"   Current: {len(clv_values)} bets")
            print(f"   Additional needed: {30 - len(clv_values)}")
            return

        # Statistical significance test
        t_stat, p_value = stats.ttest_1samp(clv_values, 0)
        mean_clv = np.mean(clv_values)

        print(f"\n📈 CLV Statistics:")
        print(f"   Mean CLV: {mean_clv:.4f}")
        print(f"   Std Dev: {np.std(clv_values):.4f}")
        print(f"   Min: {np.min(clv_values):.4f}")
        print(f"   Max: {np.max(clv_values):.4f}")

        print(f"\n🔬 Statistical Significance:")
        print(f"   T-statistic: {t_stat:.4f}")
        print(f"   P-value: {p_value:.6f}")

        if p_value < 0.05 and mean_clv > 0:
            print(f"   ✅ SIGNIFICANT EDGE DETECTED")
            print(f"   Interpretation: {mean_clv:.2%} edge with {100*(1-p_value):.1f}% confidence")
        elif p_value < 0.05 and mean_clv < 0:
            print(f"   ❌ SIGNIFICANT LOSS DETECTED")
            print(f"   System is consistently losing - needs review")
        else:
            print(f"   ⚠️ INSUFFICIENT EVIDENCE")
            print(f"   Cannot conclude edge exists with statistical confidence")

        # Required sample size for confidence
        required = StatisticalSignificance.required_sample_size(mean_clv, np.std(clv_values))
        print(f"\n🎯 Required sample size for 95% confidence: {required:.0f} bets")
        print(f"   Current progress: {len(clv_values)/required:.1%}")

        # Confidence interval
        ci_lower, ci_upper = StatisticalSignificance.calculate_confidence_interval(clv_values)
        print(f"\n📏 95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

        if ci_lower > 0:
            print(f"   ✅ Even lower bound is positive - strong signal")
        elif ci_upper < 0:
            print(f"   ❌ Even upper bound is negative - system is broken")
        else:
            print(f"   ⚠️ Interval crosses zero - more data needed")


if __name__ == "__main__":
    asyncio.run(validate_performance())