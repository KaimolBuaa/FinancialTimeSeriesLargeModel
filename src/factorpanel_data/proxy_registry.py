"""Immutable ProxyFactor-v0 factor and label definitions."""

from __future__ import annotations

from dataclasses import dataclass


FACTOR_WINDOWS = (2, 3, 5, 10, 20, 30, 60, 120)


@dataclass(frozen=True)
class FactorDefinition:
    name: str
    expression: str
    family: str
    window: int | None = None


@dataclass(frozen=True)
class LabelDefinition:
    name: str
    expression: str
    horizon: int


def build_proxy_factor_registry() -> tuple[FactorDefinition, ...]:
    base = (
        FactorDefinition("pf_kmid", "($close-$open)/($open+1e-12)", "kbar"),
        FactorDefinition("pf_klen", "($high-$low)/($open+1e-12)", "kbar"),
        FactorDefinition(
            "pf_kmid2", "($close-$open)/($high-$low+1e-12)", "kbar"
        ),
        FactorDefinition(
            "pf_kup", "($high-Greater($open,$close))/($open+1e-12)", "kbar"
        ),
        FactorDefinition(
            "pf_kup2",
            "($high-Greater($open,$close))/($high-$low+1e-12)",
            "kbar",
        ),
        FactorDefinition(
            "pf_klow", "(Less($open,$close)-$low)/($open+1e-12)", "kbar"
        ),
        FactorDefinition(
            "pf_klow2",
            "(Less($open,$close)-$low)/($high-$low+1e-12)",
            "kbar",
        ),
        FactorDefinition(
            "pf_ksft", "(2*$close-$high-$low)/($open+1e-12)", "kbar"
        ),
        FactorDefinition(
            "pf_ksft2",
            "(2*$close-$high-$low)/($high-$low+1e-12)",
            "kbar",
        ),
        FactorDefinition(
            "pf_open_close", "$open/($close+1e-12)-1", "price_ratio"
        ),
        FactorDefinition(
            "pf_high_close", "$high/($close+1e-12)-1", "price_ratio"
        ),
        FactorDefinition(
            "pf_low_close", "$low/($close+1e-12)-1", "price_ratio"
        ),
        FactorDefinition(
            "pf_vwap_close", "$vwap/($close+1e-12)-1", "price_ratio"
        ),
        FactorDefinition(
            "pf_return_1", "$close/(Ref($close,1)+1e-12)-1", "change"
        ),
        FactorDefinition(
            "pf_volume_change_1",
            "$volume/(Ref($volume,1)+1e-12)-1",
            "change",
        ),
        FactorDefinition(
            "pf_amount_change_1",
            "$amount/(Ref($amount,1)+1e-12)-1",
            "change",
        ),
    )
    templates = {
        "roc": "$close/(Ref($close,{w})+1e-12)-1",
        "ma": "$close/(Mean($close,{w})+1e-12)-1",
        "std": "Std($close,{w})/($close+1e-12)",
        "beta": "Slope($close,{w})/($close+1e-12)",
        "rsqr": "Rsquare($close,{w})",
        "max": "$close/(Max($high,{w})+1e-12)-1",
        "min": "$close/(Min($low,{w})+1e-12)-1",
        "rsv": (
            "($close-Min($low,{w}))/(Max($high,{w})-Min($low,{w})+1e-12)"
        ),
        "corr": "Corr($close,Log($volume+1),{w})",
        "cord": (
            "Corr($close/(Ref($close,1)+1e-12)-1,"
            "Log($volume/(Ref($volume,1)+1e-12)+1),{w})"
        ),
        "cntd": (
            "Mean($close>Ref($close,1),{w})-"
            "Mean($close<Ref($close,1),{w})"
        ),
        "sumd": (
            "(Sum(Greater($close-Ref($close,1),0),{w})-"
            "Sum(Greater(Ref($close,1)-$close,0),{w}))/"
            "(Sum(Abs($close-Ref($close,1)),{w})+1e-12)"
        ),
        "vma": "$volume/(Mean($volume,{w})+1e-12)-1",
        "vstd": "Std($volume,{w})/(Mean($volume,{w})+1e-12)",
    }
    rolling = tuple(
        FactorDefinition(
            name=f"pf_{family}_{window}",
            expression=template.format(
                w=max(window, 3) if family == "rsqr" else window
            ),
            family=family,
            window=window,
        )
        for family, template in templates.items()
        for window in FACTOR_WINDOWS
    )
    registry = base + rolling
    if len(registry) != 128 or len({item.name for item in registry}) != 128:
        raise RuntimeError("ProxyFactor-v0 registry must contain 128 unique factors")
    if any(",-" in item.expression.replace(" ", "") for item in registry):
        raise RuntimeError("ProxyFactor-v0 factor expressions must be causal")
    return registry


def build_label_registry() -> tuple[LabelDefinition, ...]:
    return tuple(
        LabelDefinition(
            name=f"ret_{horizon}d",
            expression=f"Log(Ref($close,-{horizon})/($close+1e-12))",
            horizon=horizon,
        )
        for horizon in (1, 5, 20)
    )


PROXY_FACTOR_REGISTRY = build_proxy_factor_registry()
PROXY_LABEL_REGISTRY = build_label_registry()
