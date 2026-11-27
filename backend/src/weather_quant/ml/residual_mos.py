"""Residual LightGBM MOS over Level-1 EMOS with bounded corrections & Temperature Scaling.

Fits LightGBM quantile boosters on residuals ``y_obs - baseline_emos_mean`` with
bounded corrections ``[-max_delta_c, +max_delta_c]`` so extreme heatwaves/cold snaps
extrapolate linearly via EMOS rather than plateauing at tree leaf bounds.
"""

from __future__ import annotations

import asyncio
import json
import math
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize_scalar

from weather_quant.domain.clock import Clock, SystemClock, ensure_utc
from weather_quant.domain.models import (
    EnsembleSnapshot,
    EventProbabilitySnapshot,
    MetarObservation,
    ModelVersionRecord,
    Station,
    WeatherEvent,
)
from weather_quant.ml.dataset_builder import (
    FEATURE_COLUMNS,
    TOP_LIQUIDITY_STATIONS,
    MosDatasetSample,
    features_to_vector,
)
from weather_quant.ml.quantile_mos import (
    DEFAULT_LGBM_PARAMS,
    DEFAULT_MODELS_DIR,
    DEFAULT_QUANTILES,
    ChronologicalBenchmarkReport,
    QuantileMosModel,
    evaluate_mos_vs_baseline,
    register_model_version,
    split_chronologically,
)
from weather_quant.probability.engine import DailyBiasObservation, map_extremes_c_to_probabilities

if TYPE_CHECKING:
    from weather_quant.storage.database import Database

DEFAULT_MAX_RESIDUAL_DELTA_C = 3.0
_LOG_EPS = 1e-9


def apply_temperature_scaling(
    probs: Sequence[float],
    temperature: float,
    impossible_mask: Sequence[bool] | None = None,
) -> list[float]:
    """Apply 1-parameter logit Temperature Scaling ``p_i^(1/tau) / Z`` preserving sum=1."""
    n = len(probs)
    if n == 0:
        return []
    tau = max(0.25, min(4.0, float(temperature)))
    mask = list(impossible_mask) if impossible_mask is not None else [False] * n
    scaled = [
        (max(0.0, float(p)) ** (1.0 / tau)) if (not mask[i] and p > 0.0) else 0.0
        for i, p in enumerate(probs)
    ]
    total = sum(scaled)
    if total <= 0.0:
        possible = [i for i in range(n) if not mask[i]]
        if not possible:
            return [0.0] * n
        u = 1.0 / len(possible)
        return [u if not mask[i] else 0.0 for i in range(n)]
    out = [x / total for x in scaled]
    drift = 1.0 - sum(out)
    if drift != 0.0:
        anchor = max((i for i in range(n) if not mask[i]), key=lambda i: out[i])
        out[anchor] = max(0.0, out[anchor] + drift)
    return out


def fit_temperature_scaling(
    probs_list: Sequence[Sequence[float]],
    true_indices: Sequence[int],
) -> float:
    """Fit 1-parameter Temperature Scaling minimizing out-of-fold Multi-Class Log-Loss."""
    if not probs_list or len(probs_list) != len(true_indices):
        return 1.0

    def nll(tau: float) -> float:
        loss = 0.0
        for probs, y_idx in zip(probs_list, true_indices):
            cal = apply_temperature_scaling(probs, tau)
            p_true = cal[y_idx] if 0 <= y_idx < len(cal) else 0.0
            loss -= math.log(max(_LOG_EPS, min(1.0, p_true)))
        return loss / len(probs_list)

    res = minimize_scalar(nll, bounds=(0.5, 2.5), method="bounded")
    return round(float(res.x), 4) if res.success else 1.0


def evaluate_station_overfit_gate(
    *,
    mos_val_brier: float,
    base_val_brier: float,
    mos_test_brier: float,
    base_test_brier: float,
    mos_train_pinball: float | None = None,
    mos_val_pinball: float | None = None,
    max_train_val_ratio: float = 2.0,
) -> bool:
    """Return True only if ML MOS beats EMOS baseline on both val & test without severe train/val gap."""
    if mos_val_brier > base_val_brier + 1e-6:
        return False
    if mos_test_brier > base_test_brier + 1e-6:
        return False
    if (
        mos_train_pinball is not None
        and mos_val_pinball is not None
        and mos_train_pinball > 1e-6
        and (mos_val_pinball / mos_train_pinball) > max_train_val_ratio
    ):
        return False
    return True


class ResidualMosModel(QuantileMosModel):
    """Residual Quantile LightGBM MOS predicting bounded corrections over EMOS baseline."""

    def __init__(
        self,
        *,
        model_version: str,
        quantiles: Sequence[float],
        boosters: dict[float, lgb.Booster],
        covered_stations: Sequence[str] = TOP_LIQUIDITY_STATIONS,
        feature_columns: Sequence[str] = FEATURE_COLUMNS,
        lgbm_params: dict[str, Any] | None = None,
        max_residual_delta_c: float = DEFAULT_MAX_RESIDUAL_DELTA_C,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(
            model_version=model_version,
            quantiles=quantiles,
            boosters=boosters,
            covered_stations=covered_stations,
            feature_columns=feature_columns,
            lgbm_params=lgbm_params,
        )
        self.max_residual_delta_c = float(max_residual_delta_c)
        self.temperature = float(temperature)

    @classmethod
    def fit(
        cls,
        train_samples: Sequence[MosDatasetSample],
        *,
        model_version: str,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        covered_stations: Sequence[str] = TOP_LIQUIDITY_STATIONS,
        lgbm_params: dict[str, Any] | None = None,
        max_residual_delta_c: float = DEFAULT_MAX_RESIDUAL_DELTA_C,
        temperature: float = 1.0,
    ) -> ResidualMosModel:
        """Fit LightGBM quantile boosters on residuals ``y_obs - baseline_mean_c``."""
        valid = [s for s in train_samples if s.target_temp_c is not None]
        if not valid:
            raise ValueError("Cannot train ResidualMosModel on empty training dataset.")

        x_train = np.asarray([features_to_vector(s.features) for s in valid], dtype=np.float64)
        y_residual = np.asarray(
            [float(s.target_temp_c) - float(s.baseline_mean_c) for s in valid],  # type: ignore[arg-type]
            dtype=np.float64,
        )

        merged_params = dict(DEFAULT_LGBM_PARAMS)
        if lgbm_params:
            merged_params.update(lgbm_params)

        num_boost_round = int(merged_params.get("n_estimators", 80))
        core_params: dict[str, Any] = {
            "objective": "quantile",
            "boosting_type": "gbdt",
            "learning_rate": float(merged_params.get("learning_rate", 0.06)),
            "num_leaves": int(merged_params.get("num_leaves", 12)),
            "max_depth": int(merged_params.get("max_depth", 4)),
            "min_data_in_leaf": int(merged_params.get("min_child_samples", 5)),
            "feature_fraction": float(merged_params.get("colsample_bytree", 0.85)),
            "bagging_fraction": float(merged_params.get("subsample", 0.85)),
            "bagging_freq": 1,
            "reg_alpha": float(merged_params.get("reg_alpha", 0.1)),
            "reg_lambda": float(merged_params.get("reg_lambda", 1.0)),
            "seed": int(merged_params.get("random_state", 42)),
            "verbose": -1,
            "num_threads": int(merged_params.get("num_threads", 1)),
        }

        sorted_q = tuple(sorted(float(q) for q in quantiles))
        boosters: dict[float, lgb.Booster] = {}
        for q in sorted_q:
            ds = lgb.Dataset(
                x_train,
                label=y_residual,
                feature_name=list(FEATURE_COLUMNS),
                free_raw_data=False,
            )
            boosters[q] = lgb.train(dict(core_params, alpha=q), ds, num_boost_round=num_boost_round)

        return cls(
            model_version=model_version,
            quantiles=sorted_q,
            boosters=boosters,
            covered_stations=covered_stations,
            feature_columns=FEATURE_COLUMNS,
            lgbm_params=merged_params,
            max_residual_delta_c=max_residual_delta_c,
            temperature=temperature,
        )

    def predict_quantiles(
        self,
        feature_dicts: Sequence[dict[str, float]],
    ) -> list[dict[float, float]]:
        """Predict non-crossing quantiles: ``emos_base + clip(delta_q, -max_delta, +max_delta)``."""
        if not feature_dicts:
            return []
        x_mat = np.asarray(
            [features_to_vector(f, columns=self.feature_columns) for f in feature_dicts],
            dtype=np.float64,
        )
        raw_by_q = {q: self.boosters[q].predict(x_mat) for q in self.quantiles}

        results: list[dict[float, float]] = []
        cap = self.max_residual_delta_c
        median_q = min(self.quantiles, key=lambda q: abs(q - 0.5))
        for idx, feat in enumerate(feature_dicts):
            emos_base = float(feat["ensemble_mean_c"]) + float(feat.get("rolling_bias_c", 0.0))
            raw_median = float(raw_by_q[median_q][idx])
            clamped_median = max(-cap, min(cap, raw_median))
            shift_adj = clamped_median - raw_median
            deltas = sorted(float(raw_by_q[q][idx]) + shift_adj for q in self.quantiles)
            results.append({q: emos_base + d for q, d in zip(self.quantiles, deltas)})
        return results

    def predict_event_probabilities(
        self,
        *,
        station: Station,
        event: WeatherEvent,
        ensemble: EnsembleSnapshot,
        recent_bias_obs: Sequence[DailyBiasObservation] = (),
        intraday_metars: Sequence[MetarObservation] = (),
        clock: Clock | None = None,
        strict_temporal: bool = False,
        baseline_snapshot: EventProbabilitySnapshot | None = None,
    ) -> EventProbabilitySnapshot:
        snap = super().predict_event_probabilities(
            station=station,
            event=event,
            ensemble=ensemble,
            recent_bias_obs=recent_bias_obs,
            intraday_metars=intraday_metars,
            clock=clock,
            strict_temporal=strict_temporal,
            baseline_snapshot=baseline_snapshot,
        )
        if abs(self.temperature - 1.0) <= 1e-6:
            return snap

        scaled = apply_temperature_scaling(
            [b.fair_probability for b in snap.buckets],
            self.temperature,
            impossible_mask=[b.is_impossible for b in snap.buckets],
        )
        updated_buckets = [
            b.model_copy(update={"fair_probability": scaled[i], "intraday_adjusted_probability": scaled[i]})
            for i, b in enumerate(snap.buckets)
        ]
        return snap.model_copy(
            update={
                "buckets": updated_buckets,
                "probability_sum": sum(b.fair_probability for b in updated_buckets),
            }
        )


def save_residual_mos_model(
    mos_model: ResidualMosModel,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
) -> Path:
    """Persist trained ResidualMosModel LightGBM boosters + temperature metadata as JSON."""
    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = mos_model.model_version.replace("/", "_").replace("\\", "_")
    path = out_dir / f"{safe_name}.json"
    payload = {
        "model_family": "residual",
        "model_version": mos_model.model_version,
        "quantiles": list(mos_model.quantiles),
        "covered_stations": list(mos_model.covered_stations),
        "feature_columns": list(mos_model.feature_columns),
        "lgbm_params": mos_model.lgbm_params,
        "max_residual_delta_c": mos_model.max_residual_delta_c,
        "temperature": mos_model.temperature,
        "boosters": {str(q): booster.model_to_string() for q, booster in mos_model.boosters.items()},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _train_and_eval_residual_sync(
    samples: Sequence[MosDatasetSample],
    *,
    model_version: str,
    train_end_utc: datetime,
    val_start_utc: datetime,
    test_start_utc: datetime,
    quantiles: Sequence[float],
    covered_stations: Sequence[str],
    lgbm_params: dict[str, Any] | None,
    max_residual_delta_c: float,
) -> tuple[ResidualMosModel, ChronologicalBenchmarkReport, bool]:
    train_s, val_s, test_s = split_chronologically(
        samples,
        train_end=train_end_utc,
        val_start=val_start_utc,
        test_start=test_start_utc,
    )
    model = ResidualMosModel.fit(
        train_s,
        model_version=model_version,
        quantiles=quantiles,
        covered_stations=covered_stations,
        lgbm_params=lgbm_params,
        max_residual_delta_c=max_residual_delta_c,
        temperature=1.0,
    )
    # Fit 1-parameter Temperature Scaling on validation set multi-class Log-Loss
    from weather_quant.probability.calibration import build_synthetic_degree_buckets, resolve_bucket_index
    from weather_quant.probability.engine import round_half_away

    val_labeled = [s for s in val_s if s.target_temp_c is not None]
    if val_labeled:
        q_preds = model.predict_quantiles([s.features for s in val_labeled])
        probs_list: list[list[float]] = []
        true_indices: list[int] = []
        for sample, q_map in zip(val_labeled, q_preds):
            assert sample.target_temp_c is not None
            center = round_half_away(sample.baseline_mean_c)
            buckets = build_synthetic_degree_buckets(center)
            true_idx = resolve_bucket_index(sample.target_temp_c, buckets)
            extremes = model.quantiles_to_extremes(q_map)
            probs = map_extremes_c_to_probabilities(extremes, buckets, [False] * len(buckets))
            probs_list.append(probs)
            true_indices.append(true_idx)
        model.temperature = fit_temperature_scaling(probs_list, true_indices)

    report = evaluate_mos_vs_baseline(
        model,
        train_samples=train_s,
        val_samples=val_s,
        test_samples=test_s,
    )
    passed_gate = evaluate_station_overfit_gate(
        mos_val_brier=report.validation_mos.brier_score,
        base_val_brier=report.validation_baseline.brier_score,
        mos_test_brier=report.test_mos.brier_score,
        base_test_brier=report.test_baseline.brier_score,
    )
    return model, report, passed_gate


async def train_and_register_residual_mos(
    *,
    db: Database,
    samples: Sequence[MosDatasetSample],
    model_version: str,
    train_end_utc: datetime,
    val_start_utc: datetime,
    test_start_utc: datetime,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    covered_stations: Sequence[str] = TOP_LIQUIDITY_STATIONS,
    lgbm_params: dict[str, Any] | None = None,
    max_residual_delta_c: float = DEFAULT_MAX_RESIDUAL_DELTA_C,
    activate: bool = False,
    clock: Clock | None = None,
    models_dir: Path | str | None = DEFAULT_MODELS_DIR,
) -> tuple[ResidualMosModel, ModelVersionRecord, ChronologicalBenchmarkReport]:
    """Train ResidualMosModel + Temperature Scaling out-of-band and register Model Version."""
    async with db.conn.execute(
        "SELECT 1 FROM model_versions WHERE model_version = ?",
        (model_version,),
    ) as cursor:
        if await cursor.fetchone() is not None:
            raise ValueError(f"ModelVersion '{model_version}' already exists and is immutable.")

    mos_model, report, passed_gate = await asyncio.to_thread(
        _train_and_eval_residual_sync,
        list(samples),
        model_version=model_version,
        train_end_utc=train_end_utc,
        val_start_utc=val_start_utc,
        test_start_utc=test_start_utc,
        quantiles=quantiles,
        covered_stations=covered_stations,
        lgbm_params=lgbm_params,
        max_residual_delta_c=max_residual_delta_c,
    )

    artifact_path: Path | None = None
    if models_dir is not None:
        artifact_path = await asyncio.to_thread(save_residual_mos_model, mos_model, models_dir)

    now_utc = ensure_utc((clock or SystemClock()).now())
    config_payload: dict[str, Any] = {
        "model_family": "residual",
        "quantiles": list(mos_model.quantiles),
        "covered_stations": list(mos_model.covered_stations),
        "feature_columns": list(mos_model.feature_columns),
        "lgbm_params": mos_model.lgbm_params,
        "max_residual_delta_c": mos_model.max_residual_delta_c,
        "temperature": mos_model.temperature,
        "passed_overfit_gate": passed_gate,
        "artifact_path": str(artifact_path) if artifact_path is not None else None,
        "reliability_curve": [b.model_dump(mode="json") for b in report.test_mos.reliability_curve],
        "benchmark": report.model_dump(mode="json"),
    }
    record = ModelVersionRecord(
        model_version=model_version,
        model_type="residual_lightgbm_mos",
        description=(
            f"Residual LightGBM MOS over EMOS (tau={mos_model.temperature:.2f}, "
            f"gate={'PASS' if passed_gate else 'HOLD'}) covering {len(mos_model.covered_stations)} Stations"
        ),
        feature_schema_hash=mos_model.feature_schema_hash,
        train_start_utc=report.train_start_utc,
        train_end_utc=report.train_end_utc,
        brier_score=report.test_mos.brier_score,
        pinball_loss=report.test_mos.pinball_loss,
        config_json=config_payload,
        is_active=activate,
        created_at_utc=now_utc,
    )
    saved_record = await register_model_version(db, record, activate=activate)
    return mos_model, saved_record, report
