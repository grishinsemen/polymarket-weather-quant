"""Chronological calibration bench, Quantile LightGBM MOS trainer, and unified provider."""

from __future__ import annotations

import asyncio
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import TYPE_CHECKING, Any, Sequence
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
from pydantic import BaseModel, Field

from weather_quant.domain.clock import (
    Clock,
    SystemClock,
    TemporalIntegrityError,
    assert_before_decision,
    ensure_utc,
)
from weather_quant.domain.models import (
    Bucket,
    BucketFairProbability,
    EnsembleSnapshot,
    EventProbabilitySnapshot,
    MetarObservation,
    ModelVersionRecord,
    Polarity,
    Station,
    WeatherEvent,
)
from weather_quant.ml.dataset_builder import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_HASH,
    TOP_LIQUIDITY_STATIONS,
    MosDatasetSample,
    extract_features_from_ensemble,
    features_to_vector,
)
from weather_quant.probability.engine import (
    DEFAULT_MODEL_VERSION,
    DailyBiasObservation,
    EnsembleProbabilityEngine,
    map_extremes_to_probabilities,
    to_event_unit,
)

if TYPE_CHECKING:
    from weather_quant.storage.database import Database

DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 80,
    "learning_rate": 0.08,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": 42,
    "verbose": -1,
    "num_threads": 1,
}


class ReliabilityBin(BaseModel):
    """Single bin on a probability calibration reliability diagram."""

    bin_lower: float
    bin_upper: float
    count: int
    mean_predicted_prob: float
    empirical_frequency: float


class ModelBenchmarkMetrics(BaseModel):
    """Calibration & sharpness diagnostics on a chronological evaluation split."""

    brier_score: float
    pinball_loss: float
    reliability_curve: list[ReliabilityBin] = Field(default_factory=list)
    sample_count: int


class ChronologicalBenchmarkReport(BaseModel):
    """Side-by-side validation and test metrics comparing 3-layer baseline vs Quantile MOS."""

    model_version: str
    baseline_model_version: str = DEFAULT_MODEL_VERSION
    train_start_utc: datetime
    train_end_utc: datetime
    val_start_utc: datetime
    val_end_utc: datetime
    test_start_utc: datetime
    test_end_utc: datetime
    train_count: int
    val_count: int
    test_count: int
    validation_baseline: ModelBenchmarkMetrics
    validation_mos: ModelBenchmarkMetrics
    test_baseline: ModelBenchmarkMetrics
    test_mos: ModelBenchmarkMetrics


def _ensure_utc(ts: datetime) -> datetime:
    return ensure_utc(ts)


def split_chronologically(
    samples: Sequence[MosDatasetSample],
    *,
    train_end: datetime,
    val_start: datetime,
    test_start: datetime,
) -> tuple[list[MosDatasetSample], list[MosDatasetSample], list[MosDatasetSample]]:
    """Split samples strictly in time enforcing train_end < val_start < test_start."""
    t_train_end = _ensure_utc(train_end)
    t_val_start = _ensure_utc(val_start)
    t_test_start = _ensure_utc(test_start)

    if not (t_train_end < t_val_start < t_test_start):
        raise TemporalIntegrityError(
            "Chronological split requires strict ordering train_end < val_start < test_start; "
            f"got train_end={t_train_end.isoformat()}, val_start={t_val_start.isoformat()}, "
            f"test_start={t_test_start.isoformat()}"
        )

    prev_ts: datetime | None = None
    train: list[MosDatasetSample] = []
    val: list[MosDatasetSample] = []
    test: list[MosDatasetSample] = []

    for sample in samples:
        ts = _ensure_utc(sample.decision_time_utc)
        if prev_ts is not None and ts < prev_ts:
            raise TemporalIntegrityError(
                f"Samples must be chronologically sorted: {ts.isoformat()} < {prev_ts.isoformat()}"
            )
        prev_ts = ts

        if sample.target_temp_c is None:
            continue

        if ts < t_train_end:
            train.append(sample)
        elif t_val_start <= ts < t_test_start:
            val.append(sample)
        elif ts >= t_test_start:
            test.append(sample)

    if not train or not val or not test:
        raise ValueError(
            f"Chronological split produced empty partition: "
            f"train={len(train)}, val={len(val)}, test={len(test)}"
        )

    return train, val, test


def compute_brier_score(
    bucket_probs_list: Sequence[Sequence[float]],
    outcome_indices: Sequence[int],
) -> float:
    """Compute multi-class Brier score: mean_i sum_k (p_{i,k} - y_{i,k})^2."""
    if not bucket_probs_list or len(bucket_probs_list) != len(outcome_indices):
        raise ValueError("bucket_probs_list and outcome_indices must be non-empty and equal length.")
    total = 0.0
    for probs, true_idx in zip(bucket_probs_list, outcome_indices):
        for k, p in enumerate(probs):
            y = 1.0 if k == true_idx else 0.0
            total += (p - y) ** 2
    return total / len(bucket_probs_list)


def compute_pinball_loss(
    y_true: Sequence[float],
    quantile_preds: dict[float, Sequence[float]],
) -> float:
    """Compute mean quantile pinball loss across all quantiles and observations."""
    if not y_true or not quantile_preds:
        raise ValueError("y_true and quantile_preds must be non-empty.")
    losses: list[float] = []
    for q, preds in quantile_preds.items():
        if len(preds) != len(y_true):
            raise ValueError(f"Quantile {q} prediction length {len(preds)} != y_true length {len(y_true)}.")
        for actual, pred in zip(y_true, preds):
            diff = actual - pred
            losses.append(max(q * diff, (q - 1.0) * diff))
    return sum(losses) / len(losses)


def compute_reliability_curve(
    predicted_probs: Sequence[float],
    binary_outcomes: Sequence[int],
    n_bins: int = 10,
) -> list[ReliabilityBin]:
    """Bin predicted probabilities into [0, 1] reliability bins vs empirical hit frequency."""
    if len(predicted_probs) != len(binary_outcomes):
        raise ValueError("predicted_probs and binary_outcomes must have equal length.")
    if not predicted_probs or n_bins <= 0:
        return []

    bins: list[ReliabilityBin] = []
    width = 1.0 / n_bins
    for b in range(n_bins):
        low = b * width
        high = (b + 1) * width
        indices = [
            i
            for i, p in enumerate(predicted_probs)
            if (low <= p < high) or (b == n_bins - 1 and math.isclose(p, 1.0) and high == 1.0)
        ]
        if not indices:
            continue
        mean_p = sum(predicted_probs[i] for i in indices) / len(indices)
        emp = sum(binary_outcomes[i] for i in indices) / len(indices)
        bins.append(
            ReliabilityBin(
                bin_lower=round(low, 4),
                bin_upper=round(high, 4),
                count=len(indices),
                mean_predicted_prob=round(mean_p, 6),
                empirical_frequency=round(emp, 6),
            )
        )
    return bins


class QuantileMosModel:
    """Multi-quantile LightGBM MOS model calibrated per station microclimate."""

    def __init__(
        self,
        *,
        model_version: str,
        quantiles: Sequence[float],
        boosters: dict[float, lgb.Booster],
        covered_stations: Sequence[str] = TOP_LIQUIDITY_STATIONS,
        feature_columns: Sequence[str] = FEATURE_COLUMNS,
        lgbm_params: dict[str, Any] | None = None,
    ) -> None:
        self.model_version = model_version
        self.quantiles = tuple(sorted(float(q) for q in quantiles))
        self.boosters = boosters
        self.covered_stations = tuple(s.strip().upper() for s in covered_stations)
        self.feature_columns = tuple(feature_columns)
        self.feature_schema_hash = FEATURE_SCHEMA_HASH
        self.lgbm_params = dict(lgbm_params or DEFAULT_LGBM_PARAMS)

    @classmethod
    def fit(
        cls,
        train_samples: Sequence[MosDatasetSample],
        *,
        model_version: str,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        covered_stations: Sequence[str] = TOP_LIQUIDITY_STATIONS,
        lgbm_params: dict[str, Any] | None = None,
    ) -> QuantileMosModel:
        """Fit independent LightGBM quantile boosters on leak-free MosDatasetSamples."""
        valid = [s for s in train_samples if s.target_temp_c is not None]
        if not valid:
            raise ValueError("Cannot train QuantileMosModel on empty training dataset.")

        merged_params = dict(DEFAULT_LGBM_PARAMS)
        merged_params.setdefault("target_mode", "residual")
        if lgbm_params:
            merged_params.update(lgbm_params)

        x_train = np.asarray([features_to_vector(s.features) for s in valid], dtype=np.float64)
        if merged_params.get("target_mode") == "residual":
            y_train = np.asarray(
                [
                    float(s.target_temp_c) - float(s.features.get("ensemble_mean_c", s.baseline_mean_c))
                    for s in valid
                ],
                dtype=np.float64,
            )
        else:
            y_train = np.asarray([float(s.target_temp_c) for s in valid], dtype=np.float64)

        num_boost_round = int(merged_params.get("n_estimators", 80))
        core_params: dict[str, Any] = {
            "objective": "quantile",
            "boosting_type": "gbdt",
            "learning_rate": float(merged_params.get("learning_rate", 0.08)),
            "num_leaves": int(merged_params.get("num_leaves", 15)),
            "max_depth": int(merged_params.get("max_depth", 4)),
            "min_data_in_leaf": int(merged_params.get("min_child_samples", 5)),
            "feature_fraction": float(merged_params.get("colsample_bytree", 0.9)),
            "bagging_fraction": float(merged_params.get("subsample", 0.9)),
            "bagging_freq": 1 if float(merged_params.get("subsample", 0.9)) < 1.0 else 0,
            "seed": int(merged_params.get("random_state", 42)),
            "verbose": int(merged_params.get("verbose", merged_params.get("verbosity", -1))),
            "num_threads": int(merged_params.get("num_threads", merged_params.get("n_jobs", 1))),
        }

        sorted_q = tuple(sorted(float(q) for q in quantiles))
        boosters: dict[float, lgb.Booster] = {}
        for q in sorted_q:
            dataset = lgb.Dataset(
                x_train,
                label=y_train,
                feature_name=list(FEATURE_COLUMNS),
                free_raw_data=False,
            )
            params = dict(core_params, alpha=q)
            boosters[q] = lgb.train(params, dataset, num_boost_round=num_boost_round)

        return cls(
            model_version=model_version,
            quantiles=sorted_q,
            boosters=boosters,
            covered_stations=covered_stations,
            feature_columns=FEATURE_COLUMNS,
            lgbm_params=merged_params,
        )

    def predict_quantiles(
        self,
        feature_dicts: Sequence[dict[str, float]],
    ) -> list[dict[float, float]]:
        """Predict non-crossing temperature quantiles for a batch of feature dicts."""
        if not feature_dicts:
            return []
        x_mat = np.asarray([features_to_vector(f) for f in feature_dicts], dtype=np.float64)
        raw_by_q = {q: self.boosters[q].predict(x_mat) for q in self.quantiles}

        is_residual = self.lgbm_params.get("target_mode") == "residual"
        results: list[dict[float, float]] = []
        for idx, feat in enumerate(feature_dicts):
            base = float(feat.get("ensemble_mean_c", 0.0)) if is_residual else 0.0
            # Sort across quantiles to eliminate quantile crossing
            sorted_vals = sorted(base + float(raw_by_q[q][idx]) for q in self.quantiles)
            results.append({q: val for q, val in zip(self.quantiles, sorted_vals)})
        return results

    @staticmethod
    def quantiles_to_extremes(
        quantile_map: dict[float, float],
        n_samples: int = 101,
    ) -> list[float]:
        """Interpolate predicted quantiles into a continuous temperature sample grid."""
        qs = sorted(quantile_map.keys())
        vals = [quantile_map[q] for q in qs]
        if len(qs) == 1:
            return [vals[0]] * n_samples

        # Mild linear tail slopes so 0.01..0.99 capture tail mass realistically
        low_slope = (vals[1] - vals[0]) / max(qs[1] - qs[0], 1e-6)
        high_slope = (vals[-1] - vals[-2]) / max(qs[-1] - qs[-2], 1e-6)
        low_slope = max(low_slope, 0.2)
        high_slope = max(high_slope, 0.2)

        probs = np.linspace(0.01, 0.99, n_samples)
        samples: list[float] = []
        for p in probs:
            pf = float(p)
            if pf <= qs[0]:
                samples.append(vals[0] - low_slope * (qs[0] - pf))
            elif pf >= qs[-1]:
                samples.append(vals[-1] + high_slope * (pf - qs[-1]))
            else:
                samples.append(float(np.interp(pf, qs, vals)))
        return samples

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
        """Produce calibrated Bucket Fair Probabilities with intraday Routine METAR zeroing."""
        active_clock: Clock = clock or SystemClock()
        decision_time = _ensure_utc(active_clock.now())
        station_tz = ZoneInfo(station.timezone)

        prior_metars: list[MetarObservation] = []
        target_icao = station.icao.strip().upper()
        for obs in intraday_metars:
            if (
                not obs.is_routine
                or obs.raw_report.strip().upper().startswith("SPECI")
                or obs.station_icao.strip().upper() != target_icao
            ):
                continue
            if strict_temporal:
                assert_before_decision(obs.observed_at_utc, decision_time, label="intraday_metar")
            elif _ensure_utc(obs.observed_at_utc) >= decision_time:
                continue
            prior_metars.append(obs)
        today_metars = [
            obs
            for obs in prior_metars
            if _ensure_utc(obs.observed_at_utc).astimezone(station_tz).date() == event.local_date
        ]

        features = extract_features_from_ensemble(
            station=station,
            local_date=event.local_date,
            polarity=event.polarity,
            ensemble=ensemble,
            decision_time_utc=decision_time,
            recent_bias_obs=recent_bias_obs,
            prior_observations=prior_metars,
        )
        q_pred = self.predict_quantiles([features])[0]
        mos_extremes_c = self.quantiles_to_extremes(q_pred)

        impossible_mask = [False] * len(event.buckets)
        if today_metars:
            today_metars.sort(key=lambda o: _ensure_utc(o.observed_at_utc))
            obs_temps_c = [
                EnsembleProbabilityEngine.obs_settlement_temp_c(obs, event.remark_tenths_enabled)
                for obs in today_metars
            ]
            realized_extreme_c = (
                max(obs_temps_c) if event.polarity == Polarity.HIGHEST else min(obs_temps_c)
            )
            realized_extreme_unit = to_event_unit(realized_extreme_c, event.unit)
            impossible_mask = EnsembleProbabilityEngine.compute_impossible_mask(
                event.buckets,
                event.polarity,
                realized_extreme_unit,
            )

            # Layer 3 intraday conditioning: calibrate hourly trajectories to MOS premarket
            # mean/spread, then apply decaying live METAR residual drift over remaining hours.
            day_indices = [
                idx
                for idx, ts in enumerate(ensemble.hourly_timestamps_utc)
                if _ensure_utc(ts).astimezone(station_tz).date() == event.local_date
            ]
            day_timestamps = [_ensure_utc(ensemble.hourly_timestamps_utc[idx]) for idx in day_indices]
            day_trajectories = [
                [float(traj[idx]) for idx in day_indices]
                for traj in ensemble.member_temps_c
                if day_indices and len(traj) > max(day_indices)
            ]
            if day_trajectories:
                raw_extremes_c = [
                    max(traj) if event.polarity == Polarity.HIGHEST else min(traj)
                    for traj in day_trajectories
                ]
                raw_mean = float(np.mean(raw_extremes_c))
                raw_std = float(np.std(raw_extremes_c))
                mos_mean = float(np.mean(mos_extremes_c))
                mos_std = float(np.std(mos_extremes_c))
                mos_bias_c = mos_mean - raw_mean
                mos_scale = (
                    max(0.5, min(3.0, mos_std / raw_std))
                    if raw_std > 1e-6 and mos_std > 1e-6
                    else 1.0
                )

                prob_engine = EnsembleProbabilityEngine()
                mos_trajectories = prob_engine.calibrate_day_trajectories(
                    day_trajectories,
                    bias_c=mos_bias_c,
                    scale=mos_scale,
                )
                latest_obs_utc = _ensure_utc(today_metars[-1].observed_at_utc)
                latest_temp_c = obs_temps_c[-1]
                shifted_trajectories, _, _ = prob_engine.compute_intraday_shifted_trajectories(
                    day_timestamps,
                    mos_trajectories,
                    latest_obs_utc,
                    latest_temp_c,
                )
                future_indices = [
                    idx for idx, ts in enumerate(day_timestamps) if ts > latest_obs_utc
                ]
                intraday_extremes_c: list[float] = []
                for shifted_traj in shifted_trajectories:
                    future_vals = [shifted_traj[idx] for idx in future_indices]
                    if event.polarity == Polarity.HIGHEST:
                        intraday_extremes_c.append(max([realized_extreme_c, *future_vals]))
                    else:
                        intraday_extremes_c.append(min([realized_extreme_c, *future_vals]))
                mos_extremes_c = intraday_extremes_c
            else:
                if event.polarity == Polarity.HIGHEST:
                    mos_extremes_c = [max(x, realized_extreme_c) for x in mos_extremes_c]
                else:
                    mos_extremes_c = [min(x, realized_extreme_c) for x in mos_extremes_c]

        mos_extremes_unit = [to_event_unit(x, event.unit) for x in mos_extremes_c]
        mos_probs = map_extremes_to_probabilities(
            mos_extremes_unit,
            event.buckets,
            impossible_mask,
            smoothing_alpha=0.001,
        )

        baseline_by_id = (
            {b.bucket_id: b for b in baseline_snapshot.buckets}
            if baseline_snapshot is not None
            else {}
        )
        buckets = [
            BucketFairProbability(
                event_id=event.event_id,
                bucket_id=bucket.bucket_id,
                label=bucket.label,
                fair_probability=mos_probs[i],
                raw_ensemble_probability=(
                    baseline_by_id[bucket.bucket_id].raw_ensemble_probability
                    if bucket.bucket_id in baseline_by_id
                    else None
                ),
                bias_adjusted_probability=(
                    baseline_by_id[bucket.bucket_id].bias_adjusted_probability
                    if bucket.bucket_id in baseline_by_id
                    else None
                ),
                intraday_adjusted_probability=mos_probs[i],
                is_impossible=impossible_mask[i],
                model_version=self.model_version,
            )
            for i, bucket in enumerate(event.buckets)
        ]

        return EventProbabilitySnapshot(
            event_id=event.event_id,
            station_icao=station.icao,
            model_version=self.model_version,
            computed_at_utc=decision_time,
            buckets=buckets,
            probability_sum=sum(b.fair_probability for b in buckets),
        )


def _eval_buckets_for_sample(target_c: float, baseline_mean_c: float) -> tuple[list[Bucket], int]:
    """Create a deterministic 1C integer Bucket partition covering both baseline and target."""
    center = int(round((target_c + baseline_mean_c) / 2.0))
    low = min(int(round(target_c)), center) - 4
    high = max(int(round(target_c)), center) + 4
    buckets: list[Bucket] = [
        Bucket(
            bucket_id=f"eval-{low}-below",
            market_id="eval",
            token_id="eval",
            label=f"{low}°C or below",
            lower_bound=None,
            upper_bound=float(low),
            unit="C",
            outcome_index=0,
        )
    ]
    idx = 1
    for deg in range(low + 1, high):
        buckets.append(
            Bucket(
                bucket_id=f"eval-{deg}",
                market_id="eval",
                token_id="eval",
                label=f"{deg}°C",
                lower_bound=float(deg),
                upper_bound=float(deg),
                unit="C",
                outcome_index=idx,
            )
        )
        idx += 1
    buckets.append(
        Bucket(
            bucket_id=f"eval-{high}-above",
            market_id="eval",
            token_id="eval",
            label=f"{high}°C or higher",
            lower_bound=float(high),
            upper_bound=None,
            unit="C",
            outcome_index=idx,
        )
    )

    rounded_target = int(math.floor(target_c + 0.5)) if target_c >= 0 else -int(math.floor(-target_c + 0.5))
    if rounded_target <= low:
        true_idx = 0
    elif rounded_target >= high:
        true_idx = len(buckets) - 1
    else:
        true_idx = rounded_target - low
    return buckets, true_idx


def _evaluate_split(
    mos_model: QuantileMosModel,
    samples: Sequence[MosDatasetSample],
) -> tuple[ModelBenchmarkMetrics, ModelBenchmarkMetrics]:
    """Evaluate 3-layer baseline vs Quantile MOS on a chronological sample slice."""
    y_true = [float(s.target_temp_c) for s in samples if s.target_temp_c is not None]
    eval_samples = [s for s in samples if s.target_temp_c is not None]
    q_preds = mos_model.predict_quantiles([s.features for s in eval_samples])

    mos_q_by_level: dict[float, list[float]] = {q: [] for q in mos_model.quantiles}
    base_q_by_level: dict[float, list[float]] = {q: [] for q in mos_model.quantiles}

    mos_bucket_probs: list[list[float]] = []
    base_bucket_probs: list[list[float]] = []
    outcome_indices: list[int] = []
    mos_flat_probs: list[float] = []
    base_flat_probs: list[float] = []
    flat_outcomes: list[int] = []

    grid_probs = np.linspace(0.01, 0.99, 101)
    for sample, q_map in zip(eval_samples, q_preds):
        assert sample.target_temp_c is not None
        buckets, true_idx = _eval_buckets_for_sample(sample.target_temp_c, sample.baseline_mean_c)
        outcome_indices.append(true_idx)
        no_impossible = [False] * len(buckets)

        mos_extremes = mos_model.quantiles_to_extremes(q_map)
        m_probs = map_extremes_to_probabilities(mos_extremes, buckets, no_impossible, smoothing_alpha=0.001)
        mos_bucket_probs.append(m_probs)

        dist = NormalDist(mu=sample.baseline_mean_c, sigma=max(sample.baseline_std_c, 0.35))
        base_extremes = [dist.inv_cdf(float(p)) for p in grid_probs]
        b_probs = map_extremes_to_probabilities(base_extremes, buckets, no_impossible, smoothing_alpha=0.001)
        base_bucket_probs.append(b_probs)

        for q in mos_model.quantiles:
            mos_q_by_level[q].append(q_map[q])
            base_q_by_level[q].append(dist.inv_cdf(q))

        for k in range(len(buckets)):
            mos_flat_probs.append(m_probs[k])
            base_flat_probs.append(b_probs[k])
            flat_outcomes.append(1 if k == true_idx else 0)

    baseline_metrics = ModelBenchmarkMetrics(
        brier_score=compute_brier_score(base_bucket_probs, outcome_indices),
        pinball_loss=compute_pinball_loss(y_true, base_q_by_level),
        reliability_curve=compute_reliability_curve(base_flat_probs, flat_outcomes),
        sample_count=len(eval_samples),
    )
    mos_metrics = ModelBenchmarkMetrics(
        brier_score=compute_brier_score(mos_bucket_probs, outcome_indices),
        pinball_loss=compute_pinball_loss(y_true, mos_q_by_level),
        reliability_curve=compute_reliability_curve(mos_flat_probs, flat_outcomes),
        sample_count=len(eval_samples),
    )
    return baseline_metrics, mos_metrics


def evaluate_mos_vs_baseline(
    mos_model: QuantileMosModel,
    *,
    train_samples: Sequence[MosDatasetSample],
    val_samples: Sequence[MosDatasetSample],
    test_samples: Sequence[MosDatasetSample],
) -> ChronologicalBenchmarkReport:
    """Compute chronological validation & backtest diagnostics comparing baseline vs Quantile MOS."""
    val_base, val_mos = _evaluate_split(mos_model, val_samples)
    test_base, test_mos = _evaluate_split(mos_model, test_samples)

    return ChronologicalBenchmarkReport(
        model_version=mos_model.model_version,
        baseline_model_version=DEFAULT_MODEL_VERSION,
        train_start_utc=min(_ensure_utc(s.decision_time_utc) for s in train_samples),
        train_end_utc=max(_ensure_utc(s.decision_time_utc) for s in train_samples),
        val_start_utc=min(_ensure_utc(s.decision_time_utc) for s in val_samples),
        val_end_utc=max(_ensure_utc(s.decision_time_utc) for s in val_samples),
        test_start_utc=min(_ensure_utc(s.decision_time_utc) for s in test_samples),
        test_end_utc=max(_ensure_utc(s.decision_time_utc) for s in test_samples),
        train_count=len(train_samples),
        val_count=len(val_samples),
        test_count=len(test_samples),
        validation_baseline=val_base,
        validation_mos=val_mos,
        test_baseline=test_base,
        test_mos=test_mos,
    )


def _train_and_eval_sync(
    samples: Sequence[MosDatasetSample],
    *,
    model_version: str,
    train_end_utc: datetime,
    val_start_utc: datetime,
    test_start_utc: datetime,
    quantiles: Sequence[float],
    covered_stations: Sequence[str],
    lgbm_params: dict[str, Any] | None,
) -> tuple[QuantileMosModel, ChronologicalBenchmarkReport]:
    train_s, val_s, test_s = split_chronologically(
        samples,
        train_end=train_end_utc,
        val_start=val_start_utc,
        test_start=test_start_utc,
    )
    mos_model = QuantileMosModel.fit(
        train_s,
        model_version=model_version,
        quantiles=quantiles,
        covered_stations=covered_stations,
        lgbm_params=lgbm_params,
    )
    report = evaluate_mos_vs_baseline(
        mos_model,
        train_samples=train_s,
        val_samples=val_s,
        test_samples=test_s,
    )
    return mos_model, report


async def register_model_version(
    db: Database,
    record: ModelVersionRecord,
    *,
    activate: bool = False,
) -> ModelVersionRecord:
    """Register an immutable ModelVersionRecord in SQLite model_versions."""
    async with db.conn.execute(
        "SELECT 1 FROM model_versions WHERE model_version = ?",
        (record.model_version,),
    ) as cursor:
        existing = await cursor.fetchone()
    if existing is not None:
        raise ValueError(f"ModelVersion '{record.model_version}' already exists and is immutable.")

    if activate:
        await db.conn.execute("UPDATE model_versions SET is_active = 0")

    stored = record.model_copy(update={"is_active": activate})
    await db.conn.execute(
        """
        INSERT INTO model_versions (
            model_version, model_type, description, feature_schema_hash,
            train_start_utc, train_end_utc, brier_score, pinball_loss,
            config_json, is_active, created_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            stored.model_version,
            stored.model_type,
            stored.description,
            stored.feature_schema_hash,
            stored.train_start_utc.isoformat() if stored.train_start_utc else None,
            stored.train_end_utc.isoformat() if stored.train_end_utc else None,
            stored.brier_score,
            stored.pinball_loss,
            json.dumps(stored.config_json),
            int(stored.is_active),
            stored.created_at_utc.isoformat(),
        ),
    )
    await db.conn.commit()
    return stored


DEFAULT_MODELS_DIR = Path("data/models")


def save_mos_model(
    mos_model: QuantileMosModel,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
) -> Path:
    """Persist trained QuantileMosModel LightGBM boosters as JSON artifact."""
    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = mos_model.model_version.replace("/", "_").replace("\\", "_")
    path = out_dir / f"{safe_name}.json"
    payload = {
        "model_version": mos_model.model_version,
        "quantiles": list(mos_model.quantiles),
        "covered_stations": list(mos_model.covered_stations),
        "feature_columns": list(mos_model.feature_columns),
        "lgbm_params": mos_model.lgbm_params,
        "boosters": {str(q): booster.model_to_string() for q, booster in mos_model.boosters.items()},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_mos_model(
    model_version: str,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
) -> QuantileMosModel | None:
    """Load persisted QuantileMosModel from disk if present."""
    safe_name = model_version.replace("/", "_").replace("\\", "_")
    path = Path(models_dir) / f"{safe_name}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    boosters = {
        float(q): lgb.Booster(model_str=model_str)
        for q, model_str in (payload.get("boosters") or {}).items()
    }
    if not boosters:
        return None
    return QuantileMosModel(
        model_version=str(payload["model_version"]),
        quantiles=[float(q) for q in payload["quantiles"]],
        boosters=boosters,
        covered_stations=payload.get("covered_stations") or TOP_LIQUIDITY_STATIONS,
        feature_columns=payload.get("feature_columns") or FEATURE_COLUMNS,
        lgbm_params=payload.get("lgbm_params"),
    )


async def load_mos_provider_for_version(
    db: Database,
    model_version: str,
    *,
    existing_provider: MosProbabilityProvider | None = None,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
) -> MosProbabilityProvider | None:
    """Resolve a MosProbabilityProvider for model_version from memory or persisted boosters."""
    if existing_provider is not None and existing_provider.mos_model is not None:
        if existing_provider.mos_model.model_version == model_version:
            return existing_provider
    versions = await db.list_model_versions()
    rec = next((v for v in versions if v.model_version == model_version), None)
    if rec is None or ("mos" not in rec.model_type.lower() and "mos" not in rec.model_version.lower()):
        return None
    artifact_path = (rec.config_json or {}).get("artifact_path")
    if artifact_path and Path(artifact_path).exists():
        payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        boosters = {float(q): lgb.Booster(model_str=s) for q, s in (payload.get("boosters") or {}).items()}
        loaded = QuantileMosModel(
            model_version=str(payload["model_version"]),
            quantiles=[float(q) for q in payload["quantiles"]],
            boosters=boosters,
            covered_stations=payload.get("covered_stations") or TOP_LIQUIDITY_STATIONS,
            feature_columns=payload.get("feature_columns") or FEATURE_COLUMNS,
            lgbm_params=payload.get("lgbm_params"),
        ) if boosters else None
    else:
        loaded = await asyncio.to_thread(load_mos_model, model_version, models_dir)
    if loaded is None:
        return None
    return MosProbabilityProvider(mos_model=loaded, covered_stations=loaded.covered_stations)


async def train_and_register_mos(
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
    activate: bool = False,
    clock: Clock | None = None,
    models_dir: Path | str | None = DEFAULT_MODELS_DIR,
) -> tuple[QuantileMosModel, ModelVersionRecord, ChronologicalBenchmarkReport]:
    """Run out-of-band Quantile LightGBM training via asyncio.to_thread and register Model Version."""
    # Pre-check immutability before running training work
    async with db.conn.execute(
        "SELECT 1 FROM model_versions WHERE model_version = ?",
        (model_version,),
    ) as cursor:
        if await cursor.fetchone() is not None:
            raise ValueError(f"ModelVersion '{model_version}' already exists and is immutable.")

    mos_model, report = await asyncio.to_thread(
        _train_and_eval_sync,
        list(samples),
        model_version=model_version,
        train_end_utc=train_end_utc,
        val_start_utc=val_start_utc,
        test_start_utc=test_start_utc,
        quantiles=quantiles,
        covered_stations=covered_stations,
        lgbm_params=lgbm_params,
    )

    artifact_path: Path | None = None
    if models_dir is not None:
        artifact_path = await asyncio.to_thread(save_mos_model, mos_model, models_dir)

    now_utc = _ensure_utc((clock or SystemClock()).now())
    config_payload: dict[str, Any] = {
        "quantiles": list(mos_model.quantiles),
        "covered_stations": list(mos_model.covered_stations),
        "feature_columns": list(mos_model.feature_columns),
        "lgbm_params": mos_model.lgbm_params,
        "artifact_path": str(artifact_path) if artifact_path is not None else None,
        "reliability_curve": [b.model_dump(mode="json") for b in report.test_mos.reliability_curve],
        "benchmark": report.model_dump(mode="json"),
    }
    record = ModelVersionRecord(
        model_version=model_version,
        model_type="quantile_lightgbm_mos",
        description=(
            f"Quantile LightGBM MOS ({len(mos_model.quantiles)} quantiles) "
            f"covering {len(mos_model.covered_stations)} top Stations"
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


class MosProbabilityProvider:
    """Unified Fair Probability provider: Quantile MOS for top Stations, 3-layer baseline fallback otherwise."""

    def __init__(
        self,
        *,
        mos_model: QuantileMosModel | None = None,
        baseline_engine: EnsembleProbabilityEngine | None = None,
        covered_stations: Sequence[str] = TOP_LIQUIDITY_STATIONS,
    ) -> None:
        self.mos_model = mos_model
        self.baseline_engine = baseline_engine or EnsembleProbabilityEngine()
        self.covered_stations = {s.strip().upper() for s in covered_stations}

    def set_mos_model(self, mos_model: QuantileMosModel) -> None:
        """Attach a newly trained QuantileMosModel for live covered-station inference."""
        self.mos_model = mos_model
        self.covered_stations = set(mos_model.covered_stations)

    def is_station_covered(self, station_icao: str) -> bool:
        """Return True if a trained MOS model is available and covers this Station."""
        return (
            self.mos_model is not None
            and station_icao.strip().upper() in self.covered_stations
        )

    def compute_probabilities(
        self,
        station: Station,
        event: WeatherEvent,
        ensemble: EnsembleSnapshot,
        recent_bias_obs: Sequence[DailyBiasObservation] = (),
        intraday_metars: Sequence[MetarObservation] = (),
        clock: Clock | None = None,
        strict_temporal: bool = False,
    ) -> EventProbabilitySnapshot:
        """Compute Fair Probabilities using Quantile MOS for covered Stations or 3-layer baseline otherwise."""
        baseline_snap = self.baseline_engine.compute_probabilities(
            station=station,
            event=event,
            ensemble=ensemble,
            recent_bias_obs=recent_bias_obs,
            intraday_metars=intraday_metars,
            clock=clock,
            model_version=DEFAULT_MODEL_VERSION,
            strict_temporal=strict_temporal,
        )
        if not self.is_station_covered(station.icao):
            return baseline_snap

        assert self.mos_model is not None
        return self.mos_model.predict_event_probabilities(
            station=station,
            event=event,
            ensemble=ensemble,
            recent_bias_obs=recent_bias_obs,
            intraday_metars=intraday_metars,
            clock=clock,
            strict_temporal=strict_temporal,
            baseline_snapshot=baseline_snap,
        )
