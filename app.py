"""
app.py — BridgingBipolar Clinical API
======================================
Backend Flask synchronisé avec le notebook BridgingBipolar.

PROTOCOLE CLINIQUE :
  ── Baseline (3 sessions · 1 répétition chacune) ──────────────────────────
  Session 1 : écriture lettre l + YMRS complet + MADRS complet
  Session 2 : écriture lettre l + ASRM court   + PHQ-9 court
  Session 3 : écriture lettre l + ASRM court   + PHQ-9 court → FIT du modèle

  ── Monitoring (1 répétition · zéro questionnaire par défaut) ─────────────
  Chaque session : écriture → score z_robust calculé
  Déviation J1   (consecutive=1) → revenir le lendemain, pas de questionnaire
  Déviation J2   (consecutive≥2 + QC≥0.65) → ALERTE CONFIRMÉE :
                   - DRE activé → direction manic_risk / depressive_risk
                   - YMRS complet + MADRS complet requis
                   - Classification clinique possible

Routes exposées (attendues par capture_bridgingbipolar.html) :
  POST /onboarding/<patient_id>/session   → session baseline (1 rép)
  POST /score                             → scoring monitoring (1 rép)
  POST /questionnaires/score              → soumission scores YMRS/MADRS/ASRM/PHQ-9
  GET  /questionnaire_catalog             → catalogue questionnaires (fr/en/ar)
  GET  /protocol_config                   → constantes QC du notebook
  GET  /dashboard/alerts                  → alertes clinicien 7 derniers jours
  GET  /admin/locks                       → état des verrous patients

Constantes synchronisées avec Section 1 du notebook :
  MIN_POINTS_PER_REP_STRICT = 80
  MIN_PATH_PER_REP          = 400.0
  LOOPS_WARMUP              = 0
  TARGET_LOOPS_PER_REP      = 25
  MIN_VALID_LOOPS_PER_REP   = 18
  QC_GATE                   = 0.65
  GLOBAL_SEED               = 42
  CALIBRATED_THRESHOLD_MARGIN = 1.15
  CALIBRATED_THRESHOLD_FLOOR  = 3.0

Usage :
  pip install flask flask-cors numpy pandas scikit-learn scipy joblib
  python app.py
  → http://localhost:5000
"""

import json
import math
import hashlib
import random
import threading
import logging
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from scipy import stats
import joblib

# ── Optional PyTorch (AE model) ──────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = nn = None
    TORCH_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONSTANTES QC (source de vérité — synchronisée avec notebook §1)
# ═══════════════════════════════════════════════════════════════════════════════

GLOBAL_SEED                  = 42
MIN_POINTS_PER_REP_STRICT    = 80
MIN_PATH_PER_REP             = 400.0
LOOPS_WARMUP                 = 0
TARGET_LOOPS_PER_REP         = 25
MIN_VALID_LOOPS_PER_REP      = 18
QC_GATE                      = 0.65       # §9 — seuil confiance QC pour alerte
CALIBRATED_THRESHOLD_MARGIN  = 1.15       # §9 — CalibratedAnomalyDetector
CALIBRATED_THRESHOLD_FLOOR   = 3.0        # §9 — plancher z-sigma minimal
PERSISTENCE_REQUIRED         = 2          # §9 — alertes consécutives requises
AUGMENT_RATIO                = 5          # §5 — ratio augmentation baseline

# ── Protocole baseline ────────────────────────────────────────────────────────
# 3 sessions · 1 répétition chacune
# Session 1 : écriture + YMRS complet + MADRS complet
# Session 2 : écriture + ASRM court + PHQ-9 court
# Session 3 : écriture + ASRM court + PHQ-9 court → fit après cette session
# Monitoring : écriture seule · déviation → revenir le lendemain
# 2 déviations consécutives → DRE + YMRS+MADRS complet
BASELINE_REQUIRED_SESSIONS   = 3

# Proxy conditions → label binaire (notebook §6)
CONDITION_TO_PROXY = {
    'stable': 0, 'noisy': 0, 'bad_capture': 0,
    'fast': 1, 'slow': 1, 'fatigue': 1, 'mixed': 1,
}

np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

# ── Artefacts sauvegardés par le notebook §12 ─────────────────────────────────
ARTIFACT_ROOT = Path('artifacts_mvp')
ALERT_LOG     = Path('alert_log.json')

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('bridgingbipolar')

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FEATURE EXTRACTION (23 features, identique notebook §3)
# ═══════════════════════════════════════════════════════════════════════════════

def safe_std(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    return float(np.std(a, ddof=1)) if a.size > 1 else 0.0

def safe_cv(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    if a.size < 2:
        return 0.0
    m = np.mean(a)
    return float(np.std(a, ddof=1) / (abs(m) + 1e-12))

def robust_entropy(a: np.ndarray, bins: int = 16) -> float:
    a = np.asarray(a, dtype=float)
    if a.size < 3:
        return 0.0
    hist, _ = np.histogram(a, bins=bins, density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return 0.0
    p = hist / hist.sum()
    return float(-(p * np.log2(p + 1e-12)).sum())

def sample_entropy(ts: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
    ts = np.asarray(ts, dtype=float)
    n = ts.size
    if n < (m + 2):
        return 0.0
    sd = np.std(ts)
    if sd < 1e-12:
        return 0.0
    r = r_ratio * sd

    def _phi(mm):
        cnt = total = 0
        for i in range(n - mm):
            tpl = ts[i:i + mm]
            for j in range(i + 1, n - mm + 1):
                if np.max(np.abs(tpl - ts[j:j + mm])) <= r:
                    cnt += 1
                total += 1
        return cnt / max(total, 1)

    a, b = _phi(m + 1), _phi(m)
    if a <= 0 or b <= 0:
        return 0.0
    return float(-np.log(a / b + 1e-12))

def lz_complexity_binary(bits) -> float:
    s = ''.join('1' if b else '0' for b in bits)
    if not s:
        return 0.0
    i, k, n, c = 0, 1, len(s), 1
    while True:
        if i + k > n:
            c += 1
            break
        sub = s[i:i + k]
        if any(s[j:j + k] == sub for j in range(i)):
            k += 1
        else:
            c += 1
            i += k
            if i >= n:
                break
            k = 1
    return float(c)

def estimate_loop_heights(y: np.ndarray, n_segments: int = TARGET_LOOPS_PER_REP) -> np.ndarray:
    """Extrait la hauteur (bbox) de chaque boucle estimée — notebook §3."""
    if y.size < 10:
        return np.array([])
    seg_size = max(1, len(y) // n_segments)
    heights = []
    for i in range(0, len(y) - seg_size, seg_size):
        seg = y[i:i + seg_size]
        heights.append(float(seg.max() - seg.min()))
    return np.array(heights)

def estimate_loop_count(points: list) -> int:
    """Compte les boucles via zero-crossing sur y lissé (identique notebook §9)."""
    if len(points) < 12:
        return 0
    ys = np.array([p['y'] for p in points], dtype=float)
    kernel = np.ones(5) / 5.0
    smooth = np.convolve(ys, kernel, mode='same')
    dy = np.diff(smooth)
    dy = np.where(np.abs(dy) < 1e-6, 0.0, dy)
    sign = np.sign(dy)
    sign = sign[sign != 0]
    if sign.size < 4:
        return 0
    turning = int(np.sum(sign[1:] * sign[:-1] < 0))
    return max(0, turning // 2)

def extract_repetition_features(points: list) -> dict:
    """
    Extrait les 23 features cinématiques d'une répétition (identique notebook §3).
    Lève ValueError si trop peu de points.
    """
    if len(points) < MIN_POINTS_PER_REP_STRICT:
        raise ValueError(f'Répétition trop courte: {len(points)} points < {MIN_POINTS_PER_REP_STRICT}')

    x  = np.array([p['x'] for p in points], dtype=float)
    y  = np.array([p['y'] for p in points], dtype=float)
    t  = np.array([p['t'] for p in points], dtype=float)
    pr = np.array([p.get('pressure', 0.5) for p in points], dtype=float)

    dt   = np.diff(t)
    dt   = dt[dt > 0]
    dx, dy_ = np.diff(x), np.diff(y)
    dist = np.sqrt(dx ** 2 + dy_ ** 2)
    vel  = dist / (dt + 1e-9) if len(dt) > 0 else np.array([0.0])
    acc  = np.diff(vel) / (dt[1:] + 1e-9) if len(vel) > 1 else np.array([0.0])

    path     = float(dist.sum()) if len(dist) > 0 else 1.0
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    pause_dt = dt[dt > np.percentile(dt, 90)] if len(dt) > 0 else np.array([0.0])

    # size_decay_slope — notebook v8 fix : pente de la hauteur de boucle
    loop_heights = estimate_loop_heights(y, n_segments=TARGET_LOOPS_PER_REP)
    if loop_heights.size >= 6:
        slope = float(np.polyfit(np.arange(loop_heights.size), loop_heights, 1)[0])
    else:
        slope = 0.0

    # Pression binaire pour LZ complexity
    pr_bin = (pr > float(np.median(pr))).astype(int)

    feats = {
        # Temporelles
        'duration_s':          duration,
        'mean_dt':             float(np.mean(dt)) if len(dt) > 0 else 0.0,
        'std_dt':              safe_std(dt),
        'pause_ratio':         float(pause_dt.sum() / (duration + 1e-9)),
        # Cinématiques
        'mean_velocity':       float(np.mean(vel)),
        'std_velocity':        safe_std(vel),
        'p95_velocity':        float(np.percentile(vel, 95)) if len(vel) > 0 else 0.0,
        'mean_acceleration':   float(np.mean(np.abs(acc))),
        # Géométriques
        'path_length':         path,
        # Tortuosité v8 : path / bbox_height (évite div/0 sur trajectoires circulaires)
        'tortuosity':          float(path / max(float(y.max() - y.min()), 5.0)),
        'bbox_width':          float(x.max() - x.min()),
        'bbox_height':         float(y.max() - y.min()),
        'max_extent':          float(np.sqrt((x.max() - x.min())**2 + (y.max() - y.min())**2)),
        'size_decay_slope':    slope,
        # Pression
        'mean_pressure':       float(np.mean(pr)),
        'std_pressure':        safe_std(pr),
        'pressure_entropy':    robust_entropy(pr),
        # Complexité non-linéaire
        'velocity_sample_entropy':   sample_entropy(vel),
        'pressure_lempel_ziv':       lz_complexity_binary(pr_bin),
        'pressure_fractal_dim':      float(math.log(len(pr)) / (math.log(len(pr)) + math.log(1.0 / (safe_std(pr) + 1e-9)) + 1e-9)),
        # Variabilité inter-répétitions (calculée en bundle, défaut 0.0)
        'velocity_cv':   0.0,
        'duration_cv':   0.0,
        'path_cv':       0.0,
    }
    return feats

def extract_session_bundle(repetitions: list) -> dict:
    """
    Extrait les features de toutes les répétitions et calcule la variabilité
    inter-répétitions (velocity_cv, duration_cv, path_cv) — notebook §3.
    """
    rep_feats = []
    for rep in repetitions:
        try:
            rep_feats.append(extract_repetition_features(rep))
        except ValueError as e:
            log.warning(f'Répétition ignorée: {e}')

    if not rep_feats:
        raise ValueError('Aucune répétition valide dans la session')

    # Variabilité inter-répétitions
    if len(rep_feats) >= 2:
        vels  = [f['mean_velocity'] for f in rep_feats]
        durs  = [f['duration_s']    for f in rep_feats]
        paths = [f['path_length']   for f in rep_feats]
        vel_cv  = safe_cv(np.array(vels))
        dur_cv  = safe_cv(np.array(durs))
        path_cv = safe_cv(np.array(paths))
        for f in rep_feats:
            f['velocity_cv'] = vel_cv
            f['duration_cv'] = dur_cv
            f['path_cv']     = path_cv

    return {'repetition_features': rep_feats, 'n_reps': len(rep_feats)}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — QC CONFIDENCE (identique notebook §9)
# ═══════════════════════════════════════════════════════════════════════════════

def repetition_qc_stats(points: list) -> dict:
    x  = np.array([p['x'] for p in points], dtype=float)
    y  = np.array([p['y'] for p in points], dtype=float)
    t  = np.array([p['t'] for p in points], dtype=float)
    pr = np.array([p.get('pressure', 0.5) for p in points], dtype=float)
    duration = float(max(0.0, t[-1] - t[0])) if len(t) > 1 else 0.0
    dx, dy_  = np.diff(x), np.diff(y)
    path     = float(np.sqrt(dx**2 + dy_**2).sum()) if len(x) > 1 else 0.0
    return {
        'n_points': len(points),
        'n_loops':  estimate_loop_count(points),
        'duration': duration,
        'path':     path,
        'std_pressure': safe_std(pr),
    }

def qc_confidence_session(repetitions: list) -> float:
    """Score QC de confiance [0,1] — identique notebook §9. Seuil = QC_GATE = 0.65."""
    if not repetitions:
        return 0.0
    stats_list = [repetition_qc_stats(rep) for rep in repetitions if len(rep) >= 2]
    if not stats_list:
        return 0.0
    df = pd.DataFrame(stats_list)
    sc = 1.0
    if df['n_points'].min() < MIN_POINTS_PER_REP_STRICT:
        sc -= 0.30
    if df['n_loops'].median() < MIN_VALID_LOOPS_PER_REP:
        sc -= 0.35
    if (df['n_loops'].max() - df['n_loops'].min()) > 8:
        sc -= 0.15
    if df['path'].max() / max(float(df['path'].min()), 1e-6) > 1.8:
        sc -= 0.20
    if df['std_pressure'].mean() > 0.20:
        sc -= 0.10
    if df['path'].median() < MIN_PATH_PER_REP:
        sc -= 0.20
    return float(max(0.0, min(1.0, sc)))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 & 5 — PATIENT BASELINE + AUGMENTATION (identique notebook §4/§5)
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_ORDER = [
    'duration_s', 'mean_dt', 'std_dt', 'pause_ratio',
    'mean_velocity', 'std_velocity', 'p95_velocity', 'mean_acceleration',
    'path_length', 'tortuosity', 'bbox_width', 'bbox_height',
    'max_extent', 'size_decay_slope',
    'mean_pressure', 'std_pressure', 'pressure_entropy',
    'velocity_cv', 'duration_cv', 'path_cv',
    'velocity_sample_entropy', 'pressure_lempel_ziv', 'pressure_fractal_dim',
]  # 23 features — §3

@dataclass
class PatientBaseline:
    patient_id:    str
    feature_order: Optional[List[str]] = None
    sessions:      Dict[str, Dict[str, Any]] = field(default_factory=dict)
    scaler:        StandardScaler = field(default_factory=StandardScaler)
    scaler_fitted: bool = False
    feature_hash:  str = ''
    fitted_at:     str = ''

    def add_session(self, session_id: str, repetition_features: list, phase: str = 'baseline'):
        if self.feature_order is None:
            self.feature_order = list(repetition_features[0].keys())
        cleaned = [{k: float(r.get(k, 0.0)) for k in self.feature_order} for r in repetition_features]
        self.sessions[session_id] = {'phase': phase, 'repetition_features': cleaned}

    def get_repetition_matrix(self, phase: str = 'baseline') -> np.ndarray:
        rows = []
        for payload in self.sessions.values():
            if payload['phase'] == phase:
                rows.extend(payload['repetition_features'])
        if not rows:
            raise ValueError(f'Aucune répétition pour phase={phase}')
        fo = self.feature_order or FEATURE_ORDER
        return pd.DataFrame(rows)[fo].values.astype(np.float32)

    def fit_normalization(self):
        Xb = self.get_repetition_matrix('baseline')
        self.scaler.fit(Xb)
        self.scaler_fitted = True
        self.feature_hash  = hashlib.sha256(
            json.dumps(self.feature_order).encode()
        ).hexdigest()[:12]
        self.fitted_at = datetime.now(timezone.utc).isoformat()

    def transform_repetition_features(self, rep_feats: list) -> np.ndarray:
        fo = self.feature_order or FEATURE_ORDER
        X  = pd.DataFrame([{k: float(r.get(k, 0.0)) for k in fo} for r in rep_feats])[fo].values.astype(np.float32)
        return self.scaler.transform(X)

def augment_baseline_features(Xb_s: np.ndarray, ratio: int = AUGMENT_RATIO, seed: int = GLOBAL_SEED) -> np.ndarray:
    """
    Augmentation uniforme §5 :  40% jitter gaussien · 30% magnitude warping · 30% interpolation.
    Appliquée uniquement sur le baseline normalisé.
    """
    rng = np.random.default_rng(seed)
    n, d = Xb_s.shape
    n_aug = n * ratio
    chunks = {'jitter': int(0.4 * n_aug), 'warp': int(0.3 * n_aug)}
    chunks['interp'] = n_aug - chunks['jitter'] - chunks['warp']
    parts = [Xb_s]

    # Jitter gaussien σ=2%
    idx = rng.integers(0, n, size=chunks['jitter'])
    parts.append(Xb_s[idx] + rng.normal(0, 0.02, (chunks['jitter'], d)))

    # Magnitude warping
    idx = rng.integers(0, n, size=chunks['warp'])
    scales = rng.uniform(0.95, 1.05, (chunks['warp'], d))
    parts.append(Xb_s[idx] * scales)

    # Interpolation convexe
    idx1 = rng.integers(0, n, size=chunks['interp'])
    idx2 = rng.integers(0, n, size=chunks['interp'])
    alphas = rng.uniform(0.3, 0.7, (chunks['interp'], 1))
    parts.append(alphas * Xb_s[idx1] + (1 - alphas) * Xb_s[idx2])

    return np.vstack(parts).astype(np.float32)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6/7 — ANOMALY DETECTOR (CalibratedAnomalyDetector — notebook §9)
# ═══════════════════════════════════════════════════════════════════════════════

def threshold_bootstrap(scores, percentile: float = 99.0, n_boot: int = 700,
                         alpha: float = 0.95, seed: int = GLOBAL_SEED) -> float:
    rng = np.random.default_rng(seed)
    s   = np.asarray(scores, dtype=float)
    boot = [np.percentile(rng.choice(s, size=len(s), replace=True), percentile)
            for _ in range(n_boot)]
    return float(np.quantile(boot, alpha))

@dataclass
class AnomalyDetector:
    mode:                 str   = 'lof'      # sélectionné par benchmark §7 → LOF par défaut
    threshold_percentile: float = 99.0
    persistence_required: int   = PERSISTENCE_REQUIRED
    random_state:         int   = GLOBAL_SEED
    threshold_margin:     float = CALIBRATED_THRESHOLD_MARGIN
    threshold_floor:      float = CALIBRATED_THRESHOLD_FLOOR
    threshold_:           Optional[float] = None
    consecutive_count:    int   = 0
    model_:               Any   = None
    b_med_:               float = 0.0
    b_scale_:             float = 1.0
    baseline_session_scores_: Optional[List[float]] = None

    def _build_model(self, Xb_aug: np.ndarray):
        if self.mode == 'lof':
            return LocalOutlierFactor(
                n_neighbors=min(15, max(5, len(Xb_aug) - 1)),
                novelty=True, contamination=0.10
            ).fit(Xb_aug)
        elif self.mode == 'isolation_forest':
            return IsolationForest(
                n_estimators=300, contamination=0.10,
                random_state=self.random_state
            ).fit(Xb_aug)
        raise ValueError(f'Mode inconnu: {self.mode}')

    def _score_reps(self, X: np.ndarray) -> np.ndarray:
        if self.mode == 'lof':
            return -self.model_.decision_function(X)
        elif self.mode == 'isolation_forest':
            return -self.model_.score_samples(X)
        return np.zeros(len(X))

    def fit(self, pb: PatientBaseline):
        Xb    = pb.get_repetition_matrix('baseline')
        Xb_s  = pb.scaler.transform(Xb)
        Xb_aug = augment_baseline_features(Xb_s, ratio=AUGMENT_RATIO, seed=self.random_state)
        self.model_ = self._build_model(Xb_aug)

        b_scores = []
        for sid, payload in pb.sessions.items():
            if payload['phase'] != 'baseline':
                continue
            Xs = pb.transform_repetition_features(payload['repetition_features'])
            b_scores.append(float(np.median(self._score_reps(Xs))))
        self.baseline_session_scores_ = b_scores

        b_arr     = np.asarray(b_scores, dtype=float)
        self.b_med_ = float(np.median(b_arr))
        b_mad     = float(np.median(np.abs(b_arr - self.b_med_)))
        iqr       = float(np.percentile(b_arr, 75) - np.percentile(b_arr, 25))
        b_std     = float(np.std(b_arr, ddof=1)) if len(b_arr) > 1 else 1.0
        scale     = 1.4826 * b_mad
        if scale < 1e-6:
            scale = (iqr / 1.349) if iqr > 0 else b_std
        if scale < 1e-6:
            scale = 1.0
        self.b_scale_ = float(scale)

        z_baseline = [(s - self.b_med_) / self.b_scale_ for s in b_scores]
        b_med_z    = float(np.median(z_baseline))
        b_std_z    = float(np.std(z_baseline, ddof=1)) if len(z_baseline) > 1 else 1.0
        raw_thr    = threshold_bootstrap(z_baseline, percentile=self.threshold_percentile)
        raw_thr    = raw_thr * self.threshold_margin
        raw_thr    = max(raw_thr, b_med_z + 2.5 * b_std_z, self.threshold_floor)
        self.threshold_ = float(raw_thr)

    def score_session(self, pb: PatientBaseline, repetition_features: list) -> dict:
        X      = pb.transform_repetition_features(repetition_features)
        sc_raw = float(np.median(self._score_reps(X)))
        z_rob  = float((sc_raw - self.b_med_) / self.b_scale_)
        exceeded = z_rob > self.threshold_
        self.consecutive_count = self.consecutive_count + 1 if exceeded else 0
        deviation = self.consecutive_count >= self.persistence_required
        return {
            'score':               z_rob,
            'score_raw':           sc_raw,
            'threshold':           float(self.threshold_),
            'deviation':           bool(deviation),
            'consecutive_count':   int(self.consecutive_count),
            'exceeded_threshold':  bool(exceeded),
            'mode':                self.mode,
            'z_robust':            z_rob,
        }

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — CUSUM + MANN-KENDALL (identique notebook §11)
# ═══════════════════════════════════════════════════════════════════════════════

def cusum_upward(scores, mu0: float = 0.0, k: float = None):
    s_arr = np.asarray(scores, dtype=float)
    if k is None:
        std_s = float(np.std(s_arr, ddof=1)) if len(s_arr) > 1 else 1.0
        k = max(0.1, 0.25 * std_s)
    g, gt = [], 0.0
    for s in s_arr:
        gt = max(0.0, gt + (s - (mu0 + k)))
        g.append(gt)
    return np.array(g, dtype=float), float(k)

def mann_kendall_test(series, min_len: int = 8) -> dict:
    y = np.asarray(series, dtype=float)
    if len(y) < min_len:
        return {'tau': 0.0, 'p_value': None, 'trend': 'insufficient_data'}
    tau, pval = stats.kendalltau(np.arange(len(y), dtype=float), y)
    trend = ('increasing' if (tau > 0 and pval < 0.05) else
             'decreasing' if (tau < 0 and pval < 0.05) else 'no_trend')
    return {'tau': float(tau), 'p_value': float(pval), 'trend': trend}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 15b — DIRECTION RULE ENGINE (DRE v7, identique notebook §15b)
# ═══════════════════════════════════════════════════════════════════════════════

class DirectionRuleEngine:
    THRESHOLD_CONFIRMED = 1.2
    THRESHOLD_UNCERTAIN = 0.6
    DRE_FEATURES = [
        'mean_velocity', 'bbox_height', 'mean_pressure',
        'size_decay_slope', 'pause_ratio', 'duration_s'
    ]

    def __init__(self):
        self.baseline_mu_  = {}
        self.baseline_std_ = {}
        self.fitted_       = False

    def fit(self, baseline_df: pd.DataFrame):
        for col in self.DRE_FEATURES:
            if col in baseline_df.columns:
                vals = baseline_df[col].dropna().values.astype(float)
                self.baseline_mu_[col]  = float(np.mean(vals)) if len(vals) else 0.0
                self.baseline_std_[col] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 1.0
        self.fitted_ = True

    def _zscore(self, feat: str, value: float) -> float:
        return (value - self.baseline_mu_.get(feat, 0.0)) / (self.baseline_std_.get(feat, 1.0) + 1e-9)

    def predict(self, session_features: dict) -> dict:
        if not self.fitted_:
            return {'direction': 'uncertain', 'score_manie': 0.0, 'score_depression': 0.0, 'confidence': 0.0}
        z_vel  = self._zscore('mean_velocity',    session_features.get('mean_velocity', 0))
        z_bbox = self._zscore('bbox_height',       session_features.get('bbox_height', 0))
        z_pres = self._zscore('mean_pressure',     session_features.get('mean_pressure', 0))
        z_dec  = self._zscore('size_decay_slope',  session_features.get('size_decay_slope', 0))
        z_pau  = self._zscore('pause_ratio',       session_features.get('pause_ratio', 0))
        z_dur  = self._zscore('duration_s',        session_features.get('duration_s', 0))

        sm = 0.25*z_vel + 0.25*z_bbox + 0.20*z_pres + 0.15*max(0.0, z_dec) + 0.15*(-z_dur)
        sd = 0.25*z_pau + 0.25*(-z_vel) + 0.20*(-z_pres) + 0.15*max(0.0, -z_dec) + 0.15*z_dur
        ms = max(sm, sd)

        if sm >= self.THRESHOLD_CONFIRMED and sm > sd:
            direction = 'manic_risk'
        elif sd >= self.THRESHOLD_CONFIRMED and sd > sm:
            direction = 'depressive_risk'
        elif ms >= self.THRESHOLD_UNCERTAIN:
            direction = 'uncertain_leaning_' + ('manic' if sm > sd else 'depressive')
        else:
            direction = 'uncertain'

        return {
            'direction':        direction,
            'score_manie':      round(float(sm), 3),
            'score_depression': round(float(sd), 3),
            'confidence':       round(float(np.clip(ms / (self.THRESHOLD_CONFIRMED * 2), 0, 1)), 3),
        }

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — PATIENT LOCK MANAGER + IN-MEMORY STATE
# ═══════════════════════════════════════════════════════════════════════════════

class PatientLockManager:
    """Thread-safe — évite les race conditions sur 10 requêtes concurrentes (§P8)."""
    def __init__(self):
        self._locks: Dict[str, threading.Lock] = {}
        self._meta_lock = threading.Lock()

    def acquire(self, patient_id: str):
        with self._meta_lock:
            if patient_id not in self._locks:
                self._locks[patient_id] = threading.Lock()
        return self._locks[patient_id]

    def active_count(self) -> int:
        with self._meta_lock:
            return sum(1 for lk in self._locks.values() if lk.locked())

lock_manager = PatientLockManager()

# État en mémoire (chargé depuis artifacts_mvp/ si disponible)
patient_baselines:  Dict[str, PatientBaseline]      = {}
patient_detectors:  Dict[str, AnomalyDetector]      = {}
patient_dre:        Dict[str, DirectionRuleEngine]   = {}
session_history:    Dict[str, List[dict]]            = {}  # patient_id → liste de résultats

def simulated_clinical_state(condition: str) -> str:
    if condition in ['fast', 'mixed']:
        return 'manic_risk'
    if condition in ['slow', 'fatigue']:
        return 'depressive_risk'
    return 'stable_like'

def _load_patient_artifacts(patient_id: str) -> bool:
    """Charge les artefacts sauvegardés par le notebook §12 (scaler + detector_state)."""
    p_dir = ARTIFACT_ROOT / patient_id
    if not p_dir.exists():
        return False
    try:
        scaler   = joblib.load(p_dir / 'scaler.joblib')
        det_st   = joblib.load(p_dir / 'detector_state.joblib')
        meta     = joblib.load(p_dir / 'metadata.joblib')

        pb = PatientBaseline(patient_id=patient_id)
        pb.scaler        = scaler
        pb.scaler_fitted = True
        pb.feature_order = meta.get('feature_order', FEATURE_ORDER)
        pb.feature_hash  = meta.get('feature_hash', '')
        pb.fitted_at     = meta.get('fitted_at', '')
        patient_baselines[patient_id] = pb

        det = AnomalyDetector(mode=det_st.get('mode', 'lof'))
        det.model_                   = det_st.get('model_')
        det.threshold_               = det_st.get('threshold_')
        det.threshold_floor          = det_st.get('threshold_floor_', CALIBRATED_THRESHOLD_FLOOR)
        det.baseline_session_scores_ = det_st.get('baseline_session_scores_', [])
        det.consecutive_count        = det_st.get('consecutive_count', 0)
        det.b_med_                   = det_st.get('b_med_', 0.0)
        det.b_scale_                 = det_st.get('b_scale_', 1.0)
        patient_detectors[patient_id] = det

        log.info(f'Artefacts chargés: {patient_id} | mode={det.mode} | thr={det.threshold_:.3f}')
        return True
    except Exception as e:
        log.warning(f'Chargement artefacts {patient_id} échoué: {e}')
        return False

def _get_or_create_patient(patient_id: str) -> tuple:
    if patient_id not in patient_baselines:
        _load_patient_artifacts(patient_id)
    if patient_id not in patient_baselines:
        pb  = PatientBaseline(patient_id=patient_id)
        det = AnomalyDetector(mode='lof')
        patient_baselines[patient_id] = pb
        patient_detectors[patient_id] = det
    return patient_baselines[patient_id], patient_detectors[patient_id]

def _log_alert(patient_id: str, result: dict):
    """Persiste les alertes dans alert_log.json pour /dashboard/alerts."""
    try:
        log_data = []
        if ALERT_LOG.exists():
            log_data = json.loads(ALERT_LOG.read_text())
        log_data.append({
            'patient_id':   patient_id,
            'timestamp':    datetime.now(timezone.utc).isoformat(),
            'score':        result.get('score'),
            'threshold':    result.get('threshold'),
            'deviation':    result.get('deviation'),
            'direction':    result.get('direction_prediction', 'uncertain'),
            'qc_confidence': result.get('qc_confidence'),
            'consecutive_count': result.get('consecutive_count', 0),
        })
        ALERT_LOG.write_text(json.dumps(log_data, indent=2))
    except Exception as e:
        log.warning(f'Écriture alert_log échouée: {e}')

# ═══════════════════════════════════════════════════════════════════════════════
# QUESTIONNAIRES CLINIQUES (YMRS · MADRS · ASRM · PHQ-9)
# ═══════════════════════════════════════════════════════════════════════════════

QUESTIONNAIRE_CATALOG = {
    'ymrs': {
        'items': [
            {'id': 'elevated_mood',               'max_score': 4},
            {'id': 'increased_motor_activity',    'max_score': 4},
            {'id': 'sexual_interest',             'max_score': 4},
            {'id': 'sleep',                       'max_score': 4},
            {'id': 'irritability',                'max_score': 8},
            {'id': 'speech',                      'max_score': 8},
            {'id': 'language_thought_disorder',   'max_score': 4},
            {'id': 'content',                     'max_score': 8},
            {'id': 'disruptive_aggressive_behavior','max_score': 8},
            {'id': 'appearance',                  'max_score': 4},
            {'id': 'insight',                     'max_score': 4},
        ],
        'labels': {
            'fr': {
                'elevated_mood': 'Humeur élevée',
                'increased_motor_activity': 'Activité motrice augmentée',
                'sexual_interest': 'Intérêt sexuel',
                'sleep': 'Sommeil',
                'irritability': 'Irritabilité',
                'speech': 'Débit de parole',
                'language_thought_disorder': 'Trouble langage-pensée',
                'content': 'Contenu de pensée',
                'disruptive_aggressive_behavior': 'Comportement agressif',
                'appearance': 'Apparence',
                'insight': 'Insight',
            },
            'en': {
                'elevated_mood': 'Elevated mood',
                'increased_motor_activity': 'Increased motor activity',
                'sexual_interest': 'Sexual interest',
                'sleep': 'Sleep',
                'irritability': 'Irritability',
                'speech': 'Speech rate/amount',
                'language_thought_disorder': 'Language-thought disorder',
                'content': 'Thought content',
                'disruptive_aggressive_behavior': 'Disruptive/aggressive behavior',
                'appearance': 'Appearance',
                'insight': 'Insight',
            },
        },
    },
    'madrs': {
        'items': [
            {'id': 'apparent_sadness',         'max_score': 6},
            {'id': 'reported_sadness',         'max_score': 6},
            {'id': 'inner_tension',            'max_score': 6},
            {'id': 'reduced_sleep',            'max_score': 6},
            {'id': 'reduced_appetite',         'max_score': 6},
            {'id': 'concentration_difficulties','max_score': 6},
            {'id': 'lassitude',                'max_score': 6},
            {'id': 'inability_to_feel',        'max_score': 6},
            {'id': 'pessimistic_thoughts',     'max_score': 6},
            {'id': 'suicidal_thoughts',        'max_score': 6},
        ],
        'labels': {
            'fr': {
                'apparent_sadness': 'Tristesse apparente',
                'reported_sadness': 'Tristesse exprimée',
                'inner_tension': 'Tension intérieure',
                'reduced_sleep': 'Réduction du sommeil',
                'reduced_appetite': "Réduction de l'appétit",
                'concentration_difficulties': 'Difficultés de concentration',
                'lassitude': 'Lassitude',
                'inability_to_feel': 'Incapacité à ressentir',
                'pessimistic_thoughts': 'Pensées pessimistes',
                'suicidal_thoughts': 'Pensées suicidaires',
            },
            'en': {
                'apparent_sadness': 'Apparent sadness',
                'reported_sadness': 'Reported sadness',
                'inner_tension': 'Inner tension',
                'reduced_sleep': 'Reduced sleep',
                'reduced_appetite': 'Reduced appetite',
                'concentration_difficulties': 'Concentration difficulties',
                'lassitude': 'Lassitude',
                'inability_to_feel': 'Inability to feel',
                'pessimistic_thoughts': 'Pessimistic thoughts',
                'suicidal_thoughts': 'Suicidal thoughts',
            },
        },
    },
}

def build_catalog_response(lang: str) -> dict:
    lang = lang if lang in ('fr', 'en', 'ar') else 'fr'
    scales = {}
    for scale_id, scale_data in QUESTIONNAIRE_CATALOG.items():
        label_map = scale_data.get('labels', {}).get(lang) or scale_data.get('labels', {}).get('fr', {})
        items = []
        for item in scale_data['items']:
            items.append({
                'id':        item['id'],
                'max_score': item['max_score'],
                'label':     label_map.get(item['id'], item['id']),
            })
        scales[scale_id] = {'items': items}
    return {'language': lang, 'scales': scales}

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES FLASK
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/protocol_config', methods=['GET'])
def protocol_config():
    """Retourne les constantes QC + protocole clinique au HTML capture."""
    return jsonify({
        'qc': {
            'min_points_per_rep_strict': MIN_POINTS_PER_REP_STRICT,
            'min_path_per_rep':          MIN_PATH_PER_REP,
            'min_loops_per_rep':         MIN_VALID_LOOPS_PER_REP,
            'qc_gate':                   QC_GATE,
        },
        'baseline': {
            'required_sessions':        BASELINE_REQUIRED_SESSIONS,
            'repetitions_per_session':  1,
            'session_protocol': {
                1: {'questionnaires': ['ymrs_full', 'madrs_full'],
                    'label': 'Écriture + YMRS complet + MADRS complet'},
                2: {'questionnaires': ['asrm_short', 'phq9_short'],
                    'label': 'Écriture + ASRM court + PHQ-9 court'},
                3: {'questionnaires': ['asrm_short', 'phq9_short'],
                    'label': 'Écriture + ASRM court + PHQ-9 court → FIT modèle'},
            },
        },
        'monitoring': {
            'repetitions_per_session':   1,
            'questionnaire_default':     'none',
            'alert_j1': {
                'consecutive_required':  1,
                'action':               'revenir_le_lendemain',
                'questionnaire':        'none',
            },
            'alert_confirmed': {
                'consecutive_required':  2,
                'qc_gate_required':      QC_GATE,
                'action':               'dre_classification',
                'questionnaire':        ['ymrs_full', 'madrs_full'],
                'dre_directions':       ['manic_risk', 'depressive_risk', 'uncertain'],
            },
        },
        'tasks': {
            'letter_l': {
                'required_reps':         1,
                'warmup_loops':          LOOPS_WARMUP,
                'target_counted_loops':  TARGET_LOOPS_PER_REP,
            },
        },
        'model': {
            'selected':           'LOF',
            'mode':               'lof',
            'threshold_margin':   CALIBRATED_THRESHOLD_MARGIN,
            'threshold_floor':    CALIBRATED_THRESHOLD_FLOOR,
            'persistence_required': PERSISTENCE_REQUIRED,
        },
        'global_seed': GLOBAL_SEED,
    })


@app.route('/questionnaire_catalog', methods=['GET'])
def questionnaire_catalog():
    """Catalogue des questionnaires YMRS + MADRS pour l'interface vocale."""
    lang = request.args.get('lang', 'fr')
    return jsonify(build_catalog_response(lang))


@app.route('/onboarding/<patient_id>/session', methods=['POST'])
def onboarding_session(patient_id: str):
    """
    Enregistrement d'une session baseline (protocole 3 sessions · 1 répétition).

    Protocole questionnaires :
      Session 1 → YMRS complet + MADRS complet   (onboarding clinique)
      Session 2 → ASRM court + PHQ-9 court
      Session 3 → ASRM court + PHQ-9 court → FIT du modèle après cette session

    Le frontend envoie les scores questionnaires dans le corps JSON.
    Après session 3 : scaler + AnomalyDetector + DRE fittés → monitoring prêt.
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({'error': 'Corps JSON manquant'}), 400

    lock = lock_manager.acquire(patient_id)
    with lock:
        try:
            repetitions = data.get('repetitions', [])
            if not repetitions:
                return jsonify({'error': 'Aucune répétition fournie'}), 422

            # QC — seuil bas pour le baseline (patient peut être peu habitué)
            qc = qc_confidence_session(repetitions)
            if qc < 0.35:
                return jsonify({
                    'error': f'Qualité session insuffisante: conf={qc:.2f} < 0.35',
                    'qc_confidence': round(qc, 3),
                }), 422

            # Features (1 répétition)
            bundle = extract_session_bundle(repetitions)

            pb, det = _get_or_create_patient(patient_id)
            n_baseline = sum(1 for v in pb.sessions.values() if v['phase'] == 'baseline')
            session_number = n_baseline + 1  # 1, 2 ou 3
            session_id = data.get('session_id', f'{patient_id}_B{session_number:02d}')

            if session_number > BASELINE_REQUIRED_SESSIONS:
                return jsonify({
                    'error': f'Baseline déjà complète ({BASELINE_REQUIRED_SESSIONS} sessions enregistrées). '
                             f'Utilisez /score pour le monitoring.',
                    'n_baseline': n_baseline,
                }), 409

            pb.add_session(session_id, bundle['repetition_features'], phase='baseline')
            n_baseline = session_number  # mis à jour

            # ── Questionnaire attendu selon session ───────────────────────────
            if session_number == 1:
                questionnaire_required = ['ymrs_full', 'madrs_full']
                questionnaire_label    = 'YMRS complet + MADRS complet (onboarding clinique)'
            else:
                questionnaire_required = ['asrm_short', 'phq9_short']
                questionnaire_label    = 'ASRM court + PHQ-9 court (suivi rapide)'

            # ── Fit après la 3e session ───────────────────────────────────────
            model_fitted = False
            if n_baseline >= BASELINE_REQUIRED_SESSIONS and not pb.scaler_fitted:
                pb.fit_normalization()
                det.fit(pb)
                model_fitted = True
                log.info(f'Modèle fitted: {patient_id} | {n_baseline} sessions baseline | thr={det.threshold_:.3f}')

                baseline_rows = [r for v in pb.sessions.values()
                                 if v['phase'] == 'baseline'
                                 for r in v['repetition_features']]
                if baseline_rows:
                    dre = DirectionRuleEngine()
                    dre.fit(pd.DataFrame(baseline_rows))
                    patient_dre[patient_id] = dre
                    log.info(f'DRE fitted: {patient_id} | {len(baseline_rows)} reps baseline')

            remaining = max(0, BASELINE_REQUIRED_SESSIONS - n_baseline)
            if model_fitted:
                msg = (f'✅ Baseline complète ({n_baseline}/{BASELINE_REQUIRED_SESSIONS}) — '
                       f'Modèle prêt. Monitoring peut commencer.')
            else:
                msg = (f'Session baseline #{n_baseline}/{BASELINE_REQUIRED_SESSIONS} enregistrée. '
                       f'{remaining} session(s) restante(s).')

            return jsonify({
                'status':                 'baseline_recorded',
                'patient_id':            patient_id,
                'session_id':            session_id,
                'session_number':        session_number,
                'n_baseline':            n_baseline,
                'baseline_complete':     n_baseline >= BASELINE_REQUIRED_SESSIONS,
                'remaining_sessions':    remaining,
                'qc_confidence':         round(qc, 3),
                'model_fitted':          pb.scaler_fitted,
                'questionnaire_required': questionnaire_required,
                'questionnaire_label':   questionnaire_label,
                'message':               msg,
            })
        except Exception as e:
            log.error(f'Erreur onboarding {patient_id}: {e}', exc_info=True)
            return jsonify({'error': str(e)}), 500


@app.route('/score', methods=['POST'])
def score_session():
    """
    Scoring anomalie — session monitoring (1 répétition, zéro questionnaire par défaut).

    Logique de décision :
      score z_robust ≤ seuil              → OK, rien à faire
      score z_robust > seuil             → déviation (exceeded_threshold=True)
        consecutive_count = 1            → ALERTE_J1 : revenir le lendemain
        consecutive_count ≥ 2            → ALERTE_CONFIRMÉE :
                                            - DRE activé → direction manic/depressive
                                            - Questionnaire YMRS+MADRS complet requis
                                            - QC gate appliquée (conf ≥ 0.65)
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({'error': 'Corps JSON manquant'}), 400

    patient_id = data.get('patient_id', '').strip()
    if not patient_id:
        return jsonify({'error': 'patient_id manquant'}), 422

    lock = lock_manager.acquire(patient_id)
    with lock:
        try:
            repetitions = data.get('repetitions', [])
            if not repetitions:
                return jsonify({'error': 'Aucune répétition fournie'}), 422

            # QC gate §9
            qc = qc_confidence_session(repetitions)

            # Features
            bundle = extract_session_bundle(repetitions)

            pb, det = _get_or_create_patient(patient_id)

            if not pb.scaler_fitted or det.model_ is None:
                return jsonify({
                    'error': (f'Baseline incomplète — '
                              f'effectuez {BASELINE_REQUIRED_SESSIONS} sessions baseline via '
                              f'/onboarding/<patient_id>/session'),
                    'qc_confidence': round(qc, 3),
                }), 409

            # ── Score anomalie §9 ─────────────────────────────────────────────
            result = det.score_session(pb, bundle['repetition_features'])
            consecutive = result['consecutive_count']

            # QC gate : alerte supprimée si qualité insuffisante
            alert_confirmed = result['deviation'] and (qc >= QC_GATE)

            # ── Décision clinique ─────────────────────────────────────────────
            # exceeded_threshold = True mais consecutive=1 → alerte J1 (revenir demain)
            # consecutive ≥ 2 + QC OK → alerte confirmée → DRE + YMRS+MADRS
            alert_j1        = result['exceeded_threshold'] and consecutive == 1
            alert_confirmed = result['exceeded_threshold'] and consecutive >= 2 and (qc >= QC_GATE)

            # ── Direction DRE §15b (uniquement si alerte confirmée) ───────────
            direction_pred = None
            dre_scores     = {}
            if alert_confirmed and patient_id in patient_dre:
                mean_feats = {k: float(np.mean([r.get(k, 0) for r in bundle['repetition_features']]))
                              for k in FEATURE_ORDER}
                dre_out        = patient_dre[patient_id].predict(mean_feats)
                direction_pred = dre_out['direction']
                dre_scores     = dre_out

            # ── Drift CUSUM §11 ───────────────────────────────────────────────
            session_id = data.get('session_id', '')
            hist = session_history.setdefault(patient_id, [])
            hist.append({'score': result['score'], 'session_id': session_id})
            z_series = [h['score'] for h in hist]
            cusum_vals, cusum_k = cusum_upward(z_series, mu0=0.0)
            cusum_value = float(cusum_vals[-1])
            cusum_alert = cusum_value > 4.0
            mk = mann_kendall_test(z_series)

            # ── Questionnaire selon le statut ─────────────────────────────────
            # Alerte confirmée : DRE d'abord
            #   DRE confident (score ≥ 1.2) → classification directe, AUCUN questionnaire
            #   DRE uncertain  (score < 0.6) → ASRM court + PHQ-9 court pour trancher
            if alert_confirmed:
                dre_is_confident = (dre_scores.get('confidence', 0) or 0) >= 0.5  # correspond à ms ≥ 1.2 normalisé
                if dre_is_confident and direction_pred and direction_pred != 'uncertain':
                    questionnaire_required = []
                    questionnaire_label    = f'Aucun questionnaire — DRE classifie avec confiance ({direction_pred})'
                else:
                    questionnaire_required = ['asrm_short', 'phq9_short']
                    questionnaire_label    = 'ASRM court + PHQ-9 court — DRE incertain, questionnaire court requis'
                status_label  = 'ALERTE_CONFIRMÉE'
                clinical_label = f'Alerte confirmée · direction DRE: {direction_pred or "incertain"}'
            elif alert_j1:
                questionnaire_required = []
                questionnaire_label    = 'Aucun questionnaire — revenir demain pour confirmer'
                status_label           = 'ALERTE_J1'
                clinical_label         = 'Déviation J1 — revenir le lendemain'
            else:
                questionnaire_required = []
                questionnaire_label    = ''
                status_label           = 'OK'
                clinical_label         = 'Normal — aucune déviation'

            response = {
                'patient_id':             patient_id,
                'session_id':             session_id,
                # Scores
                'score':                  round(result['score'], 4),
                'threshold':              round(result['threshold'], 4),
                'z_robust':               round(result['z_robust'], 4),
                # Décision
                'exceeded_threshold':     result['exceeded_threshold'],
                'consecutive_count':      consecutive,
                'alert_j1':               bool(alert_j1),
                'alert_confirmed':        bool(alert_confirmed),
                'deviation':              result['deviation'],
                # QC
                'qc_confidence':          round(qc, 3),
                'qc_passed':              qc >= QC_GATE,
                # Direction (seulement si alerte confirmée)
                'direction_prediction':   direction_pred,
                'score_manie':            dre_scores.get('score_manie'),
                'score_depression':       dre_scores.get('score_depression'),
                'dre_confidence':         dre_scores.get('confidence'),
                # Drift
                'cusum_value':            round(cusum_value, 3),
                'cusum_alert':            cusum_alert,
                'cusum_k':                round(cusum_k, 3),
                'mk_trend':               mk['trend'],
                'mk_tau':                 mk.get('tau'),
                # Labels
                'status_label':           status_label,
                'clinical_label':         clinical_label,
                'mode':                   result['mode'],
                'n_sessions_scored':      len(hist),
                # Questionnaire
                'questionnaire_required': questionnaire_required,
                'questionnaire_label':    questionnaire_label,
                # Message résumé pour le clinicien
                'message': (
                    f'🚨 ALERTE CONFIRMÉE — {consecutive} déviations consécutives. '
                    f'Direction DRE : {direction_pred}. '
                    f'{"Classification directe (DRE confiant)." if not questionnaire_required else "DRE incertain → ASRM + PHQ-9 court requis."}'
                    if alert_confirmed else
                    f'⏰ Déviation J1 (z={result["score"]:.2f} > seuil={result["threshold"]:.2f}). '
                    f'Revenir demain.'
                    if alert_j1 else
                    f'✅ Normal — z={result["score"]:.2f}, seuil={result["threshold"]:.2f}'
                ),
            }

            if alert_confirmed:
                _log_alert(patient_id, response)

            return jsonify(response)

        except Exception as e:
            log.error(f'Erreur score {patient_id}: {e}', exc_info=True)
            return jsonify({'error': str(e)}), 500


@app.route('/questionnaires/score', methods=['POST'])
def questionnaires_score():
    """
    Reçoit les réponses YMRS + MADRS, calcule les totaux,
    retourne un label clinique et peut déclencher un questionnaire de suivi.
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({'error': 'Corps JSON manquant'}), 400

    answers = data.get('questionnaire_answers', {})
    lang    = data.get('lang', 'fr')

    ymrs_total  = sum(int(v) for v in answers.get('ymrs', {}).values() if v is not None)
    madrs_total = sum(int(v) for v in answers.get('madrs', {}).values() if v is not None)

    # Interprétation clinique YMRS (Youngetal. 1978)
    if ymrs_total >= 20:
        ymrs_label = 'Manie sévère'
    elif ymrs_total >= 12:
        ymrs_label = 'Manie modérée'
    elif ymrs_total >= 6:
        ymrs_label = 'Hypomanie légère'
    else:
        ymrs_label = 'Euthymie (manie)'

    # Interprétation MADRS (Montgomery & Åsberg 1979)
    if madrs_total >= 35:
        madrs_label = 'Dépression sévère'
    elif madrs_total >= 20:
        madrs_label = 'Dépression modérée'
    elif madrs_total >= 7:
        madrs_label = 'Dépression légère'
    else:
        madrs_label = 'Euthymie (dépression)'

    # Direction clinique combinée
    if ymrs_total >= 12 and ymrs_total > madrs_total:
        direction = 'manic_risk'
        clinical_label = f'Risque maniaque — {ymrs_label}'
    elif madrs_total >= 20 and madrs_total > ymrs_total:
        direction = 'depressive_risk'
        clinical_label = f'Risque dépressif — {madrs_label}'
    else:
        direction = 'stable_like'
        clinical_label = 'Euthymie — pas d\'alerte clinique questionnaire'

    return jsonify({
        'ymrs_total':    ymrs_total,
        'madrs_total':   madrs_total,
        'ymrs_label':    ymrs_label,
        'madrs_label':   madrs_label,
        'direction':     direction,
        'clinical_label': clinical_label,
        'label':         clinical_label,
        'message':       f'YMRS={ymrs_total} · MADRS={madrs_total} — {clinical_label}',
        'score':         max(ymrs_total / 60.0, madrs_total / 60.0),
        'threshold':     0.20,
        'deviation':     direction != 'stable_like',
    })


@app.route('/dashboard/alerts', methods=['GET'])
def dashboard_alerts():
    """
    Vue agrégée des alertes pour le clinicien — 7 derniers jours.
    Correspond à GET /dashboard/alerts?days=7 dans le HTML.
    """
    days = int(request.args.get('days', 7))
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    alerts_by_patient: Dict[str, dict] = {}
    if ALERT_LOG.exists():
        try:
            log_data = json.loads(ALERT_LOG.read_text())
            for entry in log_data:
                ts = datetime.fromisoformat(entry['timestamp'])
                if ts < cutoff:
                    continue
                pid = entry['patient_id']
                if pid not in alerts_by_patient:
                    alerts_by_patient[pid] = {
                        'patient_id':    pid,
                        'n_alerts_in_window': 0,
                        'last_score':    None,
                        'direction':     entry.get('direction', 'uncertain'),
                        'last_timestamp': entry['timestamp'],
                    }
                alerts_by_patient[pid]['n_alerts_in_window'] += 1
                alerts_by_patient[pid]['last_score']    = entry.get('score')
                alerts_by_patient[pid]['last_timestamp']= entry['timestamp']
                alerts_by_patient[pid]['direction']     = entry.get('direction', 'uncertain')
        except Exception as e:
            log.warning(f'Lecture alert_log échouée: {e}')

    total_monitored = len(patient_baselines)
    return jsonify({
        'total_alerts':             len(alerts_by_patient),
        'total_patients_monitored': total_monitored,
        'days_window':              days,
        'alerts':                   list(alerts_by_patient.values()),
    })


@app.route('/admin/locks', methods=['GET'])
def admin_locks():
    """État des verrous patients (threading.Lock) — §P8."""
    return jsonify({
        'active_patient_locks': lock_manager.active_count(),
        'known_patients':       list(patient_baselines.keys()),
        'n_patients':           len(patient_baselines),
    })


# ── Health check ──────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':  'ok',
        'project': 'BridgingBipolar',
        'version': '1.0',
        'notebook_constants': {
            'GLOBAL_SEED':               GLOBAL_SEED,
            'MIN_POINTS_PER_REP_STRICT': MIN_POINTS_PER_REP_STRICT,
            'MIN_PATH_PER_REP':          MIN_PATH_PER_REP,
            'TARGET_LOOPS_PER_REP':      TARGET_LOOPS_PER_REP,
            'MIN_VALID_LOOPS_PER_REP':   MIN_VALID_LOOPS_PER_REP,
            'QC_GATE':                   QC_GATE,
        },
        'torch_available':   TORCH_AVAILABLE,
        'patients_loaded':   list(patient_baselines.keys()),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP — chargement des artefacts notebook §12 si disponibles
# ═══════════════════════════════════════════════════════════════════════════════

def _startup():
    if ARTIFACT_ROOT.exists():
        for p_dir in ARTIFACT_ROOT.iterdir():
            if p_dir.is_dir():
                pid = p_dir.name
                ok = _load_patient_artifacts(pid)
                if ok and pid in patient_baselines:
                    # Fit DRE sur les données baseline chargées
                    pb = patient_baselines[pid]
                    baseline_rows = [r for v in pb.sessions.values()
                                     if v.get('phase') == 'baseline'
                                     for r in v['repetition_features']]
                    if baseline_rows:
                        dre = DirectionRuleEngine()
                        dre.fit(pd.DataFrame(baseline_rows))
                        patient_dre[pid] = dre
    log.info(
        f'BridgingBipolar API démarrée | Patients: {list(patient_baselines.keys())} | '
        f'Torch: {"✅" if TORCH_AVAILABLE else "❌ (optionnel)"}'
    )


if __name__ == '__main__':
    _startup()
    app.run(host='0.0.0.0', port=5000, debug=False)
