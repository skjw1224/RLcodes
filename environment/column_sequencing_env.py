"""
6성분 경질 탄화수소 증류 시퀀싱 환경
======================================
FUG 숏컷 모델 + 강화학습 환경

위에서 아래로 읽으면서 이해할 수 있도록 구성:
  섹션 1: VLE 기초 (Antoine, 상대휘발도)
  섹션 2: FUG 숏컷 계산 (Fenske, Underwood, Gilliland)
  섹션 3: bubble point (고압 운전용)
  섹션 4: 비용 모델 (TAC 추정)
  섹션 5: 컬럼 solver
  섹션 6: 강화학습 환경 (Gym-style Tree-MDP)
"""
import numpy as np
from collections import deque

from .column_sequencing_config import *


# ============================================================
# 섹션 1: VLE 기초 함수
# ============================================================

def antoine_psat(T_celsius, A, B, C):
    """
    Antoine 식으로 순수 성분의 포화 증기압을 계산한다.
    log10(Psat[mmHg]) = A - B / (C + T[°C])

    Returns: Psat in Pa
    """
    psat_mmHg = 10.0 ** (A - B / (C + T_celsius))
    psat_Pa = psat_mmHg * 133.322  # mmHg → Pa
    return psat_Pa


def get_relative_volatilities(T_celsius, P_total, antoine_params, hk_idx):
    """
    주어진 온도에서 각 성분의 상대휘발도를 계산한다.
    alpha_i = Psat_i / Psat_HK  (Heavy Key 기준)

    Returns: alpha 배열
    """
    psats = np.array([
        antoine_psat(T_celsius, *antoine_params[i])
        for i in range(len(antoine_params))
    ])
    alpha = psats / psats[hk_idx]
    return alpha


# ============================================================
# 섹션 2: FUG 숏컷 계산
# ============================================================

def fenske_min_stages(alpha_LK, recovery_LK=0.99, recovery_HK=0.99):
    """
    Fenske 식: 최소 이론 단수를 계산한다.

    Nmin = ln[(d_LK/b_LK) * (b_HK/d_HK)] / ln(alpha_LK)
         = ln[(rec_LK/(1-rec_LK)) * (rec_HK/(1-rec_HK))] / ln(alpha_LK)

    Args:
        alpha_LK: Light Key의 상대휘발도 (HK 기준, 즉 > 1)
        recovery_LK: LK의 distillate 회수율 (기본 0.99)
        recovery_HK: HK의 bottoms 회수율 (기본 0.99)

    Returns: Nmin (최소 단수)
    """
    numerator = np.log(
        (recovery_LK / (1.0 - recovery_LK)) *
        (recovery_HK / (1.0 - recovery_HK))
    )
    denominator = np.log(alpha_LK)
    Nmin = numerator / denominator
    return Nmin


def underwood_min_reflux(alphas, z_feed, distillate_fracs, lk_idx, hk_idx, q=1.0):
    """
    Underwood 식: 최소 환류비(Rmin)를 계산한다.

    Step 1: theta를 구한다 (Underwood root)
      sum(alpha_i * z_i / (alpha_i - theta)) = 1 - q
      saturated liquid feed이면 q=1 → 우변 = 0
      theta는 alpha_HK와 alpha_LK 사이에 존재한다.

    Step 2: Rmin을 구한다
      Rmin + 1 = sum(alpha_i * xD_i / (alpha_i - theta))

    Args:
        alphas: 각 성분의 상대휘발도 배열
        z_feed: 피드 조성 (몰분율)
        distillate_fracs: distillate 조성 (몰분율)
        lk_idx: Light Key 인덱스
        hk_idx: Heavy Key 인덱스
        q: feed thermal condition (1.0 = saturated liquid)

    Returns: Rmin
    """
    theta_low = alphas[hk_idx] + 1e-6
    theta_high = alphas[lk_idx] - 1e-6

    if theta_low >= theta_high:
        theta_low = min(alphas[hk_idx], alphas[lk_idx]) + 1e-6
        theta_high = max(alphas[hk_idx], alphas[lk_idx]) - 1e-6

    target = 1.0 - q  # saturated liquid이면 0

    for _ in range(200):
        theta = (theta_low + theta_high) / 2.0
        f_val = np.sum(alphas * z_feed / (alphas - theta))

        if f_val > target:
            theta_high = theta
        else:
            theta_low = theta

        if abs(f_val - target) < 1e-10:
            break

    Rmin_plus_1 = np.sum(alphas * distillate_fracs / (alphas - theta))
    Rmin = Rmin_plus_1 - 1.0
    Rmin = max(Rmin, 0.01)
    return Rmin


def gilliland_actual_stages(Nmin, R, Rmin):
    """
    Gilliland 상관식 (Molokanov 형태): 실제 단수를 계산한다.

    X = (R - Rmin) / (R + 1)
    Y = 1 - exp[(1 + 54.4*X) / (11 + 117.2*X) * (X - 1) / sqrt(X)]
    N = (Nmin + Y) / (1 - Y)

    Args:
        Nmin: 최소 단수 (Fenske)
        R: 실제 환류비
        Rmin: 최소 환류비 (Underwood)

    Returns: N_actual (실제 단수, 정수로 올림)
    """
    X = (R - Rmin) / (R + 1.0)
    X = np.clip(X, 0.001, 0.999)

    Y = 1.0 - np.exp(
        (1.0 + 54.4 * X) / (11.0 + 117.2 * X) * (X - 1.0) / np.sqrt(X)
    )

    N_actual = (Nmin + Y) / (1.0 - Y)
    return int(np.ceil(N_actual))


# ============================================================
# 섹션 3: bubble point (고압 운전용)
# ============================================================

def bubble_point_temperature(z, P_total, antoine_params):
    """
    고압 운전용 bubble point 계산.
    경질 탄화수소는 10 atm에서 -30°C ~ 130°C 범위이므로
    탐색 범위를 -50 ~ 250°C로 확장한다.
    Raoult's law: sum(z_i * Psat_i(T) / P_total) = 1을 bisection으로 풀기.
    """
    T_low, T_high = -50.0, 250.0

    for _ in range(100):
        T_mid = (T_low + T_high) / 2.0
        sum_y = 0.0
        for i in range(len(z)):
            A, B, C = antoine_params[i]
            psat = antoine_psat(T_mid, A, B, C)
            sum_y += z[i] * psat / P_total

        if sum_y > 1.0:
            T_high = T_mid
        else:
            T_low = T_mid

        if abs(sum_y - 1.0) < 1e-8:
            break

    return T_mid


# ============================================================
# 섹션 4: 비용 모델 (Douglas 방법 기반)
# ============================================================

def estimate_TAC(N_stages, R, distillate_flow_total, avg_latent_heat_kJ,
                 P_operating=FEED_PRESSURE):
    """
    TAC (Total Annual Cost) 추정. Douglas (1988) 방법 기반.

    Capital cost: 컬럼 shell + tray + condenser + reboiler
    Operating cost: 스팀(reboiler) + 냉각수(condenser)

    Tray cost: N^1.2 비선형 패널티 적용.
    → 단수가 많을수록(낮은 R/Rmin) 설치비가 비선형적으로 증가
    → 어려운 분리(N이 큰 컬럼)일수록 최적 R/Rmin이 더 높게 형성됨

    Args:
        N_stages: 실제 단수
        R: 실제 환류비
        distillate_flow_total: distillate 총 유량 (mol/s)
        avg_latent_heat_kJ: 평균 잠열 (kJ/mol)
        P_operating: 운전 압력 (Pa), 기본값 10 atm

    Returns: TAC ($/year)
    """
    V = distillate_flow_total * (R + 1.0)

    T_avg_K = 50.0 + 273.15  # ~50°C (고압 운전 평균 온도)
    V_volumetric = V * 8.314 * T_avg_K / P_operating
    diameter = 1.2 * np.sqrt(4.0 * V_volumetric / (np.pi * 0.7))
    diameter = max(diameter, 0.5)

    tray_spacing = 0.6
    height = N_stages * tray_spacing + 3.0

    shell_cost = 1000.0 * (diameter ** 1.066) * (height ** 0.802)
    TRAY_COEFF = 300.0 / (30.0 ** 0.3)  # ≈ 108.1
    tray_cost = TRAY_COEFF * (N_stages ** 1.2) * (diameter ** 1.55) + STAGE_FIXED_COST * N_stages

    Qr_kW = V * avg_latent_heat_kJ
    Qc_kW = Qr_kW

    condenser_cost = 500.0 * (Qc_kW ** 0.6)
    reboiler_cost = 500.0 * (Qr_kW ** 0.6)

    capital_total = CAPITAL_FACTOR * (shell_cost + tray_cost + condenser_cost + reboiler_cost)

    steam_annual = Qr_kW * ANNUAL_OPERATING_SECONDS * STEAM_COST
    cooling_annual = Qc_kW * ANNUAL_OPERATING_SECONDS * COOLING_COST

    operating_total = steam_annual + cooling_annual
    TAC = capital_total / PAYBACK_YEARS + operating_total

    return TAC


# ============================================================
# 섹션 5: 숏컷 컬럼 Solver
# ============================================================

def solve_column(feed_flows, feed_T, feed_P, lk_idx, hk_idx, R_over_Rmin):
    """
    FUG 방법으로 증류 컬럼을 계산하는 메인 함수.

    Args:
        feed_flows: 각 성분의 몰유량 배열 (mol/s)
        feed_T: 피드 온도 (°C)
        feed_P: 피드 압력 (Pa)
        lk_idx: Light Key 성분 인덱스
        hk_idx: Heavy Key 성분 인덱스
        R_over_Rmin: 환류비 배수 (R/Rmin, 예: 1.3)

    Returns:
        success: 계산 성공 여부
        distillate_flows: distillate 유량 배열 (mol/s)
        bottoms_flows: bottoms 유량 배열 (mol/s)
        column_info: 컬럼 정보 dict (N, R, Rmin, TAC 등)
    """
    n_comp = len(feed_flows)
    F_total = np.sum(feed_flows)

    if F_total < 1e-6:
        return False, np.zeros(n_comp), np.zeros(n_comp), {}

    z_feed = feed_flows / F_total

    active = feed_flows > 1e-8
    if np.sum(active) < 2:
        return False, feed_flows.copy(), np.zeros(n_comp), {}

    T_bp = bubble_point_temperature(z_feed, feed_P, ANTOINE_PARAMS)
    alphas = get_relative_volatilities(T_bp, feed_P, ANTOINE_PARAMS, hk_idx)

    if alphas[lk_idx] <= 1.0:
        return False, np.zeros(n_comp), np.zeros(n_comp), {}

    distillate_flows = np.zeros(n_comp)
    bottoms_flows = np.zeros(n_comp)

    for i in range(n_comp):
        if feed_flows[i] < 1e-8:
            continue
        if i == lk_idx:
            distillate_flows[i] = feed_flows[i] * LK_RECOVERY
            bottoms_flows[i] = feed_flows[i] * (1.0 - LK_RECOVERY)
        elif i == hk_idx:
            distillate_flows[i] = feed_flows[i] * (1.0 - HK_RECOVERY)
            bottoms_flows[i] = feed_flows[i] * HK_RECOVERY
        elif alphas[i] > alphas[lk_idx]:
            distillate_flows[i] = feed_flows[i]
        else:
            bottoms_flows[i] = feed_flows[i]

    D_total = np.sum(distillate_flows)
    B_total = np.sum(bottoms_flows)

    if D_total < 1e-6 or B_total < 1e-6:
        return False, np.zeros(n_comp), np.zeros(n_comp), {}

    xD = distillate_flows / D_total

    Nmin = fenske_min_stages(alphas[lk_idx], LK_RECOVERY, HK_RECOVERY)
    Rmin = underwood_min_reflux(alphas, z_feed, xD, lk_idx, hk_idx, q=1.0)
    R = R_over_Rmin * Rmin
    N_actual = gilliland_actual_stages(Nmin, R, Rmin)

    if N_actual > 300 or N_actual < 3:
        return False, np.zeros(n_comp), np.zeros(n_comp), {}

    avg_latent_heat = np.sum(z_feed * LATENT_HEAT)
    TAC = estimate_TAC(N_actual, R, D_total, avg_latent_heat, feed_P)

    column_info = {
        "N_min": round(Nmin, 1),
        "N_actual": N_actual,
        "R_min": round(Rmin, 3),
        "R_actual": round(R, 3),
        "R_over_Rmin": round(R_over_Rmin, 3),
        "TAC": round(TAC, 2),
        "lk_name": COMPONENT_NAMES[lk_idx],
        "hk_name": COMPONENT_NAMES[hk_idx],
        "feed_flows": feed_flows.copy(),
        "distillate_flows": distillate_flows.copy(),
        "bottoms_flows": bottoms_flows.copy(),
    }

    return True, distillate_flows, bottoms_flows, column_info


# ============================================================
# 섹션 6: 강화학습 환경 (Gym-style Tree-MDP)
# ============================================================

class HydrocarbonDistillationEnv:
    """
    6성분 경질 탄화수소 증류 시퀀싱을 위한 강화학습 환경.

    BTX (3성분)과 동일한 Tree-MDP 구조이지만:
      - State: 6차원 (정규화된 유량 [z_C2, z_C3, z_iC4, z_nC4, z_iC5, z_nC5])
      - Action: 이산(split point: 0~4) + 연속(R/Rmin: 1.05~3.0)
      - 최대 5개 컬럼 필요 (vs BTX는 2개)
      - 42가지 가능한 시퀀스 (vs BTX는 2가지)
    """

    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.feed_flows = FEED_FLOWS.copy()
        self.feed_T = FEED_TEMPERATURE
        self.feed_P = FEED_PRESSURE
        self.feed_total = FEED_FLOW_TOTAL
        self.state_dim = N_COMPONENTS

        # reward normalization
        self.max_total_revenue = sum(
            FEED_FLOWS[i] * SALES_PRICES[i] * ANNUAL_OPERATING_SECONDS
            for i in range(N_COMPONENTS)
        )
        self.reward_norm = self.max_total_revenue * 0.06

        self.stream_queue = deque()
        self.products = []
        self.columns = []
        self.total_revenue = 0.0
        self.total_TAC = 0.0
        self.episode_reward = 0.0

    def reset(self):
        self.stream_queue = deque()
        self.stream_queue.append({
            "flows": self.feed_flows.copy(),
            "T": self.feed_T,
            "P": self.feed_P,
        })
        self.products = []
        self.columns = []
        self.total_revenue = 0.0
        self.total_TAC = 0.0
        self.episode_reward = 0.0
        return self._get_state()

    def step(self, discrete_action, continuous_action):
        """
        한 단계 실행.

        Args:
            discrete_action: split point 인덱스 (0~4)
            continuous_action: R/Rmin 배수 (1.05 ~ 3.0)

        Returns: (next_state, reward, done, info)
        """
        current = self.stream_queue.popleft()
        feed_flows = current["flows"]
        feed_T = current["T"]
        feed_P = current["P"]

        stream_total = np.sum(feed_flows)
        threshold = stream_total * 0.01
        active_components = np.where(feed_flows > threshold)[0]
        n_active = len(active_components)

        if n_active <= 1:
            self._register_product(feed_flows)
            done = len(self.stream_queue) == 0
            state = self._get_state() if not done else np.zeros(self.state_dim)
            return state, 0.0, done, {"auto_submit": True}

        split = int(discrete_action)
        if split >= n_active - 1:
            split = 0

        lk_idx = active_components[split]
        hk_idx = active_components[split + 1]

        R_over_Rmin = float(np.clip(continuous_action, 1.05, 3.0))

        success, dist_flows, bot_flows, col_info = solve_column(
            feed_flows, feed_T, feed_P, lk_idx, hk_idx, R_over_Rmin
        )

        if not success:
            self._register_product(feed_flows)
            reward = -FAIL_PENALTY / self.reward_norm
            self.episode_reward += reward
            done = len(self.stream_queue) == 0
            state = self._get_state() if not done else np.zeros(self.state_dim)
            return state, reward, done, {"failed": True}

        self.columns.append(col_info)
        TAC = col_info["TAC"]
        self.total_TAC += TAC

        revenue_this_step = 0.0
        for stream_flows in [dist_flows, bot_flows]:
            s_total = np.sum(stream_flows)
            if s_total < 0.1:
                continue
            max_frac = np.max(stream_flows) / s_total
            if max_frac >= REQUIRED_PURITY:
                rev = self._register_product(stream_flows)
                revenue_this_step += rev
            else:
                z = stream_flows / s_total
                T_est = bubble_point_temperature(z, feed_P, ANTOINE_PARAMS)
                self.stream_queue.append({
                    "flows": stream_flows,
                    "T": T_est,
                    "P": feed_P,
                })

        self.total_revenue += revenue_this_step

        reward = (revenue_this_step - TAC) / self.reward_norm
        self.episode_reward += reward

        done = len(self.stream_queue) == 0
        state = self._get_state() if not done else np.zeros(self.state_dim)
        info = {
            "revenue": revenue_this_step,
            "TAC": TAC,
            "N_actual": col_info["N_actual"],
            "R_actual": col_info["R_actual"],
        }
        return state, reward, done, info

    def _get_state(self):
        if len(self.stream_queue) == 0:
            return np.zeros(self.state_dim)
        current = self.stream_queue[0]
        state = current["flows"] / self.feed_total
        return state.astype(np.float32)

    def _register_product(self, stream_flows):
        stream_total = np.sum(stream_flows)
        if stream_total < 0.01:
            self.products.append({"flows": stream_flows, "revenue": 0.0, "purity": 0.0})
            return 0.0
        max_idx = np.argmax(stream_flows)
        purity = stream_flows[max_idx] / stream_total
        if purity >= REQUIRED_PURITY:
            revenue = stream_flows[max_idx] * SALES_PRICES[max_idx] * ANNUAL_OPERATING_SECONDS
        else:
            revenue = 0.0
        self.products.append({
            "flows": stream_flows.copy(),
            "revenue": revenue,
            "purity": round(purity, 4),
            "main_component": COMPONENT_NAMES[max_idx],
        })
        return revenue

    def get_episode_summary(self):
        return {
            "columns": self.columns,
            "products": self.products,
            "total_revenue": self.total_revenue,
            "total_TAC": self.total_TAC,
            "profit": self.total_revenue - self.total_TAC,
            "episode_reward": self.episode_reward,
            "n_columns": len(self.columns),
        }
