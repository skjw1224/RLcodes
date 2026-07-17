"""
6성분 경질 탄화수소 증류 시퀀싱 환경 설정
==========================================
파일: 물성, 가격, 피드 조건 등 모든 설정값

Luyben (2011) 예제를 기반으로 한 6성분 분리 문제.
  - 6성분 → 5가지 split point → 최대 5개 컬럼
  - 가능한 시퀀스: Catalan(5) = 42가지
  - 비점이 가까운 쌍 (isobutane/n-butane) → 어려운 분리
  - 고압 운전 (10 atm) → 경질 탄화수소의 액화
"""
import numpy as np

# ============================================================
# 성분 정보
# ============================================================
COMPONENT_NAMES = ["Ethane", "Propane", "Isobutane", "n-Butane",
                   "Isopentane", "n-Pentane"]
N_COMPONENTS = 6

# 분자량 (g/mol)
MOLAR_MASS = np.array([30.07, 44.097, 58.124, 58.124, 72.151, 72.151])

# Antoine 계수: log10(Psat[mmHg]) = A - B / (C + T[°C])
# 출처: Perry's Chemical Engineers' Handbook / NIST
ANTOINE = {
    "Ethane":     {"A": 6.80266, "B": 656.400, "C": 256.000},
    "Propane":    {"A": 6.82107, "B": 803.810, "C": 247.040},
    "Isobutane":  {"A": 6.91048, "B": 946.350, "C": 246.680},
    "n-Butane":   {"A": 6.82485, "B": 943.453, "C": 239.711},
    "Isopentane": {"A": 6.78967, "B": 1020.012, "C": 233.097},
    "n-Pentane":  {"A": 6.87632, "B": 1075.780, "C": 233.205},
}
ANTOINE_PARAMS = [
    (6.80266, 656.400, 256.000),   # Ethane
    (6.82107, 803.810, 247.040),   # Propane
    (6.91048, 946.350, 246.680),   # Isobutane
    (6.82485, 943.453, 239.711),   # n-Butane
    (6.78967, 1020.012, 233.097),  # Isopentane
    (6.87632, 1075.780, 233.205),  # n-Pentane
]

# 평균 몰 증발 잠열 (kJ/mol) - 정상 끓는점 기준
LATENT_HEAT = np.array([14.7, 19.0, 21.3, 22.4, 25.8, 26.4])

# ============================================================
# 경제성 파라미터
# ============================================================
# 판매가격 ($/mol) - 원래 코드 CONFIG 0 (LuybenPart) 참고
#   Price_per_MBTU / MJ_per_MBTU * Heating_value * Molar_weight/1000
Heating_value = np.array([51.9, 50.4, 49.5, 49.4, 55.2, 55.2])   # MJ/kg
Price_per_MBTU = np.array([2.54, 4.27, 5.79, 5.31, 10.41, 10.41])  # $/Million BTU
MJ_per_MBTU = 1055.06
SALES_PRICES = Price_per_MBTU / MJ_per_MBTU * Heating_value * (MOLAR_MASS / 1000.0)
# 결과 ($/mol): [0.00376, 0.00899, 0.01579, 0.01445, 0.03929, 0.03929]

ANNUAL_OPERATING_HOURS = 8000
ANNUAL_OPERATING_SECONDS = ANNUAL_OPERATING_HOURS * 3600  # 28,800,000 s/year

PAYBACK_YEARS = 3

# Capital cost 보정 계수 (M&S index 보정 + 설치비 multiplier)
CAPITAL_FACTOR = 120

# 유틸리티 비용
STEAM_COST = 15e-6    # $/kJ
COOLING_COST = 5e-6   # $/kJ

# ============================================================
# 피드 조건
# ============================================================
FEED_FLOW_TOTAL = 100.0   # mol/s
FEED_COMPOSITION = np.array([0.05, 0.15, 0.20, 0.25, 0.20, 0.15])
FEED_FLOWS = FEED_FLOW_TOTAL * FEED_COMPOSITION  # [5, 15, 20, 25, 20, 15] mol/s
FEED_TEMPERATURE = 50.0    # °C (elevated pressure에서 액체 상태)
FEED_PRESSURE = 1013250.0  # Pa (10 atm) - 경질 탄화수소는 고압 운전

# 단당 고정 설치비 ($/stage) - 배관, 계장, 구조물 등
STAGE_FIXED_COST = 800.0

# ============================================================
# RL 환경 파라미터
# ============================================================
REQUIRED_PURITY = 0.95     # 제품 순도 요건
FAIL_PENALTY = 0.5
LK_RECOVERY = 0.99         # Light key의 distillate 회수율
HK_RECOVERY = 0.99         # Heavy key의 bottoms 회수율
