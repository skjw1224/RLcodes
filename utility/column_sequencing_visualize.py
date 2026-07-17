"""
6성분 경질 탄화수소 증류 시퀀싱 - 시각화 모듈
==============================================
BFD(Block Flow Diagram)와 학습 커브 시각화를 담당한다.
pydot(graphviz)을 사용하여 그래프 기반의 BFD를 생성한다.

학습 코드와 분리하여 교육생은 학습 로직에만 집중할 수 있다.
"""
import numpy as np
import pydot
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman"]


# ============================================================
# BFD 시각화 (pydot 기반, Utils/BFD_maker.py 스타일)
# ============================================================

def _build_topology(summary):
    """
    summary에서 컬럼-제품 간의 연결 관계를 추출한다.

    각 컬럼의 distillate/bottoms가 다음 컬럼의 feed와 일치하는지,
    또는 제품의 flows와 일치하는지를 매칭하여 그래프 edge를 구성한다.

    Returns:
        col_edges: [(from_col, from_port, to_col), ...]
        prod_edges: [(from_col, from_port, prod_idx), ...]
    """
    columns = summary["columns"]
    products = summary["products"]

    # 각 컬럼 출력을 추적
    outputs = []
    for i, col in enumerate(columns):
        outputs.append({"col": i, "port": "top", "flows": col["distillate_flows"], "used": False})
        outputs.append({"col": i, "port": "bot", "flows": col["bottoms_flows"], "used": False})

    # 컬럼 입력 매칭 (column 0은 항상 feed)
    col_edges = []
    for i in range(1, len(columns)):
        for item in outputs:
            if item["used"] or item["col"] >= i:
                continue
            if np.allclose(columns[i]["feed_flows"], item["flows"], atol=1e-4):
                col_edges.append((item["col"], item["port"], i))
                item["used"] = True
                break

    # 제품 매칭
    prod_edges = []
    for j, prod in enumerate(products):
        for item in outputs:
            if item["used"]:
                continue
            if np.allclose(prod["flows"], item["flows"], atol=1e-4):
                prod_edges.append((item["col"], item["port"], j))
                item["used"] = True
                break

    return col_edges, prod_edges


def _flow_label(flows, component_names):
    """유량 배열에서 유효 성분만 한 줄 문자열로 요약한다."""
    parts = []
    for name, flow in zip(component_names, flows):
        if flow > 0.01:
            parts.append(f"{name} {flow:.1f} mol/s")
    return "\n".join(parts)


def _split_label_html(col, component_names):
    """컬럼의 distillate/bottoms 분리를 HTML 라벨로 세로 표시한다.

    feed에 유의미하게 존재하는 성분만 대상으로,
    각 성분을 distillate/bottoms 중 더 많이 가는 쪽에 배정한다.
    """
    feed = col["feed_flows"]
    dist = col["distillate_flows"]
    bot = col["bottoms_flows"]
    total_feed = sum(feed)
    dist_comps, bot_comps = [], []
    for name, ff, df, bf in zip(component_names, feed, dist, bot):
        if total_feed > 0 and ff / total_feed < 0.01:
            continue
        if df >= bf:
            dist_comps.append(name)
        else:
            bot_comps.append(name)
    dist_str = "<BR/>".join(dist_comps) if dist_comps else "-"
    bot_str = "<BR/>".join(bot_comps) if bot_comps else "-"
    max_name = max(len(n) for n in dist_comps + bot_comps) if dist_comps + bot_comps else 1
    sep = "-" * max_name
    return f"{dist_str}<BR/>{sep}<BR/>{bot_str}<BR/>"


def draw_bfd(summary, component_names, feed_flows, feed_temperature, feed_pressure,
             save_path="optimal_bfd.png"):
    """
    pydot 기반 BFD 시각화.
    Utils/BFD_maker.py의 그래프 방식을 기반으로,
    tutorial 환경의 summary 형식에 맞게 구성했다.

    Args:
        summary: env.get_episode_summary() 반환값
        component_names: 성분 이름 리스트
        feed_flows: 피드 유량 배열
        feed_temperature: 피드 온도 (°C)
        feed_pressure: 피드 압력 (Pa)
        save_path: 저장 경로
    """
    columns = summary["columns"]
    products = summary["products"]
    col_edges, prod_edges = _build_topology(summary)

    # 경제성 요약 (그래프 하단 라벨)
    econ_label = (
        f"Revenue: ${summary['total_revenue']:,.0f}/yr  |  "
        f"TAC: ${summary['total_TAC']:,.0f}/yr  |  "
        f"Profit: ${summary['profit']:,.0f}/yr  |  "
        f"Columns: {summary['n_columns']}"
    )

    G = pydot.Dot(
        graph_type="digraph", rankdir="LR",
        bgcolor="white", fontname="Times-Bold",
        nodesep="0.6", ranksep="1.5",
        label=econ_label, labelloc="b", labeljust="c", fontsize="20",
    )

    # --- Feed 노드 ---
    P_atm = feed_pressure / 101325.0
    flow_lines = "<BR/>".join(
        f"{name} {flow:.1f} mol/s"
        for name, flow in zip(component_names, feed_flows) if flow > 0.01
    )
    feed_label = (
        f'<<FONT POINT-SIZE="18"><B>FEED</B></FONT>'
        f'<BR/><BR/>'
        f'<FONT POINT-SIZE="14">{flow_lines}'
        f'<BR/>{feed_temperature} degC, {P_atm:.0f} atm</FONT>>'
    )
    feed_node = pydot.Node(
        "feed", label=feed_label, shape="box", style="filled",
        fillcolor="#FFE66D", fontname="Times-Bold",
    )
    G.add_node(feed_node)

    # --- 컬럼 노드 ---
    for i, col in enumerate(columns):
        split = _split_label_html(col, component_names)
        label = (
            f'<<FONT POINT-SIZE="17"><B>Column {i + 1}</B></FONT>'
            f'<BR/><BR/>'
            f'<FONT POINT-SIZE="13">{split}<BR/>'
            f'N={col["N_actual"]} stages<BR/>'
            f'R={col["R_actual"]:.2f} (R/Rmin={col["R_over_Rmin"]:.2f})<BR/>'
            f'TAC ${col["TAC"] / 1e6:.2f}M/yr</FONT>>'
        )
        node = pydot.Node(
            f"col_{i}", label=label, shape="box", style="filled",
            fillcolor="#B0E0E6", fontname="Times-Bold",
        )
        G.add_node(node)

    # --- 제품 노드 ---
    for j, prod in enumerate(products):
        flows = prod["flows"]
        total = np.sum(flows)
        name = prod.get("main_component", "?")
        purity = prod.get("purity", 0)
        rev = prod.get("revenue", 0)

        if total > 0.01 and rev > 0:
            label = (
                f'<<FONT POINT-SIZE="17"><B>{name}</B></FONT>'
                f'<BR/><BR/>'
                f'<FONT POINT-SIZE="13">Purity {purity:.1%}<BR/>'
                f'{total:.1f} mol/s<BR/>'
                f'${rev / 1e6:.2f}M/yr</FONT>>'
            )
            color = "#98FB98"
        elif total > 0.01:
            label = (
                f'<<FONT POINT-SIZE="17"><B>Mixed</B></FONT>'
                f'<BR/><BR/>'
                f'<FONT POINT-SIZE="13">{total:.1f} mol/s<BR/>'
                f'$0/yr</FONT>>'
            )
            color = "#FFB3B3"
        else:
            label = '<<FONT POINT-SIZE="13">negligible</FONT>>'
            color = "#DDDDDD"

        node = pydot.Node(
            f"prod_{j}", label=label, shape="box", style="filled",
            fillcolor=color, fontname="Times-Bold",
        )
        G.add_node(node)

    # --- Edge: Feed → Column 1 ---
    if columns:
        G.add_edge(pydot.Edge("feed", "col_0", headport="w", tailport="e"))

    # --- Edge: Column → Column ---
    for from_col, from_port, to_col in col_edges:
        tailport = "ne" if from_port == "top" else "se"
        edge_label = "top" if from_port == "top" else "bot"
        G.add_edge(pydot.Edge(
            f"col_{from_col}", f"col_{to_col}",
            headport="w", tailport=tailport,
            label=f" {edge_label} ", fontname="Times-Bold", fontsize="15",
        ))

    # --- Edge: Column → Product ---
    for from_col, from_port, prod_idx in prod_edges:
        tailport = "ne" if from_port == "top" else "se"
        edge_label = "top" if from_port == "top" else "bot"
        G.add_edge(pydot.Edge(
            f"col_{from_col}", f"prod_{prod_idx}",
            headport="w", tailport=tailport,
            label=f" {edge_label} ", fontname="Times-Bold", fontsize="15",
        ))

    G.write_png(save_path, prog=["dot", "-Gdpi=600"])
    print(f"  BFD saved to {save_path}")


# ============================================================
# 학습 커브 시각화
# ============================================================

def plot_learning_curves(returns_dict, save_path="learning_curves.png", window=50):
    """
    학습 커브를 시각화한다.

    Args:
        returns_dict: {"라벨": [episode_returns], ...}
        save_path: 저장 경로
        window: 이동 평균 윈도우 크기
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

    ax = axes[0]
    for idx, (label, returns) in enumerate(returns_dict.items()):
        c = colors[idx % len(colors)]
        episodes = range(1, len(returns) + 1)
        ax.plot(episodes, returns, alpha=0.15, color=c)
        if len(returns) >= window:
            ma = np.convolve(returns, np.ones(window) / window, mode="valid")
            ax.plot(range(window, len(returns) + 1), ma,
                    color=c, linewidth=2, label=f"{label} (avg)")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Return", fontsize=12)
    ax.set_title("Learning Curves", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    labels = list(returns_dict.keys())
    final_avgs, final_stds = [], []
    for label in labels:
        r = returns_dict[label]
        last_n = r[-min(100, len(r)):]
        final_avgs.append(np.mean(last_n))
        final_stds.append(np.std(last_n))
    ax.bar(labels, final_avgs, yerr=final_stds, capsize=5,
           color=colors[:len(labels)], edgecolor="black", alpha=0.8)
    ax.set_ylabel("Average Return (last 100 eps)", fontsize=12)
    ax.set_title("Final Performance", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Learning curves saved to {save_path}")


def plot_parallel_comparison(returns_single, ts_single, returns_parallel, ts_parallel,
                             time_single, time_parallel, n_workers,
                             save_path="parallel_comparison.png", window=50):
    """
    단일 vs 병렬 학습 비교 시각화.

    Args:
        returns_single, ts_single: 단일 프로세스 결과
        returns_parallel, ts_parallel: 병렬 결과
        time_single, time_parallel: 각각의 총 소요 시간
        n_workers: worker 수
        save_path: 저장 경로
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 학습 커브 (x축: wall-clock time)
    ax = axes[0]
    for label, returns, ts, color in [
        ("Single Process", returns_single, np.array(ts_single), "#FF6B6B"),
        (f"Ray {n_workers} Workers", returns_parallel, np.array(ts_parallel), "#4ECDC4"),
    ]:
        ax.plot(ts, returns, alpha=0.15, color=color)
        if len(returns) >= window:
            ma = np.convolve(returns, np.ones(window) / window, mode="valid")
            ts_ma = ts[window - 1:]
            ax.plot(ts_ma, ma, color=color, linewidth=2, label=f"{label} (avg)")
    ax.set_xlabel("Wall-Clock Time (seconds)")
    ax.set_ylabel("Episode Return")
    ax.set_title("Learning Curves (same batch size)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 시간 비교
    ax = axes[1]
    bars = ax.bar(["Single", f"Ray\n{n_workers}w"],
                  [time_single, time_parallel],
                  color=["#FF6B6B", "#4ECDC4"], edgecolor="black", width=0.5)
    for bar, t in zip(bars, [time_single, time_parallel]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{t:.1f}s", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Training Time (seconds)")
    ax.set_title("Wall-Clock Time")
    ax.grid(True, alpha=0.3, axis="y")

    # 최종 성능
    ax = axes[2]
    ax.bar(["Single", f"Ray\n{n_workers}w"],
           [np.mean(returns_single[-100:]), np.mean(returns_parallel[-100:])],
           yerr=[np.std(returns_single[-100:]), np.std(returns_parallel[-100:])],
           capsize=5, color=["#FF6B6B", "#4ECDC4"], edgecolor="black", width=0.5)
    ax.set_ylabel("Avg Return (last 100)")
    ax.set_title("Final Performance")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Parallel comparison saved to {save_path}")
