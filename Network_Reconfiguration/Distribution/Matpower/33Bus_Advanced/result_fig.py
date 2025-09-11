import os
import pandas as pd
import numpy as np
from collections import defaultdict, deque, Counter
import matplotlib as mpl
import matplotlib.pyplot as plt
from config import switch, dg_case, pv_penetration

TIME_INDEX = 15  # 원하는 시간 인덱스

## Set directory
save_directory = os.path.dirname(__file__) + "/Pre_cal_data/"       # Set save parameter directory
output_directory = os.path.dirname(__file__) + "/Output_data/"     # Set output directory

if dg_case == 'none':
    if switch == 1:
        simul_case = '33bus_MINLP_Opt_problem_for_min_cost_with_switch_' + dg_case + '_dgs_'
    elif switch == 0:
        simul_case = '33bus_MINLP_Opt_problem_for_min_cost_without_switch_' + dg_case + '_dgs_'
else:
    if switch == 1:
        simul_case = '33bus_MINLP_Opt_problem_for_min_cost_with_switch_' + dg_case + '_dgs_' + str(pv_penetration) + '_pv_penetration_'
    elif switch == 0:
        simul_case = '33bus_MINLP_Opt_problem_for_min_cost_without_switch_' + dg_case + '_dgs_' + str(pv_penetration) + '_pv_penetration_'

# DG list
dg_list = []

if dg_case == 'none':
    dg_bus_list = []
else:
    list_dg = pd.read_excel(save_directory + f'DG_Candidates_{dg_case}.xlsx', sheet_name='Candidate')
    list_dg_df = pd.DataFrame(list_dg, columns=['Index','Bus number', 'Rating[MW]', 'Type', 'Profile'])
    dg_bus_list = list(list_dg_df['Bus number'].astype(int).unique())

# Ensure output subdirectories exist
os.makedirs(os.path.join(output_directory, "Variables"), exist_ok=True)
os.makedirs(os.path.join(output_directory, "Dual"), exist_ok=True)

# ====== 추가: Figure 저장 폴더 생성 ======
output_fig_directory = os.path.join(output_directory, "Figures", simul_case + "Fig")
os.makedirs(output_fig_directory, exist_ok=True)

VAR_XLSX = os.path.join(output_directory, "Variables", f"{simul_case}Variables.xlsx")
LINE_INFO = os.path.join(save_directory, "Line_info.csv")
LOAD_INFO = os.path.join(save_directory, "Load_info.csv")
SYSTEM_XLSX = os.path.join(save_directory, "System.xlsx")
FIG_DIR = os.path.join(output_directory, "Figures", simul_case + "Fig")
os.makedirs(FIG_DIR, exist_ok=True)

OUT_PNG_S = os.path.join(output_fig_directory, f"Tree_layout.png")
OUT_PNG_NL = os.path.join(output_fig_directory, f"Tree_layout_Netload.png")
OUT_PNG_V = os.path.join(output_fig_directory, f"Tree_layout_Vmag.png")
OUT_PNG_P_PF = os.path.join(output_fig_directory, f"Tree_layout_P_PowerFlow.png")
OUT_PNG_Q_PF = os.path.join(output_fig_directory, f"Tree_layout_Q_PowerFlow.png")

def set_png_output(case):
    OUT_PNG = os.path.join(FIG_DIR, f"{case}.png")
    return OUT_PNG

bus_info = pd.read_csv(save_directory + 'Bus_info.csv')

for b in bus_info.iterrows():
    if bus_info.loc[b[0],'Type'] == 3:
        Slackbus = bus_info.loc[b[0],'Buses']



# ===================== 데이터 로드 =====================
# 1) Line_Status: 활성 선로 플래그
line_status_df = pd.read_excel(VAR_XLSX, sheet_name="Line_Status")

# 안전을 위해 타입 정리
line_status_df["Index: Lines"] = line_status_df["Index: Lines"].astype(int)
line_status_df["Value"] = line_status_df["Value"].round().astype(int)
# 2) 선로 정보
line_info = pd.read_csv(LINE_INFO)
required_cols = {"Line_l", "from_bus", "to_bus"}
if not required_cols.issubset(set(line_info.columns)):
    raise ValueError(f"Line_info.xlsx에 {required_cols} 컬럼이 필요합니다.")
line_info["Line_l"] = line_info["Line_l"].astype(int)

# 3) Merge: Line_l(좌) ↔ Index: Lines(우)
merged_df = pd.merge(
    line_info,
    line_status_df,
    left_on="Line_l",
    right_on="Index: Lines",
    how="left",
)
# Value 컬럼의 NaN(빈 값)을 0으로 채우기
merged_df["Value"] = merged_df["Value"].fillna(0).astype(int)

# 필요한 컬럼만 정리
result_df = merged_df[["Line_l", "from_bus", "to_bus", "Value"]].copy()
result_df = result_df.rename(columns={"Line_l": "line_index", "Value": "line_status"})

# (선택) 결과 저장
try:
    result_path = os.path.join(output_directory, "result_df.xlsx")
except NameError:
    result_path = "result_df.xlsx"
result_df.to_excel(result_path, index=False)
print(f"Saved: {result_path}")

# 4) 활성 간선(branches) 만들기
active = result_df[result_df["line_status"].astype(int) >= 1].copy()
if active.empty:
    raise ValueError("활성화된 선로(line_status == 1)가 없습니다.")

# from/to를 int로 보장
active["from_bus"] = active["from_bus"].astype(int)
active["to_bus"] = active["to_bus"].astype(int)
branches = list(zip(active["from_bus"], active["to_bus"]))

# 5) (선택) 기존 좌표 힌트: 자식 정렬에만 사용
pos_hint = None
if os.path.exists(SYSTEM_XLSX):
    sysdf = pd.read_excel(SYSTEM_XLSX, sheet_name="33bus", usecols=["Bus", "x", "y"])
    pos_hint = {int(b): (float(x), float(y)) for b, x, y in zip(sysdf["Bus"], sysdf["x"], sysdf["y"])}

# ===================== 그래프 유틸 =====================
def build_components(edges):
    """무방향 그래프에서 컴포넌트 분해."""
    G = defaultdict(list)
    nodes = set()
    for u, v in edges:
        G[u].append(v); G[v].append(u)
        nodes.update([u, v])
    seen = set()
    comps = []
    for s in sorted(nodes):
        if s in seen:
            continue
        q = deque([s]); seen.add(s)
        comp_nodes = [s]
        comp_edges = []
        while q:
            u = q.popleft()
            for w in G[u]:
                comp_edges.append((u, w))
                if w not in seen:
                    seen.add(w); q.append(w); comp_nodes.append(w)
        # 무방향이라 간선 중복 제거
        comp_edges = list({tuple(sorted(e)) for e in comp_edges})
        comps.append((sorted(set(comp_nodes)), comp_edges))
    return comps

def assert_tree(nodes, edges):
    """사이클/비연결 검사. 트리면 True, 아니면 에러."""
    n = len(nodes)
    m = len(edges)
    if m != n - 1:
        raise ValueError(f"트리 조건 위반: 노드 {n}개, 간선 {m}개 (트리는 간선=n-1).")
    G = defaultdict(list)
    for u, v in edges:
        G[u].append(v); G[v].append(u)
    root = min(nodes)
    parent = {root: None}
    seen = {root}
    q = deque([root])
    while q:
        u = q.popleft()
        for w in G[u]:
            if w == parent[u]:
                continue
            if w in seen:
                raise ValueError("사이클이 존재합니다. 방사형(트리)이 아닙니다.")
            seen.add(w); parent[w] = u; q.append(w)
    if len(seen) != n:
        raise ValueError("비연결 컴포넌트가 존재합니다.")
    return True

# ===================== Tidy Tree 레이아웃 =====================
def tidy_tree_layout(nodes, edges, pos_hint=None, root=None, level_gap=1.6, sibling_gap=1.0):
    """Reingold–Tilford 스타일 간단 구현 (부모=자식 중앙, 레벨별 간격)."""
    G = defaultdict(list)
    for u, v in edges:
        G[u].append(v); G[v].append(u)

    # 루트: degree=1 리프 중 가장 작은 번호 선호
    if root is None:
        leaves = [n for n in nodes if len(G[n]) == 1]
        root = min(leaves) if leaves else min(nodes)

    # 트리화
    parent = {root: None}
    children = defaultdict(list)
    depth = {root: 0}
    q = deque([root])
    order = [root]
    seen = {root}
    while q:
        u = q.popleft()
        for w in G[u]:
            if w == parent[u]:
                continue
            parent[w] = u
            children[u].append(w)
            depth[w] = depth[u] + 1
            q.append(w)
            order.append(w)
            seen.add(w)

    # 자식 정렬: pos_hint.x 기준(없으면 번호)
    def child_sort_key(n):
        if pos_hint is not None and n in pos_hint:
            return (pos_hint[n][0], n)
        return (n, n)

    def sort_children(u):
        ch = children[u]
        ch.sort(key=child_sort_key)
        for w in ch:
            sort_children(w)
    sort_children(root)

    # 레벨별 배치
    x = {}
    y = {n: depth[n] * level_gap for n in depth}
    next_x = defaultdict(lambda: 0.0)

    subtree_nodes = defaultdict(list)
    def collect(u):
        acc = [u]
        for w in children[u]:
            acc += collect(w)
        subtree_nodes[u] = acc
        return acc
    collect(root)

    def shift_subtree(u, dx):
        for n in subtree_nodes[u]:
            x[n] = x.get(n, 0.0) + dx

    def layout(u):
        if not children[u]:
            x[u] = max(next_x[depth[u]], x.get(u, 0.0))
            next_x[depth[u]] = x[u] + sibling_gap
        else:
            for w in children[u]:
                layout(w)
            cx = (x[children[u][0]] + x[children[u][-1]]) / 2.0
            x[u] = cx
            if x[u] < next_x[depth[u]]:
                delta = next_x[depth[u]] - x[u]
                shift_subtree(u, delta)
                x[u] += delta
            next_x[depth[u]] = x[u] + sibling_gap

    layout(root)
    return {n: (x[n], y[n]) for n in x}, children, root

# ===================== 메인: 포레스트 처리 =====================
components = build_components(branches)

# 각 컴포넌트가 트리인지 확인
for nodes, edges in components:
    assert_tree(nodes, edges)

# 각 트리를 나란히 배치
COMP_GAP_X = 4.0
global_pos = {}
x_offset = 0.0
all_children = {}
roots = []

for nodes, edges in components:
    pos_comp, children, root = tidy_tree_layout(
        nodes=set(nodes),
        edges=edges,
        pos_hint=pos_hint,
        root=None,
        level_gap=1.6,
        sibling_gap=1.0
    )
    # 보기 좋게 회전/반전
    pos_comp = {n: (y, -x) for n, (x, y) in pos_comp.items()}

    # 컴포넌트 간 가로 오프셋
    min_x = min(px for px, _ in pos_comp.values())
    shifted = {n: (px - min_x + x_offset, py) for n, (px, py) in pos_comp.items()}
    global_pos.update(shifted)
    all_children.update(children)
    roots.append(root)
    max_x = max(px for px, _ in shifted.values())
    x_offset = max_x + COMP_GAP_X

# ===================== 계통 플롯 ===================== 
fig, ax = plt.subplots(figsize=(16, 8))

for u in all_children:
    for v in all_children[u]:
        x1, y1 = global_pos[u]; x2, y2 = global_pos[v]
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=1)

for n, (xp, yp) in global_pos.items():
    if n in dg_bus_list:
        ax.plot(xp, yp, 'o', color='red', markersize=10, markeredgecolor='red', zorder=3)
    elif n == Slackbus:
        ax.plot(xp, yp, 'o', color='blue', markersize=10, markeredgecolor='blue', zorder=2)
    else:
        ax.plot(xp, yp, 'o', color='black', markersize=10, markeredgecolor='black', zorder=2)

    ax.text(xp, yp-0.2, str(n), ha='center', va='center', fontweight='bold', fontsize=9)  # 버스 번호만 표시

ax.axis('off')  # x, y축 안보이게
plt.title(f'System Diagram')
plt.savefig(OUT_PNG_S, bbox_inches='tight', dpi=300)
print(f"Saved: {OUT_PNG_S}")

# ===================== 전압 플롯 =====================
fig_v, ax_v = plt.subplots(figsize=(12, 8))

# ---- 전압 데이터 불러오기 및 색상 매핑 ----
V_MAG_XLSX = os.path.join(output_directory, "Variables", f"{simul_case}Variables.xlsx")
V_MAG_SHEET = "V_mag"


v_mag_df = pd.read_excel(V_MAG_XLSX, sheet_name=V_MAG_SHEET)
required_vmag_cols = {"Index: Times", "Index: Buses", "Value"}
if not required_vmag_cols.issubset(set(v_mag_df.columns)):
    raise ValueError(f"V_mag 시트에 {required_vmag_cols} 컬럼이 필요합니다.")

v_row = v_mag_df[v_mag_df["Index: Times"] == TIME_INDEX]
if v_row.empty:
    raise ValueError(f"V_mag 시트에 Index: Times={TIME_INDEX} 행이 없습니다.")
bus_v = {int(row["Index: Buses"]): float(row["Value"]) for _, row in v_row.iterrows()}

v_values = list(bus_v.values())
vmin, vmax = min(v_values), max(v_values)
cmap_v = mpl.colormaps['plasma']
norm_v = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

# 간선(부모→자식)
for u in all_children:
    for v in all_children[u]:
        x1, y1 = global_pos[u]; x2, y2 = global_pos[v]
        ax_v.plot([x1, x2], [y1, y2], color="black", linewidth=1)

deg = Counter()
for u, v in branches:
    deg[u] += 1
    deg[v] += 1
leaf_nodes = sorted([n for n, d in deg.items() if d == 1])
max_v_nodes = [n for n, v in bus_v.items() if v == vmax]
min_v_nodes = [n for n, v in bus_v.items() if v == vmin]

for n, (xp, yp) in global_pos.items():
    v = bus_v.get(n, vmax)
    color = cmap_v(norm_v(v))
    #ax_v.plot(xp, yp, 'o', color=color, markersize=10, markeredgecolor='black')
    if n in dg_bus_list:
        ax_v.plot(xp, yp, 'o', color=color, markersize=12, markeredgecolor='red', zorder=3)
    else:
        ax_v.plot(xp, yp, 'o', color=color, markersize=10, markeredgecolor='black', zorder=2)
    ax_v.text(xp, yp - 0.2, str(n), ha='left', va='top', fontweight='bold', fontsize=8)
    if n in leaf_nodes or n in max_v_nodes or n in min_v_nodes or n in dg_bus_list:
        text_color = 'blue' if n in max_v_nodes else 'red' if n in min_v_nodes else 'black'
        ax_v.text(xp, yp + 0.15, f"{v:.4f}", ha='center', va='bottom', fontsize=9, color=text_color)

sm_v = mpl.cm.ScalarMappable(cmap=cmap_v, norm=norm_v)
sm_v.set_array([])
cbar_v = plt.colorbar(sm_v, ax=ax_v, orientation='vertical', fraction=0.03, pad=0.03)
cbar_v.set_label('Voltage Magnitude[pu]')

ax_v.set_aspect('equal', adjustable='datalim')
ax_v.axis('off')

plt.savefig(OUT_PNG_V, bbox_inches='tight', dpi=300)
print(f"Saved: {OUT_PNG_V}")
print("Leaf nodes:", leaf_nodes)
print("Max voltage node(s):", max_v_nodes, f"({vmax:.4f})")
print("Min voltage node(s):", min_v_nodes, f"({vmin:.4f})")

# ===================== Net_Load 플롯 =====================

fig_nl, ax_nl = plt.subplots(figsize=(12, 8))

NET_LOAD_XLSX = os.path.join(output_directory, "Variables", f"{simul_case}Variables.xlsx")
NET_LOAD_SHEET = "Net_load"

net_load_df = pd.read_excel(NET_LOAD_XLSX, sheet_name=NET_LOAD_SHEET)
required_netload_cols = {"Index: Times", "Index: Buses", "Value"}
if not required_netload_cols.issubset(set(net_load_df.columns)):
    raise ValueError(f"Net_load 시트에 {required_netload_cols} 컬럼이 필요합니다.")

nl_row = net_load_df[net_load_df["Index: Times"] == TIME_INDEX]
if nl_row.empty:
    raise ValueError(f"Net_load 시트에 Index: Times={TIME_INDEX} 행이 없습니다.")
bus_nl = {int(row["Index: Buses"]): float(row["Value"]) for _, row in nl_row.iterrows()}

nl_values = list(bus_nl.values())

# 슬랙 버스 제외한 넷로드 값으로 컬러맵 범위 설정
nl_values_wo_slack = [v for n, v in bus_nl.items() if n != Slackbus]
if nl_values_wo_slack:
    nlmin, nlmax = min(nl_values_wo_slack), max(nl_values_wo_slack)
else:
    nlmin, nlmax = min(nl_values), max(nl_values)

cmap_nl = mpl.colormaps['plasma']
norm_nl = mpl.colors.Normalize(vmin=nlmin, vmax=nlmax)

for u in all_children:
    for v in all_children[u]:
        x1, y1 = global_pos[u]; x2, y2 = global_pos[v]
        ax_nl.plot([x1, x2], [y1, y2], color="black", linewidth=1)

max_nl_nodes = [n for n, v in bus_nl.items() if np.isclose(v, nlmax, atol=1e-8)]
min_nl_nodes = [n for n, v in bus_nl.items() if np.isclose(v, nlmin, atol=1e-8)]

for n, (xp, yp) in global_pos.items():
    v = bus_nl.get(n, nlmax)
    if n == Slackbus:
        ax_nl.plot(xp, yp, 'o', markerfacecolor='black', markeredgecolor='black', markersize=10, zorder=2)
    elif n in dg_bus_list:
        ax_nl.plot(xp, yp, 'o', color=cmap_nl(norm_nl(v)), markersize=12, markeredgecolor='red', zorder=3)
    else:
        ax_nl.plot(xp, yp, 'o', color=cmap_nl(norm_nl(v)), markersize=10, markeredgecolor='black', zorder=2)
    ax_nl.text(xp, yp - 0.2, str(n), ha='left', va='top', fontweight='bold', fontsize=8)
    if n in leaf_nodes or n in max_v_nodes or n in min_v_nodes or n in dg_bus_list:
        text_color = 'red' if n in max_nl_nodes else 'blue' if n in min_nl_nodes else 'black'
        ax_nl.text(xp, yp + 0.15, f"{v:.4f}", ha='center', va='bottom', fontsize=9, color=text_color)

sm_nl = mpl.cm.ScalarMappable(cmap=cmap_nl, norm=norm_nl)
sm_nl.set_array([])
cbar_nl = plt.colorbar(sm_nl, ax=ax_nl, orientation='vertical', fraction=0.03, pad=0.03)
cbar_nl.set_label('Net Load [MW]')

ax_nl.set_aspect('equal', adjustable='datalim')
ax_nl.axis('off')
plt.savefig(OUT_PNG_NL, bbox_inches='tight', dpi=300)
print(f"Saved: {OUT_PNG_NL}")
print("Max Net_load node(s):", max_nl_nodes, f"({nlmax:.4f})")
print("Min Net_load node(s):", min_nl_nodes, f"({nlmin:.4f})")

# ===================== Active Power Flow(선로 전력) 플롯 =====================

fig_pf, ax_pf = plt.subplots(figsize=(12, 8))

P_PF_SHEET = "P_line_flow_sending"

pf_df = pd.read_excel(VAR_XLSX, sheet_name=P_PF_SHEET)
required_pf_cols = {"Index: Times", "Index: Lines", "Value"}
if not required_pf_cols.issubset(set(pf_df.columns)):
    raise ValueError(f"P_line_flow_sending 시트에 {required_pf_cols} 컬럼이 필요합니다.")

pf_row = pf_df[pf_df["Index: Times"] == TIME_INDEX]
if pf_row.empty:
    raise ValueError(f"P_line_flow_sending 시트에 Index: Times={TIME_INDEX} 행이 없습니다.")

# 활성화된 선로만 매핑
active_lines = result_df[result_df["line_status"] == 1]
active_line_map = {row["line_index"]: (row["from_bus"], row["to_bus"]) for _, row in active_lines.iterrows()}

# 선로별 전력값
line_pf = {int(row["Index: Lines"]): float(row["Value"]) for _, row in pf_row.iterrows()}

# 컬러맵 범위 설정
pf_values = [line_pf[idx] for idx in active_line_map if idx in line_pf]
if pf_values:
    pf_absmax = max(abs(min(pf_values)), abs(max(pf_values)))
else:
    pf_absmax = 1.0
cmap_pf = mpl.colormaps['bwr']  # blue-white-red
norm_pf = mpl.colors.Normalize(vmin=-pf_absmax, vmax=pf_absmax)

# 네트워크 구조 그리기 (간선)
for u in all_children:
    for v in all_children[u]:
        x1, y1 = global_pos[u]; x2, y2 = global_pos[v]
        ax_pf.plot([x1, x2], [y1, y2], color="gray", linewidth=0.8, zorder=1)

# 노드(버스) 검은색 점과 번호
for n, (xp, yp) in global_pos.items():
    if n == Slackbus:
        ax_pf.plot(xp, yp, 'o', color='blue', markersize=6, markeredgecolor='black', zorder=2)
        ax_pf.text(xp, yp + 0.18, str(n), color='blue', fontsize=5, ha='center', va='bottom', fontweight='bold', zorder=5)
    elif n in dg_bus_list:
        ax_pf.plot(xp, yp, 'o', color='red', markersize=6, markeredgecolor='black', zorder=2)
        ax_pf.text(xp, yp + 0.18, str(n), color='red', fontsize=5, ha='center', va='bottom', fontweight='bold', zorder=5)
    else:
        ax_pf.plot(xp, yp, 'o', color='black', markersize=6, markeredgecolor='black', zorder=2)
        ax_pf.text(xp, yp + 0.18, str(n), color='black', fontsize=5, ha='center', va='bottom', fontweight='bold', zorder=5)

# 컬러맵 범위 설정 (붉은색 계열로만)
pf_values = [abs(line_pf[idx]) for idx in active_line_map if idx in line_pf]
if pf_values:
    pf_absmax = max(pf_values)
else:
    pf_absmax = 1.0
cmap_pf = mpl.colormaps['Reds']  # 붉은색 계열 컬러맵
norm_pf = mpl.colors.Normalize(vmin=0, vmax=pf_absmax)

# 전력 흐름 화살표 그리기 (활성화된 선로만)
for line_idx, (u, v) in active_line_map.items():
    if line_idx not in line_pf:
        continue
    pf = line_pf[line_idx]
    if u not in global_pos or v not in global_pos:
        continue
    x1, y1 = global_pos[u]
    x2, y2 = global_pos[v]
    # 방향: 양수면 u→v, 음수면 v→u
    if pf >= 0:
        start, end = (x1, y1), (x2, y2)
    else:
        start, end = (x2, y2), (x1, y1)
    # 붉은색 컬러맵에서 절댓값으로 색상 추출
    color = cmap_pf(norm_pf(abs(pf)))
    ax_pf.annotate(
        '', xy=end, xytext=start,
        arrowprops=dict(arrowstyle='->', color=color, lw=2, shrinkA=8, shrinkB=8),
        zorder=3
    )
    mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
    ax_pf.text(mx, my-0.3, f"{abs(pf):.2f}", color='black', fontsize=8, ha='center', va='center', zorder=4)  # 양수로 표시


sm_pf = mpl.cm.ScalarMappable(cmap=cmap_pf, norm=norm_pf)
sm_pf.set_array([])
cbar_pf = plt.colorbar(sm_pf, ax=ax_pf, orientation='vertical', fraction=0.03, pad=0.03)
cbar_pf.set_label('Active Power Flow [MW]')

ax_pf.set_aspect('equal', adjustable='datalim')
ax_pf.axis('off')

plt.savefig(OUT_PNG_P_PF, bbox_inches='tight', dpi=300)
print(f"Saved: {OUT_PNG_P_PF}")

# ===================== Reactive Power Flow(선로 전력) 플롯 =====================

fig_pf, ax_pf = plt.subplots(figsize=(12, 8))

Q_PF_SHEET = "Q_line_flow_sending"

pf_df = pd.read_excel(VAR_XLSX, sheet_name=Q_PF_SHEET)
required_pf_cols = {"Index: Times", "Index: Lines", "Value"}
if not required_pf_cols.issubset(set(pf_df.columns)):
    raise ValueError(f"Q_line_flow_sending 시트에 {required_pf_cols} 컬럼이 필요합니다.")

pf_row = pf_df[pf_df["Index: Times"] == TIME_INDEX]
if pf_row.empty:
    raise ValueError(f"Q_line_flow_sending 시트에 Index: Times={TIME_INDEX} 행이 없습니다.")

# 활성화된 선로만 매핑
active_lines = result_df[result_df["line_status"] == 1]
active_line_map = {row["line_index"]: (row["from_bus"], row["to_bus"]) for _, row in active_lines.iterrows()}

# 선로별 전력값
line_pf = {int(row["Index: Lines"]): float(row["Value"]) for _, row in pf_row.iterrows()}

# 컬러맵 범위 설정
pf_values = [line_pf[idx] for idx in active_line_map if idx in line_pf]
if pf_values:
    pf_absmax = max(abs(min(pf_values)), abs(max(pf_values)))
else:
    pf_absmax = 1.0
cmap_pf = mpl.colormaps['bwr']  # blue-white-red
norm_pf = mpl.colors.Normalize(vmin=-pf_absmax, vmax=pf_absmax)

# 네트워크 구조 그리기 (간선)
for u in all_children:
    for v in all_children[u]:
        x1, y1 = global_pos[u]; x2, y2 = global_pos[v]
        ax_pf.plot([x1, x2], [y1, y2], color="gray", linewidth=0.8, zorder=1)

# 노드(버스) 검은색 점과 번호
for n, (xp, yp) in global_pos.items():
    if n == Slackbus:
        ax_pf.plot(xp, yp, 'o', color='blue', markersize=5, markeredgecolor='black', zorder=2)
        ax_pf.text(xp, yp + 0.18, str(n), color='blue', fontsize=5, ha='center', va='bottom', fontweight='bold', zorder=5)
    elif n in dg_bus_list:
        ax_pf.plot(xp, yp, 'o', color='red', markersize=5, markeredgecolor='black', zorder=2)
        ax_pf.text(xp, yp + 0.18, str(n), color='red', fontsize=5, ha='center', va='bottom', fontweight='bold', zorder=5)
    else:
        ax_pf.plot(xp, yp, 'o', color='black', markersize=4, markeredgecolor='black', zorder=2)
        ax_pf.text(xp, yp + 0.18, str(n), color='black', fontsize=5, ha='center', va='bottom', fontweight='bold', zorder=5)

# 컬러맵 범위 설정 (붉은색 계열로만)
pf_values = [abs(line_pf[idx]) for idx in active_line_map if idx in line_pf]
if pf_values:
    pf_absmax = max(pf_values)
else:
    pf_absmax = 1.0
cmap_pf = mpl.colormaps['Reds']  # 붉은색 계열 컬러맵
norm_pf = mpl.colors.Normalize(vmin=0, vmax=pf_absmax)

# 전력 흐름 화살표 그리기 (활성화된 선로만)
for line_idx, (u, v) in active_line_map.items():
    if line_idx not in line_pf:
        continue
    pf = line_pf[line_idx]
    if u not in global_pos or v not in global_pos:
        continue
    x1, y1 = global_pos[u]
    x2, y2 = global_pos[v]
    # 방향: 양수면 u→v, 음수면 v→u
    if pf >= 0:
        start, end = (x1, y1), (x2, y2)
    else:
        start, end = (x2, y2), (x1, y1)
    # 붉은색 컬러맵에서 절댓값으로 색상 추출
    color = cmap_pf(norm_pf(abs(pf)))
    ax_pf.annotate(
        '', xy=end, xytext=start,
        arrowprops=dict(arrowstyle='->', color=color, lw=2, shrinkA=8, shrinkB=8),
        zorder=3
    )
    mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
    ax_pf.text(mx, my-0.3, f"{abs(pf):.2f}", color='black', fontsize=8, ha='center', va='center', zorder=4)  # 양수로 표시


sm_pf = mpl.cm.ScalarMappable(cmap=cmap_pf, norm=norm_pf)
sm_pf.set_array([])
cbar_pf = plt.colorbar(sm_pf, ax=ax_pf, orientation='vertical', fraction=0.03, pad=0.03)
cbar_pf.set_label('Active Power Flow [MW]')

ax_pf.set_aspect('equal', adjustable='datalim')
ax_pf.axis('off')

plt.savefig(OUT_PNG_Q_PF, bbox_inches='tight', dpi=300)
print(f"Saved: {OUT_PNG_Q_PF}")


# ------------------- P 그래프 -------------------
P_Load_info_t = pd.read_excel(VAR_XLSX, sheet_name='PDem')
P_Load_info_t = P_Load_info_t[['Index: Buses', 'Index: Times', 'Value']]

Bus_numbers = P_Load_info_t['Index: Buses'].unique()
Hours = P_Load_info_t['Index: Times'].unique()

plt.figure(figsize=(14, 8))
for bus in Bus_numbers:
    p_profile = [P_Load_info_t[(P_Load_info_t['Index: Buses'] == bus) & (P_Load_info_t['Index: Times'] == hour)]['Value'].values[0] for hour in Hours]
    plt.plot(Hours, p_profile, label=f'Bus {bus}')
plt.xlabel('Hour')
plt.ylabel('P (MW)')
plt.title('Active Power (P) Profile for Each Bus (1~33) Over 24 Hours')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(Hours)
plt.legend(loc='upper right', ncol=2, fontsize='small')
plt.tight_layout()
p_fig_path = set_png_output("Load_P")
plt.savefig(p_fig_path)
plt.close()
print(f"Saved: {p_fig_path}")

# ------------------- Q 그래프 -------------------
Q_Load_info_t = pd.read_excel(VAR_XLSX, sheet_name='QDem')
Q_Load_info_t = Q_Load_info_t[['Index: Buses', 'Index: Times', 'Value']]

plt.figure(figsize=(14, 8))
for bus in Bus_numbers:
    q_profile = [Q_Load_info_t[(Q_Load_info_t['Index: Buses'] == bus) & (Q_Load_info_t['Index: Times'] == hour)]['Value'].values[0] for hour in Hours]
    plt.plot(Hours, q_profile, label=f'Bus {bus}')
plt.xlabel('Hour')
plt.ylabel('Q (MVAr)')
plt.title('Reactive Power (Q) Profile for Each Bus (1~33) Over 24 Hours')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(Hours)
plt.legend(loc='upper right', ncol=2, fontsize='small')
plt.tight_layout()
q_fig_path = set_png_output("Load_Q")
plt.savefig(q_fig_path)
plt.close()
print(f"Saved: {q_fig_path}")

# ------------------- Gen 그래프 -------------------
Gen_info_t = pd.read_excel(VAR_XLSX, sheet_name='PGen')
Gen_info_t = Gen_info_t[['Index: Buses', 'Index: Times', 'Value']]

plt.figure(figsize=(14, 8))
for bus in Bus_numbers:
    # 각 시간별로 해당 bus의 모든 generator의 합을 구함
    gen_profile = [
        Gen_info_t[(Gen_info_t['Index: Buses'] == bus) & (Gen_info_t['Index: Times'] == hour)]['Value'].sum()
        for hour in Hours
    ]
    plt.plot(Hours, gen_profile, label=f'Bus {bus}')
plt.xlabel('Hour')
plt.ylabel('P (MW)')
plt.title('Gen (P) Profile for Each Bus (1~33) Over 24 Hours')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(Hours)
plt.legend(loc='upper right', ncol=2, fontsize='small')
plt.tight_layout()
gen_fig_path = set_png_output("Gen_P")
plt.savefig(gen_fig_path)
plt.close()
print(f"Saved: {gen_fig_path}")

# ------------------- Gen 그래프 (누적 영역형 그래프) -------------------
# 각 bus별로 시간 순서대로 값을 정렬하여 2D 배열 생성 (bus x hour)
gen_matrix = np.zeros((len(Bus_numbers), len(Hours)))
for i, bus in enumerate(Bus_numbers):
    for j, hour in enumerate(Hours):
        gen_matrix[i, j] = Gen_info_t[(Gen_info_t['Index: Buses'] == bus) & (Gen_info_t['Index: Times'] == hour)]['Value'].sum()

plt.figure(figsize=(14, 8))
plt.stackplot(Hours, gen_matrix, labels=[f'Bus {bus}' for bus in Bus_numbers])
plt.xlabel('Hour')
plt.ylabel('P (MW)')
plt.title('Gen (P) Profile for Each Bus (1~33) Over 24 Hours (Stacked Area)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(Hours)
plt.legend(loc='upper left', ncol=2, fontsize='small')
plt.tight_layout()
gen_fig_path = set_png_output("Gen_P_stacked_area")
plt.savefig(gen_fig_path)
plt.close()
print(f"Saved: {gen_fig_path}")


# ------------------- V_mag(Voltage Magnitude) 시각화 -------------------

V_mag_info_t = pd.read_excel(VAR_XLSX, sheet_name='V_mag')
V_mag_info_t = V_mag_info_t[['Index: Buses', 'Index: Times', 'Value']]

bus_list = sorted(V_mag_info_t['Index: Buses'].unique())
time_list = sorted(V_mag_info_t['Index: Times'].unique())
V_mag_matrix = np.zeros((len(bus_list), len(time_list)))
for i, bus in enumerate(bus_list):
    for j, t in enumerate(time_list):
        V_mag_matrix[i, j] = V_mag_info_t[(V_mag_info_t['Index: Buses'] == bus) & (V_mag_info_t['Index: Times'] == t)]['Value'].values[0]

# 히트맵
fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(V_mag_matrix, aspect='auto', cmap='plasma', origin='lower')
plt.colorbar(im, ax=ax, label='Voltage Magnitude (p.u.)')
ax.set_xlabel('Time (h)')
ax.set_ylabel('Bus Number')
ax.set_title('Voltage Magnitude at Each Bus Over 24 Hours (Heatmap)')
ax.set_xticks(np.arange(len(time_list)))
ax.set_xticklabels([str(t+1) for t in range(len(time_list))])
ax.set_yticks(np.arange(len(bus_list)))
ax.set_yticklabels([str(bus) for bus in bus_list])
plt.tight_layout()
v_mag_heatmap_path = set_png_output("V_mag_heatmap")
fig.savefig(v_mag_heatmap_path)
plt.close(fig)
print(f"Saved: {v_mag_heatmap_path}")

# 꺾은선 그래프
fig, ax = plt.subplots(figsize=(14, 5))
# color_map = plt.get_cmap('tab20', len(bus_list))
# for i, bus in enumerate(bus_list):
#     color = color_map(i)
#     ax.plot(range(1, len(time_list)+1), V_mag_matrix[i, :], label=f'Bus {bus}', color=color)
# ax.set_xlabel('Time (h)')
# ax.set_ylabel('Voltage Magnitude (p.u.)')
# ax.set_title('Voltage Magnitude at Each Bus (Line Plot)')
# ax.set_xticks(np.arange(1, len(time_list)+1))
# ax.grid(True, linestyle='--', alpha=0.5)
# ax.legend(loc='best', ncol=2, fontsize='small')
# plt.tight_layout()
# v_mag_line_path = os.path.join(FIG_DIR, "V_mag_line.png")
# fig.savefig(v_mag_line_path)
# plt.close(fig)
# print(f"Saved: {v_mag_line_path}")

color_map = plt.get_cmap('plasma', len(time_list))
for j, t in enumerate(time_list):
    color = color_map(j)
    ax.plot(bus_list, V_mag_matrix[:, j], label=f'{t}h', color=color)
ax.set_xlabel('Bus Number')
ax.set_ylabel('Voltage Magnitude (p.u.)')
ax.set_title('Voltage Magnitude at Each Bus (Line Plot, Each Line = Time)')
ax.set_xticks(bus_list)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='best', ncol=2, fontsize='small')
plt.tight_layout()
v_mag_line_path = set_png_output("V_mag_line")
fig.savefig(v_mag_line_path)
plt.close(fig)
print(f"Saved: {v_mag_line_path}")

# ------------------- V_ang(Voltage Angle) 시각화 -------------------
V_ang_info_t = pd.read_excel(VAR_XLSX, sheet_name='V_ang')
V_ang_info_t = V_ang_info_t[['Index: Buses', 'Index: Times', 'Value']]

bus_list_ang = sorted(V_ang_info_t['Index: Buses'].unique())
time_list_ang = sorted(V_ang_info_t['Index: Times'].unique())
V_ang_matrix = np.zeros((len(bus_list_ang), len(time_list_ang)))
for i, bus in enumerate(bus_list_ang):
    for j, t in enumerate(time_list_ang):
        V_ang_matrix[i, j] = V_ang_info_t[(V_ang_info_t['Index: Buses'] == bus) & (V_ang_info_t['Index: Times'] == t)]['Value'].values[0] * 180 / np.pi

# 히트맵
fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(V_ang_matrix, aspect='auto', cmap='plasma', origin='lower')
plt.colorbar(im, ax=ax, label='Voltage Angle (deg)')
ax.set_xlabel('Time (h)')
ax.set_ylabel('Bus Number')
ax.set_title('Voltage Angle at Each Bus Over 24 Hours (Heatmap)')
ax.set_xticks(np.arange(len(time_list_ang)))
ax.set_xticklabels([str(t+1) for t in range(len(time_list_ang))])
ax.set_yticks(np.arange(len(bus_list_ang)))
ax.set_yticklabels([str(bus) for bus in bus_list_ang])
plt.tight_layout()
v_ang_heatmap_path = set_png_output("V_ang_heatmap")
fig.savefig(v_ang_heatmap_path)
plt.close(fig)
print(f"Saved: {v_ang_heatmap_path}")

# 꺾은선 그래프
# fig, ax = plt.subplots(figsize=(14, 5))
# color_map_ang = plt.get_cmap('tab20', len(bus_list_ang))
# for i, bus in enumerate(bus_list_ang):
#     color = color_map_ang(i)
#     ax.plot(range(1, len(time_list_ang)+1), V_ang_matrix[i, :], label=f'Bus {bus}', color=color)
# ax.set_xlabel('Time (h)')
# ax.set_ylabel('Voltage Angle (deg)')
# ax.set_title('Voltage Angle at Each Bus (Line Plot)')
# ax.set_xticks(np.arange(1, len(time_list_ang)+1))
# ax.grid(True, linestyle='--', alpha=0.5)
# ax.legend(loc='best', ncol=2, fontsize='small')
# plt.tight_layout()
# v_ang_line_path = os.path.join(FIG_DIR, "V_ang_line.png")
# fig.savefig(v_ang_line_path)
# plt.close(fig)
# print(f"Saved: {v_ang_line_path}")

fig, ax = plt.subplots(figsize=(14, 5))
color_map_time = plt.get_cmap('plasma', len(time_list_ang))
for j, t in enumerate(time_list_ang):
    color = color_map_time(j)
    ax.plot(bus_list_ang, V_ang_matrix[:, j], label=f'Time {t+1}', color=color)
ax.set_xlabel('Bus Number')
ax.set_ylabel('Voltage Angle (deg)')
ax.set_title('Voltage Angle at Each Bus (Each Line = Time)')
ax.set_xticks(bus_list_ang)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='best', ncol=2, fontsize='small')
plt.tight_layout()
v_ang_line_path = set_png_output("V_ang_line")
fig.savefig(v_ang_line_path)
plt.close(fig)
print(f"Saved: {v_ang_line_path}")

# ------------------- P_line_flow_sending 시각화 -------------------
P_line_flow_sending = pd.read_excel(VAR_XLSX, sheet_name='P_line_flow_sending')

line_list = sorted(P_line_flow_sending['Index: Lines'].unique())
time_list = sorted(P_line_flow_sending['Index: Times'].unique())
P_line_flow_matrix = np.zeros((len(line_list), len(time_list)))
for i, line in enumerate(line_list):
    for j, t in enumerate(time_list):
        value = P_line_flow_sending[
            (P_line_flow_sending['Index: Lines'] == line) & 
            (P_line_flow_sending['Index: Times'] == t)
        ]['Value']
        if not value.empty:
            P_line_flow_matrix[i, j] = value.values[0]
        else:
            P_line_flow_matrix[i, j] = np.nan  # 또는 0 등 기본값

# 히트맵
fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(P_line_flow_matrix, aspect='auto', cmap='plasma', origin='lower')
plt.colorbar(im, ax=ax, label='P_line_flow_sending (p.u.)')
ax.set_xlabel('Time (h)')
ax.set_ylabel('Line Number')
ax.set_title('Active Power Flow (Sending End) for Each Line Over 24 Hours (Heatmap)')
ax.set_xticks(np.arange(len(time_list)))
ax.set_xticklabels([str(t+1) for t in range(len(time_list))])
ax.set_yticks(np.arange(len(line_list)))
ax.set_yticklabels([str(line) for line in line_list])
plt.tight_layout()
p_line_heatmap_path = set_png_output("P_line_flow_sending_heatmap")
fig.savefig(p_line_heatmap_path)
plt.close(fig)
print(f"Saved: {p_line_heatmap_path}")

# 꺾은선 그래프
# fig, ax = plt.subplots(figsize=(14, 6))
# color_map_line = plt.get_cmap('tab20', len(line_list))
# for i, line in enumerate(line_list):
#     color = color_map_line(i)
#     ax.plot(range(1, len(time_list)+1), P_line_flow_matrix[i, :], label=f'Line {line}', color=color)
# ax.set_xlabel('Time (h)')
# ax.set_ylabel('P_line_flow_sending (p.u.)')
# ax.set_title('Active Power Flow (Sending End) for Each Line (Line Plot)')
# ax.set_xticks(np.arange(1, len(time_list)+1))
# ax.grid(True, linestyle='--', alpha=0.5)
# ax.legend(loc='best', ncol=2, fontsize='small')
# plt.tight_layout()
# p_line_line_path = os.path.join(FIG_DIR, "P_line_flow_sending_line.png")
# fig.savefig(p_line_line_path)
# plt.close(fig)
# print(f"Saved: {p_line_line_path}")

fig, ax = plt.subplots(figsize=(14, 6))
color_map_time = plt.get_cmap('plasma', len(time_list))
for j, t in enumerate(time_list):
    color = color_map_time(j)
    ax.plot(line_list, P_line_flow_matrix[:, j], label=f'{t}h', color=color)
ax.set_xlabel('Line Number')
ax.set_ylabel('P_line_flow_sending (p.u.)')
ax.set_title('Active Power Flow (Sending End) for Each Line (Each Line = Time)')
ax.set_xticks(line_list)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='best', ncol=2, fontsize='small')
plt.tight_layout()
p_line_line_path = set_png_output("P_line_flow_sending_line")
fig.savefig(p_line_line_path)
plt.close(fig)
print(f"Saved: {p_line_line_path}")