import os
import pandas as pd
import numpy as np
from collections import defaultdict, deque, Counter
import matplotlib as mpl
import matplotlib.pyplot as plt
from config import switch

## Set directory
save_directory = os.path.dirname(__file__) + "/Pre_cal_data/"       # Set save parameter directory
output_directory = os.path.dirname(__file__) + "/Output_data/"     # Set output directory

if switch == 1:
    simul_case = '33bus_MINLP_Opt_problem_for_min_cost_with_switch_'
elif switch == 0:
    simul_case = '33bus_MINLP_Opt_problem_for_min_cost_without_switch_'

# DG list
dg_list = []

list_dg = pd.read_excel(save_directory + 'DG_Candidates.xlsx', sheet_name='Candidate')
list_dg_df = pd.DataFrame(list_dg, columns=['Index','Bus number', 'Rating[MW]', 'Type', 'Profile'])
print(list_dg_df)

# Ensure output subdirectories exist
os.makedirs(os.path.join(output_directory, "Variables"), exist_ok=True)
os.makedirs(os.path.join(output_directory, "Dual"), exist_ok=True)

# ====== 추가: Figure 저장 폴더 생성 ======
output_fig_directory = os.path.join(output_directory, "Figures")
os.makedirs(output_fig_directory, exist_ok=True)

VAR_XLSX = os.path.join(output_directory, "Variables", f"{simul_case}Variables.xlsx")
LINE_INFO = os.path.join(save_directory, "Line_info.csv")
SYSTEM_XLSX = os.path.join(save_directory, "System.xlsx")
if switch == 1:
    OUT_PNG_NL = os.path.join(output_fig_directory, "33bus_tree_layout_Netload_with_switch.png")
elif switch == 0:
    OUT_PNG_NL = os.path.join(output_fig_directory, "33bus_tree_layout_Netload_without_switch.png")

if switch == 1:
    OUT_PNG_V = os.path.join(output_fig_directory, "33bus_tree_layout_Vmag_with_switch.png")
elif switch == 0:
    OUT_PNG_V = os.path.join(output_fig_directory, "33bus_tree_layout_Vmag_without_switch.png")

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
print(result_df)

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

# ===================== 전압 플롯 =====================
fig_v, ax_v = plt.subplots(figsize=(12, 8))

# ---- 전압 데이터 불러오기 및 색상 매핑 ----
V_MAG_XLSX = os.path.join(output_directory, "Variables", f"{simul_case}Variables.xlsx")
V_MAG_SHEET = "V_mag"
TIME_INDEX = 15  # 원하는 시간 인덱스

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
cmap_v = mpl.cm.get_cmap("plasma")
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
    ax_v.plot(xp, yp, 'o', color=color, markersize=10, markeredgecolor='black')
    ax_v.text(xp, yp - 0.2, str(n), ha='left', va='top', fontweight='bold', fontsize=8)
    if n in leaf_nodes or n in max_v_nodes or n in min_v_nodes:
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
nlmin, nlmax = min(nl_values), max(nl_values)
cmap_nl = plt.get_cmap('plasma')
norm_nl = mpl.colors.Normalize(vmin=nlmin, vmax=nlmax)

for u in all_children:
    for v in all_children[u]:
        x1, y1 = global_pos[u]; x2, y2 = global_pos[v]
        ax_nl.plot([x1, x2], [y1, y2], color="black", linewidth=1)

max_nl_nodes = [n for n, v in bus_nl.items() if np.isclose(v, nlmax, atol=1e-8)]
min_nl_nodes = [n for n, v in bus_nl.items() if np.isclose(v, nlmin, atol=1e-8)]
dg_bus_list = list(list_dg_df['Bus number'].astype(int).unique())

for n, (xp, yp) in global_pos.items():
    v = bus_nl.get(n, nlmax)
    color = cmap_nl(norm_nl(v))
    if n in dg_bus_list:
        ax_nl.plot(xp, yp, 'o', color=color, markersize=12, markeredgecolor='blue', zorder=3)
    else:
        ax_nl.plot(xp, yp, 'o', color=color, markersize=10, markeredgecolor='black', zorder=2)
    ax_nl.text(xp, yp - 0.2, str(n), ha='left', va='top', fontweight='bold', fontsize=8)
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