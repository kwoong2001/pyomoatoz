"""
OPF_Case_33_bus_with_matpower
- Matpower 에 기반한 OPF 구현
- 33, 69 bus 동작 확인
- Switching 고려한 버젼과 고려하지 않은 버젼 모두 구현 완료
"""

import pandas as pd
import numpy as np
import os
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from Packages.Set_values_matpower import *
from Packages.OPF_Creator_matpower import *
from Packages.set_system_env_matpower import *
from Packages.Set_Profiles import *
from pyomo import environ as pym
from matpower import start_instance
from oct2py import octave
from collections import defaultdict, deque, Counter

"""
Set model and parameters with Matpower
"""

## Set directory
save_directory = os.path.dirname(__file__) + "/Pre_cal_data/"       # Set save parameter directory
output_directory = os.path.dirname(__file__) + "/Output_data/"     # Set output directory

# Set time
T = 24

# Set and load Matpower case
m = start_instance()
mpc = m.loadcase('case33bw')

simul_case = '33bus_MINLP_Opt_problem_for_min_cost_'

# Base MVA, Bus, Branch, Generators
base_MVA = mpc['baseMVA']
    
# Find slack bus, add distributed generators, set branch status
[Slackbus, previous_branch_array, pv_curtailment_df] = Set_System_Env(np,pd,save_directory,mpc)

# Set values and parameters (Bus, Line, Gen, Load, Ymatrix, Time)
[Bus_info, Line_info, Gen_info, Load_info, Y_mat_info, Time_info]=Set_All_Values(np,pd,save_directory,m,mpc,previous_branch_array, T)

#임시로 발전기 p_min_mw 설정
# slack bus의 발전기만, 나머지는 기존 값 유지
new_p_min_mw = []
for i in range(len(Gen_info)):
    bus_num = Gen_info['bus'].iloc[i]
    if bus_num == Slackbus:
        new_p_min_mw.append(0)
    else:
        new_p_min_mw.append(Gen_info['min_p_mw'].iloc[i])
Gen_info['min_p_mw'] = new_p_min_mw

print(Gen_info)

# Set profiles of distributed generators and load
[DG_profile_df, Load_profile_df] = Set_Resource_Profiles(np,pd,save_directory,T,Load_info)

"""
Create OPF model and Run Pyomo
"""
# OPF Model Create
model = OPF_model_creator_with_switch(np,pyo,base_MVA,Slackbus,Bus_info,Line_info,Load_info,Gen_info,Time_info,DG_profile_df,Load_profile_df,pv_curtailment_df)


# Create instance for OPF Model
os.chdir(save_directory)
instance = model.create_instance(save_directory + 'Model_data.dat')
os.chdir(os.path.dirname(__file__))

print('Initializing OPF model...')

#IPOPT Solver 이용
# optimizer = pyo.SolverFactory('ipopt')
# optimizer.options['max_iter'] = 30000

#KNITRO Solver 이용
optimizer = pyo.SolverFactory('knitroampl',executable='C:/Program Files/Artelys/Knitro 14.2.0/knitroampl/knitroampl.exe')
optimizer.options['mip_multistart'] = 1
optimizer.options['mip_numthreads'] = 1
instance.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
Problem = optimizer.solve(instance,tee=True,logfile="solver_logging.log")


print('Solving OPF model...')


"""
Result
"""

print('----------------------------------------------------------------')
print(f'Objective value = {instance.obj(): .4f}')
P_total = 0
D_total = 0
for bus in Bus_info.index:
    for time in Time_info['Time']:
        for gen in Gen_info.index:
            if instance.PGen[gen, bus, time].value >= 1e-4:
                pgen = instance.PGen[gen, bus, time].value * base_MVA
            else:
                pgen = 0
            P_total = P_total + pgen

        if instance.PDem[bus, time].expr() >= 1e-4:
            pdem = instance.PDem[bus, time].expr() * base_MVA
        else:
            pdem = 0
        D_total = D_total + pdem
    


print('----------------------------------------------------------------')
print('OPF Model total gen MW:', P_total)
print('OPF Model total load MW:', D_total)



# print('----------------------------------------------------------------')
# print('MatPower validation')


# #Restore line data
# #mpc['branch'] = previous_branch_array

# # Run OPF
# mpopt = m.mpoption('verbose', 2)
# [baseMVA, bus, gen, gencost, branch, f, success, et] = m.runopf(mpc, mpopt, nout='max_nout')

# mat_gen_index = range(1,len(gen)+1)
# mat_gen_info_columns = ['bus','Pg',	'Qg','Qmax','Qmin','Vg','mBase','status','Pmax','Pmin','Pc1','Pc2','Qc1min','Qc1max','Qc2min','Qc2max','ramp_agc','ramp_10','ramp_30','ramp_q',	'apf','unknown1','unknown2','unknown3','unknown4']
# mat_gen_info = pd.DataFrame(gen,index = mat_gen_index, columns = mat_gen_info_columns)

# matpower_gen_mw_total = mat_gen_info['Pg'].sum() 

# print('----------------------------------------------------------------')
# print('Matpower total gen MW:', matpower_gen_mw_total)
# print('----------------------------------------------------------------')
# print('Difference total gen MW:', P_total - (matpower_gen_mw_total))
P_loss_total = 0

for line in Line_info.index:
    
    if instance.P_line_loss[line,time].expr() >= 1e-4:
        ploss = instance.P_line_loss[line,time].expr()
    else:
        ploss = 0
    P_loss_total = P_loss_total + ploss
    #print(f"{bus}-Bus Generation: {pgen}MW")
print(f"Total P loss: {P_loss_total}MW")

"""
Export result file
- Variable
- Dual Variable (경우에 따라 출력되지 않는 경우도 존재함)
"""
## List for storing variable dataframe
var_df_list = []

## Variables
var_idx = 0
for mv in instance.component_objects(ctype=pyo.Var):
    if mv.dim() == 1: # Index dimension == 1
        var_columns = ['Variable_name','Index: '+mv.index_set().name, 'Value']
        max_var_dim = 1
    else: # Index dimension >= 1
        var_columns = ['Variable_name']
        
        subsets_list = list(mv.index_set().domain.subsets())
        for d in subsets_list:
            var_columns.append('Index: '+d.name)
        
        var_columns.append('Value')
        
    var_index = mv.index_set().ordered_data()
    
    if mv.name == 'V_ang':  # Voltage angle
        var_df = pd.DataFrame(index=var_index, columns=var_columns)
        var_deg_df = pd.DataFrame(index=var_index, columns=var_columns)
        for idx in var_index:
            var_df.loc[idx, var_columns[0]] = mv.name
            var_deg_df.loc[idx, var_columns[0]] = 'V_ang(Deg)'
            # Handle multi-index
            if mv.dim() == 1:
                var_df.loc[idx, var_columns[1]] = idx
                var_deg_df.loc[idx, var_columns[1]] = idx
                var_df.loc[idx, var_columns[2]] = mv[idx].value
                var_deg_df.loc[idx, var_columns[2]] = mv[idx].value * 180 / np.pi
            else:
                for d in range(mv.dim()):
                    var_df.loc[idx, var_columns[d+1]] = idx[d]
                    var_deg_df.loc[idx, var_columns[d+1]] = idx[d]
                var_df.loc[idx, var_columns[mv.dim()+1]] = mv[idx].value
                var_deg_df.loc[idx, var_columns[mv.dim()+1]] = mv[idx].value * 180 / np.pi
    else:
        var_df = pd.DataFrame(index=var_index, columns=var_columns)
        for idx in var_index:
            var_df.loc[idx, var_columns[0]] = mv.name
            if mv.dim() == 1:
                var_df.loc[idx, var_columns[1]] = idx
                var_df.loc[idx, var_columns[2]] = mv[idx].value
            else:
                for d in range(mv.dim()):
                    var_df.loc[idx, var_columns[d+1]] = idx[d]
                var_df.loc[idx, var_columns[mv.dim()+1]] = mv[idx].value
    
    if mv.name == 'V_ang': # Voltage angle
        var_df_list.append(var_df)
        var_df_list.append(var_deg_df)
    else:
        var_df_list.append(var_df)
    
    var_idx+=1
    
## Expressions
expr_idx = 0
for me in instance.component_objects(ctype=pyo.Expression):
    if me.dim() == 1: # Index dimension == 1
        var_columns = ['Variable_name','Index: '+me.index_set().name, 'Value']
        max_var_dim = 1
    else: # Index dimension >= 1
        var_columns = ['Variable_name']
        
        subsets_list = list(me.index_set().domain.subsets())
        for d in subsets_list:
            var_columns.append('Index: '+d.name)    
            
        var_columns.append('Value')
        max_var_dim = me.dim()
    
    var_index = me.index_set().ordered_data()
    
    var_df = pd.DataFrame(index = var_index, columns = var_columns)
    for idx in var_index:
        var_df.loc[idx,var_columns[0]] = me.name
        if me.dim() == 1:
            var_df.loc[idx,var_columns[1]] = idx
            var_df.loc[idx,var_columns[2]] = me[idx].expr()
        else:
            for d in range(0,me.dim()):
                var_df.loc[idx,var_columns[d+1]] = idx[d]
                
            var_df.loc[idx,var_columns[me.dim()+1]] = me[idx].expr()

    var_df_list.append(var_df)
    expr_idx +=1

## Variables and Expression name list
var_n_expr_column = ['Name', 'Variable', 'Expression']
var_n_expr_list_df = pd.DataFrame(index = range(0,var_idx+expr_idx+1),columns=var_n_expr_column)
df_idx = 0
for df in var_df_list:
    if df_idx <= var_idx: # Variable list
        var_n_expr_list_df.loc[df_idx,'Name'] = df['Variable_name'].values[0]
        var_n_expr_list_df.loc[df_idx,'Variable'] = 1
        var_n_expr_list_df.loc[df_idx,'Expression'] = 0
    else: # Expression list
        var_n_expr_list_df.loc[df_idx,'Name'] = df['Variable_name'].values[0]
        var_n_expr_list_df.loc[df_idx,'Variable'] = 0
        var_n_expr_list_df.loc[df_idx,'Expression'] = 1
    df_idx += 1

var_df_list.insert(0,var_n_expr_list_df)

## List for storing dual variable dataframe
dual_var_df_list = []

## Dual Variables
try:
    for c in instance.component_objects(pyo.Constraint, active=True):
        
        if c.dim() == 1: # Index dimension == 1
            var_columns = ['Constraint_name','Index: '+c.index_set().name, 'Value']
            max_var_dim = 1
        else: # Index dimension >= 1
            var_columns = ['Constraint_name']
            
            subsets_list = list(c.index_set().domain.subsets())
            for d in subsets_list:
                var_columns.append('Index: '+d.name)
            
            var_columns.append('Value')

        var_index = c.index_set().ordered_data()
        var_df = pd.DataFrame(index = var_index, columns = var_columns)
        for idx in c:
            var_df.loc[idx,var_columns[0]] = c.name
            if c.dim() == 1:
                var_df.loc[idx,var_columns[1]] = idx
                var_df.loc[idx,var_columns[2]] = instance.dual[c[idx]]
            else:
                for d in range(0,c.dim()):
                    var_df.loc[idx,var_columns[d+1]] = idx[d]
                    
                var_df.loc[idx,var_columns[c.dim()+1]] = instance.dual[c[idx]]
        dual_var_df_list.append(var_df)
except:
    print('Check dual')
    
## Write excel
with pd.ExcelWriter(output_directory+'Variables/'+ simul_case +'Variables.xlsx') as writer:  
    for df in var_df_list:
        try:
            df.to_excel(writer, sheet_name=df['Variable_name'].values[0],index=False)
        except:
            df.to_excel(writer, sheet_name='Variable_list',index=False)

try:
    with pd.ExcelWriter(output_directory+'Dual/'+ simul_case +'Dual_Variables.xlsx') as writer:  
        for df in dual_var_df_list:
            try:
                df.to_excel(writer, sheet_name=df['Constraint_name'].values[0],index=False)
            except:
                df.to_excel(writer, sheet_name='Constraint_list',index=False)
except:
    print('Check Dual')

print("solve done!")

# DG list
dg_list = []

list_dg = pd.read_excel(save_directory + 'DG_Candidates.xlsx', sheet_name='Candidate')
list_dg_df = pd.DataFrame(list_dg, columns=['Index','Bus number', 'Rating[MW]', 'Type', 'Profile'])
print(list_dg_df)



# Ensure output subdirectories exist
os.makedirs(os.path.join(output_directory, "Variables"), exist_ok=True)
os.makedirs(os.path.join(output_directory, "Dual"), exist_ok=True)

VAR_XLSX = os.path.join(output_directory, "Variables", f"{simul_case}Variables.xlsx")
LINE_INFO = os.path.join(save_directory, "Line_info.csv")
SYSTEM_XLSX = os.path.join(save_directory, "System.xlsx")
OUT_PNG = os.path.join(output_directory, "33bus_tree_layout.png")

# ===================== 데이터 로드 =====================
# 1) Line_Status: 활성 선로 플래그
line_status_df = pd.read_excel(VAR_XLSX, sheet_name="Line_Status")
if "Index: Lines" not in line_status_df.columns or "Value" not in line_status_df.columns:
    raise ValueError("Line_Status 시트에 'Index: Lines'와 'Value'가 필요합니다.")

# 안전을 위해 타입 정리
line_status_df["Index: Lines"] = line_status_df["Index: Lines"].astype(int)
line_status_df["Value"] = line_status_df["Value"].astype(int)

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
active = result_df[result_df["line_status"].astype(int) == 1].copy()
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

# ===================== 플롯 =====================
fig, ax = plt.subplots(figsize=(12, 8))

# 간선(부모→자식)
for u in all_children:
    for v in all_children[u]:
        x1, y1 = global_pos[u]; x2, y2 = global_pos[v]
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=1)

# 노드
for n, (xp, yp) in global_pos.items():
    ax.plot(xp, yp, 'o', color='black', markersize=5)
    ax.text(xp + 0.15, yp - 0.25, str(n), ha='left', va='top')

ax.set_aspect('equal', adjustable='datalim')
ax.axis('off')
plt.savefig(OUT_PNG, bbox_inches='tight', dpi=300)
print(f"Saved: {OUT_PNG}")

# 리프 정보
deg = Counter()
for u, v in branches:
    deg[u] += 1
    deg[v] += 1
leaf_nodes = sorted([n for n, d in deg.items() if d == 1])
print("Leaf nodes:", leaf_nodes)