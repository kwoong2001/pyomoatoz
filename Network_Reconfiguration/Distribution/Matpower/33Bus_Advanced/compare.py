import os
import pandas as pd
import numpy as np
from collections import defaultdict, deque, Counter
import matplotlib as mpl
import matplotlib.pyplot as plt


## Set directory
save_directory = os.path.join(os.path.dirname(__file__), "Pre_cal_data")
output_directory = os.path.join(os.path.dirname(__file__), "Output_data")

compare_directory = os.path.join(output_directory, "Compare")
os.makedirs(compare_directory, exist_ok=True)

switch = 1
pv_penetration = 0.6
Time = 15

def get_simul_case_names(switch, pv_penetration):
    if switch == 1:
        simul_case_n = '33bus_MINLP_Opt_problem_for_min_cost_with_switch_none_dgs_'
        simul_case_m = f'33bus_MINLP_Opt_problem_for_min_cost_with_switch_mid_dgs_{pv_penetration}_pv_penetration_'
        simul_case_e = f'33bus_MINLP_Opt_problem_for_min_cost_with_switch_end_dgs_{pv_penetration}_pv_penetration_'
    else:
        simul_case_n = '33bus_MINLP_Opt_problem_for_min_cost_without_switch_none_dgs_'
        simul_case_m = f'33bus_MINLP_Opt_problem_for_min_cost_without_switch_mid_dgs_{pv_penetration}_pv_penetration_'
        simul_case_e = f'33bus_MINLP_Opt_problem_for_min_cost_without_switch_end_dgs_{pv_penetration}_pv_penetration_'
    return simul_case_n, simul_case_m, simul_case_e

simul_case_n, simul_case_m, simul_case_e = get_simul_case_names(switch, pv_penetration)

VAR_XLSX_n = os.path.join(output_directory, "Variables", f"{simul_case_n}Variables.xlsx")
VAR_XLSX_m = os.path.join(output_directory, "Variables", f"{simul_case_m}Variables.xlsx")
VAR_XLSX_e = os.path.join(output_directory, "Variables", f"{simul_case_e}Variables.xlsx")

Leaf_csv_n = os.path.join(output_directory, "Figures", simul_case_n + "leaf_nodes.csv")
Leaf_csv_m = os.path.join(output_directory, "Figures", simul_case_m + "leaf_nodes.csv")
Leaf_csv_e = os.path.join(output_directory, "Figures", simul_case_e + "leaf_nodes.csv")

leaf_nodes_n = pd.read_csv(Leaf_csv_n)['leaf_node'].tolist()
leaf_nodes_m = pd.read_csv(Leaf_csv_m)['leaf_node'].tolist()
leaf_nodes_e = pd.read_csv(Leaf_csv_e)['leaf_node'].tolist()

list_dg_xlsx_n = pd.read_excel(os.path.join(save_directory, 'DG_Candidates_none.xlsx'), sheet_name='Candidate')
list_dg_xlsx_m = pd.read_excel(os.path.join(save_directory, 'DG_Candidates_mid.xlsx'), sheet_name='Candidate')
list_dg_xlsx_e = pd.read_excel(os.path.join(save_directory, 'DG_Candidates_end.xlsx'), sheet_name='Candidate')

list_dg_n = pd.DataFrame(list_dg_xlsx_n, columns=['Index','Bus number', 'Rating[MW]', 'Type', 'Profile'])
dg_bus_list_n = list(list_dg_n['Bus number'].astype(int).unique())

list_dg_m = pd.DataFrame(list_dg_xlsx_m, columns=['Index','Bus number', 'Rating[MW]', 'Type', 'Profile'])
dg_bus_list_m = list(list_dg_m['Bus number'].astype(int).unique())

list_dg_e = pd.DataFrame(list_dg_xlsx_e, columns=['Index','Bus number', 'Rating[MW]', 'Type', 'Profile'])
dg_bus_list_e = list(list_dg_e['Bus number'].astype(int).unique())


def load_variable_sheets(file_path):

    sheets = {}
    try:
        xls = pd.ExcelFile(file_path)
        for sheet in ['V_mag', 'PGen', 'QGen','result']:
            if sheet in xls.sheet_names:
                sheets[sheet] = pd.read_excel(xls, sheet_name=sheet)
            else:
                sheets[sheet] = None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        sheets = {'V_mag': None, 'PGen': None, 'QGen': None, 'result': None}
    return sheets

sheets_n = load_variable_sheets(VAR_XLSX_n)
sheets_m = load_variable_sheets(VAR_XLSX_m)
sheets_e = load_variable_sheets(VAR_XLSX_e)

# Combine Variable_name, Index: Buses, Index: Times, and Value columns from each sheet
def combine_variable_sheets(sheets_n, sheets_m, sheets_e, Var):
    # Check if 'Index: Gens' exists in the sheets
    has_gens = 'Index: Gens' in sheets_n[Var].columns

    # Select columns accordingly
    base_cols = ['Variable_name']
    if has_gens:
        base_cols.append('Index: Gens')
    base_cols += ['Index: Buses', 'Index: Times', 'Value']

    df_n = sheets_n[Var][base_cols].copy()
    df_m = sheets_m[Var][base_cols].copy()
    df_e = sheets_e[Var][base_cols].copy()

    df_n = df_n.rename(columns={'Value': 'none'})
    df_m = df_m.rename(columns={'Value': 'mid'})
    df_e = df_e.rename(columns={'Value': 'end'})

    if has_gens:
        # Group by Variable_name, Index: Buses, Index: Times and sum over Gens
        group_cols = ['Variable_name', 'Index: Buses', 'Index: Times']
        df_n = df_n.groupby(group_cols, as_index=False).agg({'none': 'sum'})
        df_m = df_m.groupby(group_cols, as_index=False).agg({'mid': 'sum'})
        df_e = df_e.groupby(group_cols, as_index=False).agg({'end': 'sum'})
    else:
        group_cols = ['Variable_name', 'Index: Buses', 'Index: Times']

    # Merge on the appropriate columns
    merged = df_n.merge(df_m[group_cols + (['mid'] if 'mid' in df_m else [])],
                        on=group_cols, how='outer')
    merged = merged.merge(df_e[group_cols + (['end'] if 'end' in df_e else [])],
                          on=group_cols, how='outer')
    return merged

combined_vmag = combine_variable_sheets(sheets_n, sheets_m, sheets_e, 'V_mag')
print(combined_vmag.head())
combined_vmag.to_excel(os.path.join(compare_directory, f"Combined_Vmag_with_switch_{pv_penetration}_pv_penetration_{Time}_h.xlsx"), index=False)

# combined_pgen = combine_variable_sheets(sheets_n, sheets_m, sheets_e, 'PGen')
# print(combined_pgen.head())
# combined_pgen.to_excel(os.path.join(compare_directory, f"Combined_Pgen_with_switch_{pv_penetration}_pv_penetration_{Time}_h.xlsx"), index=False)

# combined_qgen = combine_variable_sheets(sheets_n, sheets_m, sheets_e, 'QGen')
# print(combined_qgen.head())
# combined_qgen.to_excel(os.path.join(compare_directory, f"Combined_Qgen_with_switch_{pv_penetration}_pv_penetration_{Time}_h.xlsx"), index=False)


# Vmag
df_time = combined_vmag[combined_vmag['Index: Times'] == Time]
plt.figure(figsize=(10, 6))
plt.plot(df_time['Index: Buses'], df_time['none'], marker='o', label='none')
plt.plot(df_time['Index: Buses'], df_time['mid'], marker='o', label='mid')
plt.plot(df_time['Index: Buses'], df_time['end'], marker='o', label='end')

# leaf_nodes 및 DG bus 각각 표시
# none
for i, leaf in enumerate(leaf_nodes_n):
    y = df_time[df_time['Index: Buses'] == leaf]['none']
    if not y.empty:
        plt.scatter(leaf, y, color='g', marker='s', s=120, label='Leaf Node (none)' if i == 0 else "")
for i, dg in enumerate(dg_bus_list_n):
    y = df_time[df_time['Index: Buses'] == dg]['none']
    if not y.empty:
        plt.scatter(dg, y, color='g', marker='o', s=120, label='DG Bus (none)' if i == 0 else "")

# mid
for i, leaf in enumerate(leaf_nodes_m):
    y = df_time[df_time['Index: Buses'] == leaf]['mid']
    if not y.empty:
        plt.scatter(leaf, y, color='b', marker='s', s=120, label='Leaf Node (mid)' if i == 0 else "")
for i, dg in enumerate(dg_bus_list_m):
    y = df_time[df_time['Index: Buses'] == dg]['mid']
    if not y.empty:
        plt.scatter(dg, y, color='b', marker='o', s=120, label='DG Bus (mid)' if i == 0 else "")

# end
for i, leaf in enumerate(leaf_nodes_e):
    y = df_time[df_time['Index: Buses'] == leaf]['end']
    if not y.empty:
        plt.scatter(leaf, y, color='r', marker='s', s=120, label='Leaf Node (end)' if i == 0 else "")
for i, dg in enumerate(dg_bus_list_e):
    y = df_time[df_time['Index: Buses'] == dg]['end']
    if not y.empty:
        plt.scatter(dg, y, color='r', marker='o', s=120, label='DG Bus (end)' if i == 0 else "")

plt.xlabel('Bus')
plt.ylabel('Vmag')
plt.title(f'Vmag_with_switch_{pv_penetration}_pv_penetration_{Time}_h')
plt.legend()
plt.grid(True)
plt.tight_layout()
if switch == 1:
    plt.savefig(os.path.join(compare_directory, f'Vmag_with_switch_{pv_penetration}_pv_penetration_{Time}_h.png'))
plt.close()

# result 시트에서 값 추출
def extract_result_values(sheets, label):
    df = sheets['result']
    if df is None:
        return None
    # Name 컬럼이 있는 경우
    if 'Name' in df.columns and 'Value' in df.columns:
        return df.set_index('Name')['Value'].rename(label)
    # 첫 번째 컬럼이 Name, 두 번째가 Value인 경우
    return df.set_index(df.columns[0])[df.columns[1]].rename(label)

result_n = extract_result_values(sheets_n, 'none')
result_m = extract_result_values(sheets_m, 'mid')
result_e = extract_result_values(sheets_e, 'end')

# 하나의 DataFrame으로 합치기
result_df = pd.concat([result_n, result_m, result_e], axis=1)
result_df = result_df.loc[['Objective_value', 'P_total', 'D_total', 'P_loss_total']]  # 원하는 순서로 정렬



# Objective_value
result_df.loc[['Objective_value']].plot(kind='bar', figsize=(6,4))
plt.ylabel('Objective_value $')
plt.title('Objective_value Comparison (none, mid, end)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(compare_directory, f'Objective_value_Comparison_with_switch_{pv_penetration}_pv_penetration_{Time}_h.png'))
plt.close()

# P_total
result_df.loc[['P_total']].plot(kind='bar', figsize=(8,5))
plt.ylabel('Value MW')
plt.title('Gen_P_total Comparison (none, mid, end)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(compare_directory, f'Gen_P_Comparison_with_switch_{pv_penetration}_pv_penetration_{Time}_h.png'))
plt.close()

# P_loss_total
result_df.loc[['P_loss_total']].plot(kind='bar', figsize=(8,5))
plt.ylabel('Value MW')
plt.title('P_loss_total Comparison (none, mid, end)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(compare_directory, f'Ploss_Comparison_with_switch_{pv_penetration}_pv_penetration_{Time}_h.png'))
plt.close()

