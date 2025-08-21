import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----- 경로 및 디렉토리 관리 -----
## Set directory
save_directory = os.path.dirname(__file__) + "/Pre_cal_data/"       # Set save parameter directory
output_directory = os.path.dirname(__file__) + "/Output_data/"     # Set output directory
# simul_case = '33bus_MINLP_Opt_problem_for_min_cost_'
simul_case = '33bus_MINLP_Opt_problem_for_min_cost_without_switch_'

# Ensure output subdirectories exist
os.makedirs(os.path.join(output_directory, "Variables"), exist_ok=True)
os.makedirs(os.path.join(output_directory, "Dual"), exist_ok=True)

VAR_XLSX = os.path.join(output_directory, "Variables", f"{simul_case}Variables.xlsx")
LINE_INFO = os.path.join(save_directory, "Line_info.csv")
LOAD_INFO = os.path.join(save_directory, "Load_info.csv")
SYSTEM_XLSX = os.path.join(save_directory, "System.xlsx")
# OUT_PNG = os.path.join(output_directory, "33bus_tree_layout.png")
OUT_PNG = os.path.join(output_directory, "33bus_tree_layout_without_switch.png")

# ----- 기존 변수 및 데이터 준비 -----
# bus_numbers, hours, Load_info_t, instance 등은 기존 코드와 동일하게 준비되어 있어야 함

# ------------------- P 그래프 -------------------
P_Load_info_t = pd.read_excel(VAR_XLSX, sheet_name='PDem')
P_Load_info_t = P_Load_info_t[['Index: Buses', 'Index: Times', 'Value']]
print(P_Load_info_t.head())

# Define directory for saving figures
FIG_DIR = os.path.join(output_directory, "Figures")
os.makedirs(FIG_DIR, exist_ok=True)

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
p_fig_path = os.path.join(FIG_DIR, "Load_P.png")
plt.savefig(p_fig_path)
plt.close()
print(f"Saved: {p_fig_path}")

# ------------------- Q 그래프 -------------------
Q_Load_info_t = pd.read_excel(VAR_XLSX, sheet_name='QDem')
Q_Load_info_t = Q_Load_info_t[['Index: Buses', 'Index: Times', 'Value']]
print(Q_Load_info_t.head())

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
q_fig_path = os.path.join(FIG_DIR, "Load_Q.png")
plt.savefig(q_fig_path)
plt.close()
print(f"Saved: {q_fig_path}")

# ------------------- V_mag(Voltage Magnitude) 시각화 -------------------

V_mag_info_t = pd.read_excel(VAR_XLSX, sheet_name='V_mag')
V_mag_info_t = V_mag_info_t[['Index: Buses', 'Index: Times', 'Value']]
print(V_mag_info_t.head())

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
v_mag_heatmap_path = os.path.join(FIG_DIR, "V_mag_heatmap.png")
fig.savefig(v_mag_heatmap_path)
plt.close(fig)
print(f"Saved: {v_mag_heatmap_path}")

# 꺾은선 그래프
fig, ax = plt.subplots(figsize=(14, 5))
color_map = plt.get_cmap('tab20', len(bus_list))
for i, bus in enumerate(bus_list):
    color = color_map(i)
    ax.plot(range(1, len(time_list)+1), V_mag_matrix[i, :], label=f'Bus {bus}', color=color)
ax.set_xlabel('Time (h)')
ax.set_ylabel('Voltage Magnitude (p.u.)')
ax.set_title('Voltage Magnitude at Each Bus (Line Plot)')
ax.set_xticks(np.arange(1, len(time_list)+1))
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='best', ncol=2, fontsize='small')
plt.tight_layout()
v_mag_line_path = os.path.join(FIG_DIR, "V_mag_line.png")
fig.savefig(v_mag_line_path)
plt.close(fig)
print(f"Saved: {v_mag_line_path}")

# ------------------- V_ang(Voltage Angle) 시각화 -------------------
V_ang_info_t = pd.read_excel(VAR_XLSX, sheet_name='V_ang')
V_ang_info_t = V_ang_info_t[['Index: Buses', 'Index: Times', 'Value']]
print(V_ang_info_t.head())

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
v_ang_heatmap_path = os.path.join(FIG_DIR, "V_ang_heatmap.png")
fig.savefig(v_ang_heatmap_path)
plt.close(fig)
print(f"Saved: {v_ang_heatmap_path}")

# 꺾은선 그래프
fig, ax = plt.subplots(figsize=(14, 5))
color_map_ang = plt.get_cmap('tab20', len(bus_list_ang))
for i, bus in enumerate(bus_list_ang):
    color = color_map_ang(i)
    ax.plot(range(1, len(time_list_ang)+1), V_ang_matrix[i, :], label=f'Bus {bus}', color=color)
ax.set_xlabel('Time (h)')
ax.set_ylabel('Voltage Angle (deg)')
ax.set_title('Voltage Angle at Each Bus (Line Plot)')
ax.set_xticks(np.arange(1, len(time_list_ang)+1))
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='best', ncol=2, fontsize='small')
plt.tight_layout()
v_ang_line_path = os.path.join(FIG_DIR, "V_ang_line.png")
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
p_line_heatmap_path = os.path.join(FIG_DIR, "P_line_flow_sending_heatmap.png")
fig.savefig(p_line_heatmap_path)
plt.close(fig)
print(f"Saved: {p_line_heatmap_path}")

# 꺾은선 그래프
fig, ax = plt.subplots(figsize=(14, 6))
color_map_line = plt.get_cmap('tab20', len(line_list))
for i, line in enumerate(line_list):
    color = color_map_line(i)
    ax.plot(range(1, len(time_list)+1), P_line_flow_matrix[i, :], label=f'Line {line}', color=color)
ax.set_xlabel('Time (h)')
ax.set_ylabel('P_line_flow_sending (p.u.)')
ax.set_title('Active Power Flow (Sending End) for Each Line (Line Plot)')
ax.set_xticks(np.arange(1, len(time_list)+1))
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='best', ncol=2, fontsize='small')
plt.tight_layout()
p_line_line_path = os.path.join(FIG_DIR, "P_line_flow_sending_line.png")
fig.savefig(p_line_line_path)
plt.close(fig)
print(f"Saved: {p_line_line_path}")