"""
Matpower 계통에 자원들 setting
- Line
- Distributed Generators
"""

# Matpower Setting 및 정보 추출
def Set_System_Env(np,pd,save_directory,mpc):
    
    #Slack 버스 찾기
    Slackbus = find_slack(mpc)
    
    #선로의 초기 상태 저장 및 모든 선로 상태 On으로 변경
    previous_branch_array = change_line_status(mpc)
    
    # Distributed generators 추가
    add_distributed_gen(np,pd,save_directory,mpc)
    
    return Slackbus, previous_branch_array

## Find Slackbus
def find_slack(mpc):    
    Slackbus = 0

    for bus_info in mpc['bus']:
        if bus_info[1] == 3: 
            Slackbus = int(bus_info[0])
            
    print(f"{len(mpc['bus'])}-buses case, Slack bus: [{Slackbus}]")
    
    return Slackbus

## Change disconnected line to connected line
def change_line_status(mpc):
    ## Change disconnected line to connected line
    previous_branch_array = mpc['branch'].copy()# Save disconnected lines data
    branch_idx = 1

    for branch in mpc['branch']:
        if branch[-3] == 0:
            branch[-3] = 1
            print(f"{branch_idx}-th line(from bus:{branch[0]}, to bus:{branch[1]}) disconnected --> connected")
        branch_idx +=1
    
    return previous_branch_array

def add_distributed_gen(np,pd,save_directory,mpc):
    
    ## Add unit gen - Matpower Generator 형식에 맞는 기본 발전기 데이터 불러오기
    Unit_DG_excel_file = save_directory + 'Basic_DG_Data.xlsx'

    #'DG_Data' 시트에서 발전기 데이터 읽기 (발전기 정보)
    df_gen = pd.read_excel(Unit_DG_excel_file, sheet_name='DG_Data')

    #'DG_Cost_Data' 시트에서 비용 데이터 읽기
    df_cost = pd.read_excel(Unit_DG_excel_file, sheet_name='DG_Cost_Data')

    #DataFrame -> numpy array 변환
    new_gen = df_gen.to_numpy().astype(float)
    new_cost = df_cost.to_numpy().astype(float)

    ## Add DG candidates - DG candidate 정보를 불러온 후 Matpower 형식에 맞게 변환
    DG_Info_excel_file = save_directory + 'DG_Candidates.xlsx'

    #'Candidate' 시트에서 후보 발전기들 정보 읽기
    df_dg_candidates = pd.read_excel(DG_Info_excel_file, sheet_name='Candidate')

    for dg in df_dg_candidates.index:
        tmp_gen = new_gen.copy()
        tmp_gen_cost = new_cost.copy()
        
        tmp_gen[0][0] = df_dg_candidates.loc[dg,'Bus number']
        tmp_gen[0][3] = df_dg_candidates.loc[dg,'Q_Control_Factor']*df_dg_candidates.loc[dg,'Rating[MW]'] #Qmax
        tmp_gen[0][4] = (-1)*df_dg_candidates.loc[dg,'Q_Control_Factor']*df_dg_candidates.loc[dg,'Rating[MW]'] #Qmin
        tmp_gen[0][8] = df_dg_candidates.loc[dg,'Rating[MW]']
        
        # 기존 mpc['gen'], mpc['gencost']가 numpy array임을 가정하고,
        # 데이터 추가 (행 방향으로, axis=0)
        mpc['gen'] = np.vstack([mpc['gen'], tmp_gen])
        mpc['gencost'] = np.vstack([mpc['gencost'], tmp_gen_cost])