"""
자원 Profiles 지정
- DG
- Load
"""

# DG와 Load Profile 생성
def Set_Resource_Profiles(np,pd,save_directory,T,Load_info):
    
    #DG Profile 생성
    DG_profile_df = DG_profiles(np,pd,save_directory,T)
    
    Load_profile_df = Load_Profiles(np,pd,save_directory,T,Load_info)
    
    return DG_profile_df, Load_profile_df

def DG_profiles(np,pd,save_directory,T):
    
    ## Load DG candidates - DG candidate 정보를 불러옴
    DG_Info_excel_file = save_directory + 'DG_Candidates.xlsx'

    #'Candidate' 시트에서 후보 발전기들 정보 읽기
    df_dg_candidates = pd.read_excel(DG_Info_excel_file, sheet_name='Candidate')
    
    DG_profile_df_index = range(1, len(df_dg_candidates)+1)
    DG_profile_df_columns = ['Gens']
    for t in range(1,T+1):
        DG_profile_df_columns.append('p_mw_'+str(t))
        DG_profile_df_columns.append('q_mvar_'+str(t))
    
    DG_profile_df = pd.DataFrame(index = DG_profile_df_index, columns = DG_profile_df_columns)
    
    # 자원별 Profile 읽기
    df_pv_profiles = pd.read_excel(DG_Info_excel_file, sheet_name='PV_Profile')
    df_wind_profiles = pd.read_excel(DG_Info_excel_file, sheet_name='Wind_Profile')

    for dg in df_dg_candidates.index:
        DG_profile_df.loc[dg+1,'Gens'] = dg+2 # 1번 Gen은 Slack bus
        if df_dg_candidates.loc[dg,'Type'] == 'PV':
            for t in range(1,T+1):
                DG_profile_df.loc[dg+1,'p_mw_'+str(t)] = df_dg_candidates.loc[dg,'Rating[MW]'] * df_pv_profiles.loc[t-1,df_dg_candidates.loc[dg,'Profile']]
                DG_profile_df.loc[dg+1,'q_mvar_'+str(t)] = df_dg_candidates.loc[dg,'Rating[MW]'] * df_pv_profiles.loc[t-1,df_dg_candidates.loc[dg,'Profile']] * df_dg_candidates.loc[dg,'Q_Control_Factor']
        elif df_dg_candidates.loc[dg,'Type'] == 'Wind':
            for t in range(1,T+1):
                DG_profile_df.loc[dg+1,'p_mw_'+str(t)] = df_dg_candidates.loc[dg,'Rating[MW]'] * df_wind_profiles.loc[t-1,df_dg_candidates.loc[dg,'Profile']]
                DG_profile_df.loc[dg+1,'q_mvar_'+str(t)] = df_dg_candidates.loc[dg,'Rating[MW]'] * df_wind_profiles.loc[t-1,df_dg_candidates.loc[dg,'Profile']] * df_dg_candidates.loc[dg,'Q_Control_Factor']
    
    DG_profile_df = DG_profile_df.set_index('Gens')
                
    return DG_profile_df

def Load_Profiles(np,pd,save_directory,T,Load_info):
    
    ## Load Load profiles - Load profiles 정보를 불러옴
    Load_profile_excel_file = save_directory + 'Load_profiles.xlsx'

    #'Load_profiles' 시트에서 후보 발전기들 정보 읽기
    df_load_profiles = pd.read_excel(Load_profile_excel_file, sheet_name='Load_profiles')
    
    # Profile index 선택
    profile_idx = 5 # 추후에는 random으로 바뀌거나 Bus에 따라 profile을 선택할 수 있도록 변경
    
    Load_profile_df_index = Load_info.index
    Load_profile_df_columns = []
    for t in range(1,T+1):
        Load_profile_df_columns.append('p_mw_'+str(t))
        Load_profile_df_columns.append('q_mvar_'+str(t))
    
    Load_profile_df = pd.DataFrame(index = Load_profile_df_index , columns = Load_profile_df_columns)
    
    for load in Load_profile_df_index:
        for t in range(1,T+1):
            Load_profile_df.loc[load,'p_mw_'+str(t)] = Load_info.loc[load,'p_mw']*df_load_profiles.loc[t-1,profile_idx]
            Load_profile_df.loc[load,'q_mvar_'+str(t)] = Load_info.loc[load,'q_mvar']*df_load_profiles.loc[t-1,profile_idx]
        
    return Load_profile_df