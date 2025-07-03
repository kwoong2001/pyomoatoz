def set_loads(os,pd,Load_info,T,in_dir):
    # # matpower
    # load_factors = {}
    # for hour in range(1, T+1):
    #     if hour == 4:
    #         load_factors[hour] = 0.5
    #     elif hour == 11:
    #         load_factors[hour] = 0.9
    #     elif hour == 12:
    #         load_factors[hour] = 0.85
    #     elif hour == 15:
    #         load_factors[hour] = 1.0
    #     elif 5 <= hour <= 11:
    #         # 04시(0.5)~11시(0.9) 선형 증가
    #         load_factors[hour] = 0.5 + (0.9 - 0.5) * (hour - 4) / (11 - 4)
    #     elif 12 <= hour <= 15:
    #         # 11시(0.9)~12시(0.85) 선형 감소, 12시(0.85)~15시(1.0) 선형 증가
    #         if hour == 12:
    #             load_factors[hour] = 0.85
    #         else:
    #             # 12시(0.85)~15시(1.0) 선형 증가
    #             load_factors[hour] = 0.85 + (1.0 - 0.85) * (hour - 12) / (15 - 12)
    #     else:
    #         # 15시(1.0)~04시(0.5) 선형 감소 (다음날 4시까지)
    #         if hour > 15:
    #             load_factors[hour] = 1.0 - (1.0 - 0.5) * (hour - 15) / ((T - 15) + 4)
    #         else:
    #             # 1~3시
    #             load_factors[hour] = 1.0 - (1.0 - 0.5) * ((hour + (T - 15)) / ((T - 15) + 4))

    # Load_info_t = Load_info.copy()

    # # 시간별 부하 데이터 생성 및 추가
    # for hour in range(1, T+1):
    #     factor = load_factors[hour]
    #     Load_info_t[f'p_mw_{hour}'] = Load_info['p_mw'] * factor
    #     Load_info_t[f'q_mvar_{hour}'] = Load_info['q_mvar'] * factor


    load_i = pd.read_csv(os.path.join(in_dir, 'Load_i.csv'))
    cluster_i = pd.read_csv(os.path.join(in_dir, 'cluster_i.csv'))

    # load_i: bus, cluster
    # cluster_i: cluster, 1, 2, ..., 24
    # load_i 기준으로 cluster_i의 정보를 cluster 열로 붙임
    load_cluster = pd.merge(load_i, cluster_i, on='cluster', how='left')
    load_cluster.index = range(1, len(load_cluster) + 1)  # 인덱스를 1부터 시작하게 설정
    load_cluster.to_csv(os.path.join(in_dir, 'load_cluster.csv'), index=True)
    
    # load_info: bus, p_mw, q_mvar
    # load_cluster: bus, 1, 2, ..., 24 (시간별 계수)
    Load_info_t = Load_info.copy()
    for hour in range(1, T+1):
        factor = load_cluster[str(hour)]
        Load_info_t[f'p_mw_{hour}'] = Load_info_t['p_mw'] * factor
        Load_info_t[f'q_mvar_{hour}'] = Load_info_t['q_mvar'] * factor
    
    Load_info_t.to_csv(os.path.join(in_dir, 'Load_info_t.csv'), index=False)

    return Load_info_t