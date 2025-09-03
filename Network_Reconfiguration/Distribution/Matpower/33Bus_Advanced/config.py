import subprocess
import os
import sys

switch = 0 # 1=with switch, 0=without switch
dg_case = 'end' # 'none', 'mid', 'end'
pv_penetration = 1.2 # 전체 부하 대비 태양광 발전 비율

base_dir = os.path.dirname(os.path.abspath(__file__))

def run_script(script):
    try:
        print(f"{script} 실행")
        subprocess.run(
            [sys.executable, os.path.join(base_dir, script)],
            check=True,
            cwd=base_dir
        )
    except subprocess.CalledProcessError as e:
        print(f"{script} 실행 중 오류 발생:", e)
        sys.exit(1)

if __name__ == "__main__":
    run_script('Case33_bus_with_switch.py')
    run_script('result_system_fig.py')
    run_script('result_v_p_fig.py')
    print("모든 실행 완료")