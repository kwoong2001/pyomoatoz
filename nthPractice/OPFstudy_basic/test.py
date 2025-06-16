import time
from matpower import start_instance

print("MATPOWER 스크립트 테스트 시작")

# 1. Octave 인스턴스 시작 시간 측정
print("Octave 인스턴스를 시작합니다...")
t0 = time.time()
m = start_instance()
t1 = time.time()
print(f"==> 인스턴스 시작 소요 시간: {t1 - t0:.4f} 초")

# 2. loadcase 실행 시간 측정
print("loadcase를 실행합니다...")
t2 = time.time()
mpc = m.loadcase('case33bw')
t3 = time.time()
print(f"==> loadcase 실행 소요 시간: {t3 - t2:.4f} 초")

print("\n테스트 완료.")