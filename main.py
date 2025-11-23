"""
F-16 시뮬레이터 - 메인 프로그램
main.py

사용자 입력을 받아 시뮬레이션 모드 선택:
1. 6자유도 비선형 시뮬레이션
2. 트림 계산
"""

import numpy as np
import matplotlib.pyplot as plt
from global_data_F16 import gdata

from simulator import F16Simulator
from plotter import F16Plotter
#from simulator_fast import F16SimulatorFast
from dynamics import F16Dynamics
from linearize import Linearizer

def main():
    """메인 프로그램"""

    # 최근 트림 결과 저장
    last_X_trim = None
    last_u_trim = None
    last_cost = None

    print("\n" + "="*70)
    print("F-16 항공기 6자유도 시뮬레이션 프로그램")
    print("="*70)
    
    # 시뮬레이터 초기화
    sim = F16Simulator()
    #sim = F16SimulatorFast() # 고속 트림버전

    if not sim.initialize('Tables'):
        print("\n초기화 실패! 프로그램을 종료합니다.")
        return
    
    while True:
        print("\n" + "="*70)
        print("메뉴")
        print("="*70)
        print("1. 6자유도 비선형 시뮬레이션")
        print("2. 트림 조건 계산")
        print("3. 선형화 계산 (트림 데이터 필요)")        
        print("0. 종료")
        print("="*70)
        
        try:
            choice = input("\n선택하세요 (0-3): ").strip()
            
            if choice == '0':
                print("\n프로그램을 종료합니다.")
                break
            
            elif choice == '1':
                print("\n" + "="*70)
                print("6자유도 비선형 시뮬레이션")
                print("="*70)
                
                # 사용자 입력
                use_default = input("\n기본조건 사용? (Y/n): ").strip().lower()
                
                if use_default == 'n':
                    # 초기 조건 입력
                    print("\n초기 조건 입력:")
                    vt = float(input("  속도 (ft/s, 기본 500): ") or "500")
                    alt = float(input("  고도 (ft, 기본 10000): ") or "10000")
                    alpha = float(input("  받음각 (deg, 기본 5): ") or "5")
                    
                    X0 = np.zeros(13)
                    X0[0] = vt
                    X0[1] = np.deg2rad(alpha)
                    X0[11] = alt
                    X0[12] = 10.0
                    
                    # print("\n제어 입력:")
                    # throttle = 0.15
                    # elevator = 0.0
                    # aileron = 0.0
                    # rudder = 0.0
            
                    # 제어 입력
                    print("\n제어 입력:")
                    throttle = float(input("  스로틀 (0-1, 기본 0.5): ") or "0.5")
                    elevator = float(input("  엘리베이터 (deg, 기본 0): ") or "0")
                    
                    u = np.array([throttle, elevator, 0.0, 0.0])


                    # u = np.zeros(4)
                    # u = np.array([throttle, elevator, aileron, rudder])
                    
                    # 시뮬레이션 시간
                    t_end = float(input("\n시뮬레이션 시간 (s, 기본 5): ") or "5")
                    
                    # 시뮬레이션 실행
                    t, states = sim.run_simulation(X0, u, (0, t_end), dt=0.01)
                else:
                    # 트림 조건으로 실행
                    t, states = sim.run_simulation()
                    # t, states = sim.run_simulation(X_trim, u_trim, (0, t_sim), dt=0.01)
                
                if t is not None:
                    # 결과 출력
                    print(f"\n시뮬레이션 완료!")
                    print(f"  최종 속도: {states[-1, 0]:.2f} ft/s")
                    print(f"  최종 고도: {states[-1, 11]:.2f} ft")
                    print(f"  최종 받음각: {np.rad2deg(states[-1, 1]):.2f}°")
                    
                    # 그래프 생성
                    show_plots = input("\n그래프 표시? (Y/n): ").strip().lower()
                    
                    if show_plots != 'n':
                        figures = F16Plotter.plot_all(t, states)
                        
                        # 저장 여부
                        save_plots = input("\n그래프 저장? (y/N): ").strip().lower()
                        if save_plots == 'y':
                            F16Plotter.save_figures(figures, 'f16_simulation')
                        
                        plt.show()
            
            elif choice == '2':
                # 기존 트림 블록 (변경: 결과를 last_*에 저장)

                # 트림 조건 입력
                print("\n트림 조건 입력:")
                speed = float(input("  속도 (ft/s, 기본 500): ") or "500")
                alt = float(input("  고도 (ft, 기본 10000): ") or "10000")
                climb = float(input("  상승각 (deg, 기본 0): ") or "0")

                print("\n" + "="*70)
                print("트림 조건")
                print("="*70)
                
                # 트림 계산
                X_trim, u_trim, cost = sim.run_trim(alt, speed, climb)

                if X_trim is not None:
                    last_X_trim = X_trim.copy()
                    last_u_trim = u_trim.copy()
                    last_cost = cost

                if X_trim is not None:
                    # 트림 조건으로 시뮬레이션 실행 여부
                    run_sim = input("\n트림 조건으로 시뮬레이션 실행? (Y/n): ").strip().lower()
                    
                    if run_sim != 'n':
                        t_sim = float(input("  시뮬레이션 시간 (s, 기본 20): ") or "20")
                        
                        print("\n트림 조건으로 시뮬레이션 실행 중...")
                        t, states = sim.run_simulation(X_trim, u_trim, (0, t_sim), dt=0.01)
                        
                        if t is not None:
                            print(f"\n시뮬레이션 완료!")
                            
                            # 트림 검증
                            final_states = states[-100:, :]  # 마지막 100개 포인트
                            vt_std = np.std(final_states[:, 0])
                            alt_std = np.std(final_states[:, 11])
                            
                            print(f"\n트림 품질 검증 (마지막 {len(final_states)}개 포인트):")
                            print(f"  속도 표준편차: {vt_std:.4f} ft/s")
                            print(f"  고도 표준편차: {alt_std:.4f} ft")
                            
                            if vt_std < 1.0 and alt_std < 10.0:
                                print("  ✓ 트림 상태 유지 확인")
                            else:
                                print("  ⚠ 트림 상태가 완전하지 않음")
                            
                            # 그래프 생성
                            show_plots = input("\n그래프 표시? (Y/n): ").strip().lower()
                            
                            if show_plots != 'n':
                                figures = F16Plotter.plot_all(t, states)
                                
                                save_plots = input("\n그래프 저장? (y/N): ").strip().lower()
                                if save_plots == 'y':
                                    F16Plotter.save_figures(figures, 'f16_trim_simulation')
                                
                                plt.show()


            elif choice == '3':
                # 선형화 메뉴
                if last_X_trim is None or last_u_trim is None:
                    run_now = input("트림 결과가 없습니다. 지금 트림을 실행할까요? (Y/n): ").strip().lower()
                    if run_now != 'n':
                        speed = float(input("  속도 (ft/s, 기본 500): ") or "500")
                        alt = float(input("  고도 (ft, 기본 10000): ") or "10000")
                        climb = float(input("  상승각 (deg, 기본 0): ") or "0")
                        last_X_trim, last_u_trim, last_cost = sim.run_trim(alt, speed, climb)

                if last_X_trim is None or last_u_trim is None:
                    print("트림 결과 없음 — 선형화를 수행할 수 없습니다.")
                else:
                    print("\n선형화 계산 진행...")
                    A, B, C, D = Linearizer.linearize(F16Dynamics.derivs, last_X_trim, last_u_trim)

                    np.set_printoptions(precision=6, suppress=True)
                    print("\nA matrix:")
                    print(A)
                    print("\nB matrix:")
                    print(B)

                    # 세로축(longitudinal) 인덱스 및 입력
                    long_idx = [0, 1, 4, 7]   # VT, Alpha, Theta, Q
                    long_short_Phugoid_sym = ['Alpha', 'Q', 'VT', 'Theta',]
                    long_u_idx = [0, 1]       # Throttle, Elevator
                    A_long = A[np.ix_(long_idx, long_idx)]
                    B_long = B[np.ix_(long_idx, long_u_idx)]

                    print("\n--- Longitudinal (VT,Alpha,Theta,Q) ---")
                    print("A_long:")
                    print(A_long)
                    print("B_long:")
                    print(B_long)
                    kk = 0

                    # 획득: 고유값, 감쇠, 자연주파수
                    eigs_long, V_long = np.linalg.eig(A_long)
                    print("Longitudinal poles and modal parameters:")
                    for lam in eigs_long:
                        sigma = np.real(lam)
                        omega = np.imag(lam)
                        wn = np.hypot(sigma, omega)
                        zeta = -sigma / wn if wn > 0 else np.nan
                        #print(f" {long_short_Phugoid_sym[kk]} : pole={lam:.6g}, wn={wn:.6g}, zeta={zeta:.6g}")
                        print(f" pole={lam:.6g}, wn={wn:.6g}, zeta={zeta:.6g}")
                        kk += 1

 
                    # 가로/방향(lateral) 인덱스 및 입력
                    lat_idx = [2, 3, 6, 8]    # Beta, Phi, P, R
                    lat_Dutch_Roll_Spiral_sym = ['P', 'R', 'Phi', 'Beta']                    
                    lat_u_idx = [2, 3]        # Aileron, Rudder
                    A_lat = A[np.ix_(lat_idx, lat_idx)]
                    B_lat = B[np.ix_(lat_idx, lat_u_idx)]

                    print("\n--- Lateral/Directional (Beta,Phi,P,R) ---")
                    print("A_lat:")
                    print(A_lat)
                    print("B_lat:")
                    print(B_lat)

                    kk = 0
                    eigs_lat, _ = np.linalg.eig(A_lat)
                    print("\nLateral poles and modal parameters:")
                    for lam in eigs_lat:
                        sigma = np.real(lam)
                        omega = np.imag(lam)
                        wn = np.hypot(sigma, omega)
                        zeta = -sigma / wn if wn > 0 else np.nan
                        #print(f" {lat_Dutch_Roll_Spiral_sym[kk]} : pole={lam:.6g}, wn={wn:.6g}, zeta={zeta:.6g}")
                        print(f" pole={lam:.6g}, wn={wn:.6g}, zeta={zeta:.6g}")
                        kk += 1
            else:
                print("\n잘못된 입력입니다. 다시 선택하세요.")
        
        except ValueError:
            print("\n잘못된 입력 형식입니다. 숫자를 입력하세요.")
        except KeyboardInterrupt:
            print("\n\n사용자가 중단했습니다.")
            break
        except Exception as e:
            print(f"\n오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("프로그램 종료")
    print("="*70)

if __name__ == "__main__":
    main()