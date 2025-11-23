"""
F-16 시뮬레이터 - 시뮬레이션 실행 모듈
simulator.py

1. 6자유도 비선형 시뮬레이션
2. 트림 계산 (COST 함수 최소화)
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from engine import EngineModel

from global_data_F16 import gdata
from dynamics import F16Dynamics

class TrimCalculator:
    """트림 조건 계산 클래스"""
    
    @staticmethod
    def CONSTR(X):
        """
        트림 제약조건 적용
        
        Parameters:
            X (np.ndarray): 상태 벡터 (13개)
            
        Returns:
            np.ndarray: 제약조건이 적용된 상태 벡터
        """
        XC = X.copy()
        CALPH = np.cos(X[1])
        SALPH = np.sin(X[1])
        CBETA = np.cos(X[2])
        
        if gdata.type == 2:  # 선회 비행
            pass
        elif gdata.TR != 0.0:  # 턴이 있을 때
            pass
        else:  # 일반 비행
            XC[3] = gdata.PHI
            D = X[1]
            
            if gdata.PHI != 0.0:
                D = -X[1]  # 역전
            
            if gdata.SINGAM != 0.0:
                SGOCB = gdata.SINGAM / CBETA
                XC[4] = D + np.arctan(SGOCB / np.sqrt(1.0 - SGOCB**2))
            else:
                XC[4] = D  # 수평 비행
            
            XC[6] = gdata.RR   # P
            XC[7] = gdata.PR   # Q
            
            if gdata.type == 3:  # 안정축 롤
                XC[8] = gdata.RR * SALPH / CALPH
            else:
                XC[8] = 0.0  # 동체축 롤
        
        return XC
    
    @staticmethod
    def COST(S):
        """
        트림 비용 함수 (개선 버전)


        목표: 모든 상태 미분을 0으로 만들기
        
        Parameters:
            S (np.ndarray): 최적화 변수 (6개)
                [throttle, elevator, alpha, aileron, rudder, beta]
                
        Returns:
            float: 비용 (상태 미분의 가중 제곱합)
        """

        penalty = 1e8

        # 백업(원래 전역값을 복원하기 위해)
        u_backup = gdata.u.copy()
        X_backup = gdata.Temp_X.copy()

        try:
            # 로컬 제어/상태 설정 (전역을 바로 덮어쓰지 않음 -> 잠시 반영)
            local_u = u_backup.copy()
            local_u[0] = S[0]       # Throttle
            local_u[1] = S[1]       # Elevator
            local_u[2] = S[3]       # Aileron
            local_u[3] = S[4]       # Rudder

            # 로컬 상태 초기값
            local_X = X_backup.copy()
            local_X[1] = S[2]       # Alpha (rad)
            local_X[2] = S[5]       # Beta (rad)
            local_X[12] = EngineModel.TGEAR(S[0])  # Engine Power

            # 제약조건 적용 (입력으로만 작동)
            local_X = TrimCalculator.CONSTR(local_X)

            # dynamics.derivs는 gdata.u를 참조할 수 있으므로 임시로 전역 u만 설정
            gdata.u = local_u.copy()

            # 정상상태 수렴 루프 (로컬 변수만 갱신)
            converged = False
            max_iterations = 5
            iteration = 0

            while not converged and iteration < max_iterations:
                try:
                    sol = solve_ivp(
                        F16Dynamics.derivs,
                        [0, 0.01],
                        local_X,
                        method='RK45',
                        rtol=1e-6,
                        atol=1e-6,
                        max_step=0.005
                    )
                except Exception:
                    return penalty

                if not getattr(sol, "success", True) or sol.y is None or sol.y.size == 0:
                    return penalty

                local_Xnew = sol.y[:, -1]

                # 비유한값/무한대 검사
                if not np.all(np.isfinite(local_Xnew)):
                    return penalty

                if np.linalg.norm(local_Xnew - local_X) < gdata.tol:
                    converged = True

                local_X = local_Xnew.copy()
                iteration += 1

            # 정상상태에서의 상태미분을 직접 계산 (부작용 없이 사용)
            XD = F16Dynamics.derivs(0.0, local_X)

            # 상태별 가중치 (원래 코드와 동일)
            weights = np.array([
                1,      # VT_dot
                100.0,  # Alpha_dot
                100.0,  # Beta_dot
                0.0,    # Phi_dot
                0.0,    # Theta_dot
                0.0,    # Psi_dot
                10.0,   # P_dot
                10.0,   # Q_dot
                10.0,   # R_dot
                0.0,    # PN_dot
                0.0,    # PE_dot
                0.0,    # Alt_dot
                0.0     # Pow_dot
            ])

            CLF16 = np.sum((XD * weights) ** 2)

            if not np.isfinite(CLF16):
                return penalty

            return float(CLF16)

        finally:
            # 반드시 전역 상태 복원
            gdata.u = u_backup.copy()
            gdata.Temp_X = X_backup.copy()
    
    @staticmethod
    def TRIMMER(X, u):
        """
        트림 계산 메인 함수
        
        Parameters:
            X (np.ndarray): 초기 상태 추정값 (13개)
            u (np.ndarray): 초기 제어 추정값 (4개)
            
        Returns:
            tuple: (X_trim, u_trim, cost)
                X_trim: 트림 상태 (13개)
                u_trim: 트림 제어 (4개)
                cost: 최종 비용
        """
        gdata.Temp_X = X.copy()
        
        # 초기 추정값
        S0 = np.array([u[0], u[1], X[1], u[2], u[3], X[2]])
        
        # 최적화 경계
        bounds = [
            (0.0, 1.0),                         # Throttle
            (-25.0, 25.0),                      # Elevator (deg)
            (np.deg2rad(-10), np.deg2rad(15)),  # Alpha (rad)
            (-21.5, 21.5),                      # Aileron (deg)
            (-30.0, 30.0),                      # Rudder (deg)
            (np.deg2rad(-10), np.deg2rad(10))   # Beta (rad)
        ]
        
        print("트림 최적화 시작...")
        print(f"초기: Throttle={S0[0]:.4f}, Elev={S0[1]:.2f}°, α={np.rad2deg(S0[2]):.2f}°")
        
        initial_cost = TrimCalculator.COST(S0)
        print(f"초기 비용: {initial_cost:.6e}\n")
        
        options_trust = {'maxiter': 1000, 'gtol': 1e-8, 'xtol': 1e-8, 'verbose': 0}
        options_slsqp = {'maxiter': 1000, 'ftol': 1e-10, 'disp': False}
    
        # outer loop: minimize를 여러 번 수행하여 cost가 목표 이하가 될 때까지 S0 갱신
        cost_tol = 1e-5            # 목표 비용 (필요시 조정)
        max_outer_iters = 100      # 반복 최대 시도 횟수
        outer_iter = 0
        result = None
        S_current = S0.copy()
        fval = np.inf

        while outer_iter < max_outer_iters and fval > cost_tol:
            try:
                # result = minimize(TrimCalculator.COST, S_current, method='trust-constr',
                #                   bounds=bounds, options=options_trust)
                result = minimize(TrimCalculator.COST, S_current, method='SLSQP',
                    bounds=bounds, options=options_slsqp)

                S_current = result.x.copy()
                fval = result.fun if hasattr(result, 'fun') else np.inf

                print(f"[outer {outer_iter}] cost={fval:.6e}")

            except Exception:
                pass

            # except Exception:
            #     # trust-constr 실패 시 SLSQP로 시도
            #     result = minimize(TrimCalculator.COST, S_current, method='SLSQP',
            #                       bounds=bounds, options=options_slsqp)

            # # 만약 최적화가 실패했거나 비정상 값이면 SLSQP로 재시도
            # if result is None or not hasattr(result, 'x'):
            #     print("Warning: optimizer returned invalid result; aborting trim outer loop")
            #     break

            # S_current = result.x.copy()
            # fval = result.fun if hasattr(result, 'fun') else np.inf

            # print(f"[outer {outer_iter}] cost={fval:.6e}")

            # # 만약 trust-constr 결과가 충분히 작지 않으면 SLSQP로 한 번 더 정제
            # if fval > cost_tol:
            #     try:
            #         result = minimize(TrimCalculator.COST, S_current, method='SLSQP',
            #                           bounds=bounds, options=options_slsqp)
            #         S_current = result.x.copy()
            #         fval = result.fun
            #         print(f"[outer {outer_iter}] SLSQP refine cost={fval:.6e}")
            #     except Exception:
            #         pass

            outer_iter += 1

        # 최종 결과 적용
        S = S_current
        # fval에는 마지막 계산된 비용
        print(f"\n최종 outer_iters={outer_iter}, 최종 cost={fval:.6e}")

        # 결과 적용
        u[0] = S[0]
        u[1] = S[1]
        gdata.Temp_X[1] = S[2]
        u[2] = S[3]
        u[3] = S[4]
        gdata.Temp_X[2] = S[5]
        
        X_Result = gdata.Temp_X.copy()
        XD = gdata.XXD
        
        # 결과 출력
        print(f"\n{'='*70}")
        print("트림 계산 결과")
        print(f"{'='*70}")
        print(f"최종 비용: {fval:.6e}")
        print(f"비용 감소: {(1 - fval/initial_cost)*100:.2f}%")
        
        if fval < 1e-10:
            print(f"\n✓✓✓ 탁월한 수렴!")
        elif fval < 1e-6:
            print(f"\n✓✓ 우수한 수렴")
        elif fval < 1e-3:
            print(f"\n✓ 양호한 수렴")
        else:
            print(f"\n⚠ 수렴 불량")
        
        print(f"\n{'='*70}")
        print("제어 입력")
        print(f"{'='*70}")
        print(f"Throttle:  {u[0]:.6f}")
        print(f"Elevator:  {u[1]:.4f}°")
        print(f"Aileron:   {u[2]:.4f}°")
        print(f"Rudder:    {u[3]:.4f}°")
        
        print(f"\n{'='*70}")
        print("상태")
        print(f"{'='*70}")
        print(f"Velocity:  {X_Result[0]:.2f} ft/s")
        print(f"Alpha:     {np.rad2deg(X_Result[1]):.4f}°")
        print(f"Beta:      {np.rad2deg(X_Result[2]):.4f}°")
        print(f"Altitude:  {X_Result[11]:.2f} ft")
        
        print(f"\n{'='*70}")
        print("상태 미분 (0에 가까워야 함)")
        print(f"{'='*70}")
        print(f"VT_dot:    {XD[0]:.4e}")
        print(f"Alpha_dot: {np.rad2deg(XD[1]):.4e}°/s")
        print(f"Alt_dot:   {XD[11]:.4e} ft/s")
        print(f"{'='*70}\n")
        
        return X_Result, u, fval


class F16Simulator:
    """F-16 시뮬레이터 메인 클래스"""
    
    def __init__(self):
        self.initialized = False
    
    def initialize(self, tables_path='Tables'):
        """
        시뮬레이터 초기화
        
        Parameters:
            tables_path (str): .mat 파일 경로
            
        Returns:
            bool: 초기화 성공 여부
        """
        print("F-16 시뮬레이터 초기화 중...")
        if gdata.load_Conf_data() & gdata.load_aero_data(tables_path) & gdata.load_Engine_data(tables_path) :
            self.initialized = True
            print("초기화 완료!\n")
            return True
        
        return False
    
    def run_simulation(self, X0=None, u=None, t_span=(0, 10), dt=0.01):
        """
        6자유도 시뮬레이션 실행
        
        Parameters:
            X0 (np.ndarray): 초기 상태 (None이면 기본값 사용)
            u (np.ndarray): 제어 입력 (None이면 기본값 사용)
            t_span (tuple): 시간 범위 (s)
            dt (float): 시간 간격 (s)
            
        Returns:
            tuple: (t, states) 시간 배열, 상태 궤적
        """
        if not self.initialized:
            print("먼저 initialize()를 실행하세요!")
            return None, None
        
        print("\n=== 6자유도 비선형 시뮬레이션 ===\n")
        
        # 기본 초기 조건
        if X0 is None:
            X0 = np.zeros(13)
            X0[0] = 500.0              # VT
            X0[1] = np.deg2rad(5.0)    # Alpha
            X0[11] = 10000.0           # Alt
            X0[12] = 10.0              # Pow
        #     X0 = X_trim

        # # 기본 제어 입력
        if u is None:
            u = np.array([0.5, 0.0, 0.0, 0.0])
        #     u = u_trim

        print(f"초기 조건: VT={X0[0]:.0f} ft/s, Alt={X0[11]:.0f} ft")
        print(f"제어 입력: Throttle={u[0]:.2f}, Elev={u[1]:.2f}°\n")
        
        t, states = F16Dynamics.simulate(X0, u, t_span, dt)
        
        print(f"시뮬레이션 완료! ({len(t)}개 시간 스텝)\n")
        
        return t, states
    
    def run_trim(self, altitude=10000.0, speed=500.0, climb_angle=0.0,
                 roll_rate=0.0, pullup_rate=0.0, turn_rate=0.0):
        """
        트림 계산
        
        Parameters:
            altitude (float): 고도 (ft)
            speed (float): 속도 (ft/s)
            climb_angle (float): 상승각 (deg)
            roll_rate (float): 롤 레이트 (rad/s)
            pullup_rate (float): 풀업 레이트 (rad/s)
            turn_rate (float): 턴 레이트 (rad/s)
            
        Returns:
            tuple: (X_trim, u_trim, cost)
        """
        if not self.initialized:
            print("먼저 initialize()를 실행하세요!")
            return None, None, None
        
        print(f"고도: {altitude} ft, 속도: {speed} ft/s")
        print(f"상승각: {climb_angle}°\n")
        
        # 트림 파라미터 설정
        gdata.RADGAM = np.deg2rad(climb_angle)
        gdata.RR = roll_rate
        gdata.PR = pullup_rate
        gdata.TR = turn_rate
        
        if turn_rate != 0:
            gdata.PHI = np.arctan2(speed**2, (speed/turn_rate) * gdata.GD)
        else:
            gdata.PHI = 0.0
        
        gdata.CPHI = np.cos(gdata.PHI)
        gdata.SPHI = np.sin(gdata.PHI)
        gdata.SINGAM = np.sin(gdata.RADGAM)
        gdata.type = 1
        
        # 트림 계산을 위한 초기 값 설정
        X = np.zeros(13)
   
        alphaRad = np.deg2rad(3)
        betaRad = np.deg2rad(0.0)
        power = 40

        X = np.array([speed,alphaRad,betaRad,0,alphaRad,0,0,0,0,0,0,altitude,power])
        u = np.array([0.15, -2, 0.0, 0.0])
        
        X_trim, u_trim, cost = TrimCalculator.TRIMMER(X, u)
        
        return X_trim, u_trim, cost