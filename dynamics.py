"""
F-16 시뮬레이터 - 6자유도 운동 방정식 모듈
dynamics.py

F-16 6자유도 비선형 운동 방정식
"""

import numpy as np
from scipy.integrate import solve_ivp

from global_data_F16 import gdata
from aerodynamics import AeroCoefficients, AeroDerivatives
from engine import EngineModel


class F16Dynamics:
    """6자유도 비선형 운동 방정식 클래스"""
    
    @staticmethod
    def derivs(t, X):
        """
        6자유도 상태 미분 방정식
        
        Parameters:
            t (float): 시간 (s)
            X (np.ndarray): 상태 벡터 (13개)
                [VT, Alpha, Beta, Phi, Theta, Psi, P, Q, R, PN, PE, Alt, Pow]
        
        Returns:
            np.ndarray: 상태 미분 벡터 (13개)
        """
        XD = np.zeros(13)
        MASS = gdata.Weight / gdata.GD
        
        # 상태 변수 할당
        VT = X[0]      # 전체 속도 (ft/s)
        ALPHA = np.rad2deg(X[1])  # 받음각 (deg)
        BETA = np.rad2deg(X[2])   # 사이드슬립각 (deg)
        PHI = X[3]     # 롤 각 (rad)
        THETA = X[4]   # 피치 각 (rad)
        PSI = X[5]     # 요 각 (rad)
        P = X[6]       # 롤 레이트 (rad/s)
        Q = X[7]       # 피치 레이트 (rad/s)
        R = X[8]       # 요 레이트 (rad/s)
        ALT = X[11]    # 고도 (ft)
        POW = X[12]    # 엔진 파워 (%)
        
        # 제어 입력
        THTL = gdata.u[0]  # 스로틀 (0-1)
        EL = gdata.u[1]    # 엘리베이터 (deg)
        AIL = gdata.u[2]   # 에일러론 (deg)
        RDR = gdata.u[3]   # 러더 (deg)
        
        # 대기 데이터 및 엔진 모델
        AMACH, QBAR = AeroCoefficients.ADC(VT, ALT)
        CPOW = EngineModel.TGEAR(THTL)
        XD[12] = EngineModel.PDOT(POW, CPOW)
        T = EngineModel.THRUST(POW, ALT, AMACH)
        
        # 공력 계수 계산
        CXT = AeroCoefficients.CX(ALPHA, EL)
        CYT = AeroCoefficients.CY(BETA, AIL, RDR)
        CZT = AeroCoefficients.CZ(ALPHA, BETA, EL)
        
        # 조종면 입력 정규화
        DAIL = AIL / 20.0
        DRDR = RDR / 30.0
        
        # 롤링 모멘트 계수
        CLT = AeroCoefficients.CL(ALPHA, BETA) + \
              AeroDerivatives.DLDA(ALPHA, BETA) * DAIL + \
              AeroDerivatives.DLDR(ALPHA, BETA) * DRDR
        
        # 피칭 모멘트 계수
        CMT = AeroCoefficients.CM(ALPHA, EL)
        
        # 요잉 모멘트 계수
        CNT = AeroCoefficients.CN(ALPHA, BETA) + \
              AeroDerivatives.DNDA(ALPHA, BETA) * DAIL + \
              AeroDerivatives.DNDR(ALPHA, BETA) * DRDR
        
        # 감쇠 미계수 추가
        TVT = 0.5 / VT
        B2V = gdata.B * TVT
        CQ = gdata.CBAR * Q * TVT
        
        D = AeroDerivatives.DAMP(ALPHA)
        
        CXT = CXT + CQ * D[0]
        CYT = CYT + B2V * (D[1] * R + D[2] * P)
        CZT = CZT + CQ * D[3]
        CLT = CLT + B2V * (D[4] * R + D[5] * P)
        CMT = CMT + CQ * D[6] + CZT * (gdata.XCGR - gdata.XCG)
        CNT = CNT + B2V * (D[7] * R + D[8] * P) - \
              CYT * (gdata.XCGR - gdata.XCG) * (gdata.CBAR / gdata.B)
        
        # 동체 속도 성분
        CBTA = np.cos(X[2])
        U = VT * np.cos(X[1]) * CBTA
        V = VT * np.sin(X[2])
        W = VT * np.sin(X[1]) * CBTA
        
        # 삼각함수
        STH = np.sin(THETA)
        CTH = np.cos(THETA)
        SPH = np.sin(PHI)
        CPH = np.cos(PHI)
        SPSI = np.sin(PSI)
        CPSI = np.cos(PSI)
        
        # 무차원 힘 및 모멘트
        QS = QBAR * gdata.S
        QSB = QS * gdata.B
        RMQS = QS / MASS
        GCTH = gdata.GD * CTH
        QSPH = Q * SPH
        AY = RMQS * CYT
        AZ = RMQS * CZT
        
        # 힘 방정식 (동체 축)
        UDOT = R * V - Q * W - gdata.GD * STH + (QS * CXT + T) / MASS
        VDOT = P * W - R * U + GCTH * SPH + AY
        WDOT = Q * U - P * V + GCTH * CPH + AZ
        DUM = (U * U + W * W)
        
        # 공기 데이터 미분
        XD[0] = (U * UDOT + V * VDOT + W * WDOT) / VT   # VT_dot
        XD[1] = (U * WDOT - W * UDOT) / DUM             # Alpha_dot
        XD[2] = (VT * VDOT - V * XD[0]) * CBTA / DUM    # Beta_dot
        
        # 모멘트 방정식
        ROLL = QSB * CLT
        PITCH = QS * gdata.CBAR * CMT
        YAW = QSB * CNT
        
        PQ = P * Q
        QR = Q * R
        QHX = Q * gdata.HX
        
        XD[6] = (gdata.XPQ * PQ - gdata.XQR * QR + gdata.AZZ * ROLL + \
                 gdata.AXZ * (YAW + QHX)) / gdata.GAM  # P_dot
        XD[7] = (gdata.YPR * P * R - gdata.AXZ * (P * P - R * R) + \
                 PITCH - R * gdata.HX) / gdata.AYY      # Q_dot
        XD[8] = (gdata.ZPQ * PQ - gdata.XPQ * QR + gdata.AXZ * ROLL + \
                 gdata.AXX * (YAW + QHX)) / gdata.GAM   # R_dot
        
        # 운동학 방정식 (자세각 미분)
        XD[3] = P + (STH / CTH) * (QSPH + R * CPH)  # Phi_dot
        XD[4] = Q * CPH - R * SPH                    # Theta_dot
        XD[5] = (QSPH + R * CPH) / CTH               # Psi_dot
        
        # 네비게이션 방정식 (위치 미분)
        T1 = SPH * CPSI
        T2 = CPH * STH
        T3 = SPH * SPSI
        S1 = CTH * CPSI
        S2 = CTH * SPSI
        S3 = T1 * STH - CPH * SPSI
        S4 = T3 * STH + CPH * CPSI
        S5 = SPH * CTH
        S6 = T2 * CPSI + T3
        S7 = T2 * SPSI - T1
        S8 = CPH * CTH
        
        XD[9] = U * S1 + V * S3 + W * S6   # PN_dot (북쪽)
        XD[10] = U * S2 + V * S4 + W * S7  # PE_dot (동쪽)
        XD[11] = U * STH - V * S5 - W * S8  # Alt_dot (고도, 음수)
        
        # 상태 미분 저장 (트림 계산용)
        gdata.XXD = XD.copy()
        
        return XD
    
    @staticmethod
    def simulate(X0, u_input, t_span, dt=0.01):
        """
        6자유도 시뮬레이션 실행 (ODE45 적분)
        
        Parameters:
            X0 (np.ndarray): 초기 상태 (13개)
            u_input (np.ndarray): 제어 입력 (4개) [throttle, elevator, aileron, rudder]
            t_span (tuple): 시간 범위 (t_start, t_end)
            dt (float): 시간 간격 (s)
            
        Returns:
            tuple: (t, states)
                t: 시간 배열
                states: 상태 궤적 (N x 13)
        """
        # 제어 입력 설정
        gdata.u = u_input.copy()
        
        # 시간 평가 점
        t_eval = np.arange(t_span[0], t_span[1], dt)
        
        # ODE 적분 (RK45 = MATLAB ode45 equivalent)
        sol = solve_ivp(
            F16Dynamics.derivs,
            t_span,
            X0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6,
            atol=1e-9
        )
        
        return sol.t, sol.y.T
