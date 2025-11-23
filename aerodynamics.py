"""
F-16 시뮬레이터 - 공력 계수 및 미계수 모듈
aerodynamics.py

공력 계수(CX, CY, CZ, CL, CM, CN) 및
공력 미계수(DAMP, DLDA, DLDR, DNDA, DNDR) 계산
"""

import numpy as np
from global_data_F16 import gdata


class AeroCoefficients:
    """F-16 공력 계수 계산 클래스"""
    
    @staticmethod
    def ADC(VT, ALT):
        """
        대기 데이터 계산기 (Air Data Computer)
        
        Parameters:
            VT (float): 전체 속도 (ft/s)
            ALT (float): 고도 (ft)
            
        Returns:
            tuple: (AMACH, QBAR) - 마하수, 동압 (lb/ft²)
        """
        R0 = 2.377e-3  # 해수면 밀도
        
        TFAC = 1.0 - 0.703e-5 * ALT
        T = 519.0 * TFAC  # 온도 (°R)
        
        if ALT >= 35000.0:
            T = 390.0
        
        RHO = R0 * TFAC ** 4.14  # 밀도
        AMACH = VT / np.sqrt(1.4 * 1716.3 * T)  # 마하수
        QBAR = 0.5 * RHO * VT * VT  # 동압
        
        return AMACH, QBAR
    
    @staticmethod
    def CX(Alpha, El):
        """
        X축 공력 계수 (추력축 방향)
        
        Parameters:
            Alpha (float): 받음각 (deg)
            El (float): 엘리베이터 (deg)
            
        Returns:
            float: CX 계수
        """
        s = 0.2 * Alpha
        k = int(np.fix(s))
        if k <= -2: k = -1
        if k >= 9: k = 8
        
        DA = s - k
        L = k + int(np.fix(1.1 * np.sign(DA)))
        
        s = El / 12.0
        M = int(np.fix(s))
        if M <= -2: M = -1
        if M >= 2: M = 1
        
        DE = s - M
        N = M + int(np.fix(1.1 * np.sign(DE)))
        
        T = gdata.CX_Data[k + 2, M + 2]
        U = gdata.CX_Data[k + 2, N + 2]
        V = T + abs(DA) * (gdata.CX_Data[L + 2, M + 2] - T)
        W = U + abs(DA) * (gdata.CX_Data[L + 2, N + 2] - U)
        
        return V + (W - V) * abs(DE)
    
    @staticmethod
    def CY(Beta, AIL, RDR):
        """
        Y축 공력 계수 (측면 방향)
        
        Parameters:
            Beta (float): 사이드슬립각 (deg)
            AIL (float): 에일러론 (deg)
            RDR (float): 러더 (deg)
            
        Returns:
            float: CY 계수
        """
        return -0.02 * Beta + 0.021 * (AIL / 20.0) + 0.086 * (RDR / 30.0)
    
    @staticmethod
    def CZ(Alpha, Beta, El):
        """
        Z축 공력 계수 (양력축 방향)
        
        Parameters:
            Alpha (float): 받음각 (deg)
            Beta (float): 사이드슬립각 (deg)
            El (float): 엘리베이터 (deg)
            
        Returns:
            float: CZ 계수
        """
        s = 0.2 * Alpha
        k = int(np.fix(s))
        if k <= -2: k = -1
        if k >= 9: k = 8
        
        DA = s - k
        L = k + int(np.fix(1.1 * np.sign(DA)))
        
        s = gdata.CZ_Data[k + 2] + abs(DA) * (gdata.CZ_Data[L + 2] - gdata.CZ_Data[k + 2])
        
        return s * (1 - (Beta / 57.3) ** 2) - 0.19 * (El / 25.0)
    
    @staticmethod
    def CL(Alpha, Beta):
        """
        롤링 모멘트 계수
        
        Parameters:
            Alpha (float): 받음각 (deg)
            Beta (float): 사이드슬립각 (deg)
            
        Returns:
            float: CL 계수
        """
        s = 0.2 * Alpha
        k = int(np.fix(s))
        if k <= -2: k = -1
        if k >= 9: k = 8
        
        dA = s - k
        l = k + int(np.fix(1.1)) if dA >= 0 else k + int(np.fix(-1.1))
        
        s = 0.2 * abs(Beta)
        m = int(np.fix(s))
        if m == 0: m = 1
        if m >= 6: m = 5
        
        dB = s - m
        n = m + int(np.fix(1.1)) if dB >= 0 else m + int(np.fix(-1.1))
        
        t = gdata.CL_Data[k + 2, m]
        U = gdata.CL_Data[k + 2, n]
        V = t + abs(dA) * (gdata.CL_Data[l + 2, m] - t)
        W = U + abs(dA) * (gdata.CL_Data[l + 2, n] - U)
        
        dum = V + (W - V) * abs(dB)
        return dum if Beta >= 0 else -dum
    
    @staticmethod
    def CM(Alpha, El):
        """
        피칭 모멘트 계수
        
        Parameters:
            Alpha (float): 받음각 (deg)
            El (float): 엘리베이터 (deg)
            
        Returns:
            float: CM 계수
        """
        s = 0.2 * Alpha
        k = int(np.fix(s))
        if k <= -2: k = -1
        if k >= 9: k = 8
        
        DA = s - k
        L = k + int(np.fix(1.1 * np.sign(DA)))
        
        s = El / 12.0
        M = int(np.fix(s))
        if M <= -2: M = -1
        if M >= 2: M = 1
        
        DE = s - M
        N = M + int(np.fix(1.1 * np.sign(DE)))
        
        T = gdata.CM_Data[k + 2, M + 2]
        U = gdata.CM_Data[k + 2, N + 2]
        V = T + abs(DA) * (gdata.CM_Data[L + 2, M + 2] - T)
        W = U + abs(DA) * (gdata.CM_Data[L + 2, N + 2] - U)
        
        return V + (W - V) * abs(DE)
    
    @staticmethod
    def CN(Alpha, Beta):
        """
        요잉 모멘트 계수
        
        Parameters:
            Alpha (float): 받음각 (deg)
            Beta (float): 사이드슬립각 (deg)
            
        Returns:
            float: CN 계수
        """
        s = 0.2 * Alpha
        k = int(np.fix(s))
        if k <= -2: k = -1
        if k >= 9: k = 8
        
        dA = s - k
        l = k + int(np.fix(1.1)) if dA >= 0 else k + int(np.fix(-1.1))
        
        s = 0.2 * abs(Beta)
        m = int(np.fix(s))
        if m == 0: m = 1
        if m >= 6: m = 5
        
        dB = s - m
        n = m + int(np.fix(1.1)) if dB >= 0 else m + int(np.fix(-1.1))
        
        t = gdata.CN_Data[k + 2, m]
        U = gdata.CN_Data[k + 2, n]
        V = t + abs(dA) * (gdata.CN_Data[l + 2, m] - t)
        W = U + abs(dA) * (gdata.CN_Data[l + 2, n] - U)
        
        dum = V + (W - V) * abs(dB)
        return dum if Beta >= 0 else -dum


class AeroDerivatives:
    """F-16 공력 미계수 계산 클래스"""
    
    @staticmethod
    def DAMP(Alpha):
        """
        감쇠 미계수
        
        Parameters:
            Alpha (float): 받음각 (deg)
            
        Returns:
            np.ndarray: 9개 감쇠 미계수 [CXq, CYr, CYp, CZq, Clr, Clp, Cmq, Cnr, Cnp]
        """
        s = 0.2 * Alpha
        K = int(np.fix(s))
        if K <= -2: K = -1
        if K >= 9: K = 8
        
        DA = s - K
        L = K + int(np.fix(1.1 * np.sign(DA)))
        
        D = np.zeros(9)
        for I in range(9):
            D[I] = gdata.Damp_Data[K + 2, I] + abs(DA) * (gdata.Damp_Data[L + 2, I] - gdata.Damp_Data[K + 2, I])
        
        return D
    
    @staticmethod
    def DLDA(Alpha, Beta):
        """
        에일러론에 의한 롤링 모멘트 미계수
        
        Parameters:
            Alpha (float): 받음각 (deg)
            Beta (float): 사이드슬립각 (deg)
            
        Returns:
            float: DLDA 미계수
        """
        S = 0.2 * Alpha
        K = int(np.fix(S))
        if K <= -2: K = -1
        if K >= 9: K = 8
        
        DA = S - K
        L = K + int(np.fix(1.1 * np.sign(DA)))
        
        S = 0.1 * Beta
        M = int(np.fix(S))
        if M <= -3: M = -2
        if M >= 3: M = 2
        
        DB = S - M
        N = M + int(np.fix(1.1 * np.sign(DB)))
        
        T = gdata.DLDA_Data[K + 2, M + 3]
        U = gdata.DLDA_Data[K + 2, N + 3]
        V = T + abs(DA) * (gdata.DLDA_Data[L + 2, M + 3] - T)
        W = U + abs(DA) * (gdata.DLDA_Data[L + 2, N + 3] - U)
        
        return V + (W - V) * abs(DB)
    
    @staticmethod
    def DLDR(Alpha, Beta):
        """
        러더에 의한 롤링 모멘트 미계수
        
        Parameters:
            Alpha (float): 받음각 (deg)
            Beta (float): 사이드슬립각 (deg)
            
        Returns:
            float: DLDR 미계수
        """
        S = 0.2 * Alpha
        K = int(np.fix(S))
        if K <= -2: K = -1
        if K >= 9: K = 8
        
        DA = S - K
        L = K + int(np.fix(1.1 * np.sign(DA)))
        
        S = 0.1 * Beta
        M = int(np.fix(S))
        if M <= -3: M = -2
        if M >= 3: M = 2
        
        DB = S - M
        N = M + int(np.fix(1.1 * np.sign(DB)))
        
        T = gdata.DLDR_Data[K + 2, M + 3]
        U = gdata.DLDR_Data[K + 2, N + 3]
        V = T + abs(DA) * (gdata.DLDR_Data[L + 2, M + 3] - T)
        W = U + abs(DA) * (gdata.DLDR_Data[L + 2, N + 3] - U)
        
        return V + (W - V) * abs(DB)
    
    @staticmethod
    def DNDA(Alpha, Beta):
        """
        에일러론에 의한 요잉 모멘트 미계수
        
        Parameters:
            Alpha (float): 받음각 (deg)
            Beta (float): 사이드슬립각 (deg)
            
        Returns:
            float: DNDA 미계수
        """
        s = 0.2 * Alpha
        k = int(np.fix(s))
        if k <= -2: k = -1
        if k >= 9: k = 8
        
        DA = s - k
        L = k + int(np.fix(1.1 * np.sign(DA)))
        
        s = 0.1 * Beta
        M = int(np.fix(s))
        if M <= -3: M = -2
        if M >= 3: M = 2
        
        DB = s - M
        N = M + int(np.fix(1.1 * np.sign(DB)))
        
        T = gdata.DNDA_Data[k + 2, M + 3]
        U = gdata.DNDA_Data[k + 2, N + 3]
        V = T + abs(DA) * (gdata.DNDA_Data[L + 2, M + 3] - T)
        W = U + abs(DA) * (gdata.DNDA_Data[L + 2, N + 3] - U)
        
        return V + (W - V) * abs(DB)
    
    @staticmethod
    def DNDR(Alpha, Beta):
        """
        러더에 의한 요잉 모멘트 미계수
        
        Parameters:
            Alpha (float): 받음각 (deg)
            Beta (float): 사이드슬립각 (deg)
            
        Returns:
            float: DNDR 미계수
        """
        s = 0.2 * Alpha
        k = int(np.fix(s))
        if k <= -2: k = -1
        if k >= 9: k = 8
        
        DA = s - k
        L = k + int(np.fix(1.1 * np.sign(DA)))
        
        s = 0.1 * Beta
        M = int(np.fix(s))
        if M <= -3: M = -2
        if M >= 3: M = 2
        
        DB = s - M
        N = M + int(np.fix(1.1 * np.sign(DB)))
        
        T = gdata.DNDR_Data[k + 2, M + 3]
        U = gdata.DNDR_Data[k + 2, N + 3]
        V = T + abs(DA) * (gdata.DNDR_Data[L + 2, M + 3] - T)
        W = U + abs(DA) * (gdata.DNDR_Data[L + 2, N + 3] - U)
        
        return V + (W - V) * abs(DB)
