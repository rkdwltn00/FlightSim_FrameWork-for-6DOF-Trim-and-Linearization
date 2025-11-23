"""
F-16 시뮬레이터 - 엔진 모델 모듈
engine.py

엔진 추력 계산 및 파워 dynamics
"""

import numpy as np
from global_data_F16 import gdata


class EngineModel:
    """F-16 엔진 모델 클래스"""
    
    @staticmethod
    def TGEAR(THTL):
        """
        스로틀과 파워 커맨드의 관계
        
        Parameters:
            THTL (float): 스로틀 위치 (0-1)
            
        Returns:
            float: 파워 커맨드 (%)
        """
        if THTL <= 0.77:
            return 64.94 * THTL
        return 217.38 * THTL - 117.38
    
    @staticmethod
    def RTAU(DPOW):
        """
        파워 변화에 대한 시간 상수 계산
        
        Parameters:
            DPOW (float): 파워 변화량 (%)
            
        Returns:
            float: 시간 상수
        """
        if DPOW <= 25.0:
            return 1.0
        elif DPOW >= 50.0:
            return 0.1
        return 1.9 - 0.036 * DPOW
    
    @staticmethod
    def PDOT(POW, CPOW):
        """
        파워 변화율 계산
        
        Parameters:
            POW (float): 현재 파워 (%)
            CPOW (float): 파워 커맨드 (%)
            
        Returns:
            float: 파워 변화율 (%/s)
        """
        if CPOW >= 50.0:
            if POW >= 50.0:
                T = 5.0
                P2 = CPOW
            else:
                P2 = 60.0
                T = EngineModel.RTAU(P2 - POW)
        else:
            if POW >= 50.0:
                T = 5.0
                P2 = 40.0
            else:
                P2 = CPOW
                T = EngineModel.RTAU(P2 - POW)
        
        return T * (P2 - POW)
    
    @staticmethod
    def THRUST(Pow, Alt, Amach):
        """
        엔진 추력 계산
        
        Parameters:
            Pow (float): 현재 파워 (%)
            Alt (float): 고도 (ft)
            Amach (float): 마하수
            
        Returns:
            float: 추력 (lb)
        """
         # 테이블 크기
        n_rows, n_cols = gdata.B_Data.shape        # 고도와 마하수에 따른 인덱스 계산

        h = 0.0001 * Alt
        i = int(np.fix(h))
        if i >= 5:
            i = 4
        
        dh = h - i
        rm = 5 * Amach
        m = int(np.fix(rm))
        if m >= 5:
            m = 4
        
        dm = rm - m
        cdh = 1 - dh
        
        # i, m, dh가 기존 코드에서 계산되었다고 가정
        i = int(i)
        m = int(m)

        # i, m 이 i+1, m+1 접근을 안전하게 허용하도록 클램프
        i = max(0, min(i, n_rows - 2))
        m = max(0, min(m, n_cols - 2))

        # dh(행 보간 계수)가 있다면 0..1로 제한
        dh = float(dh) if 'dh' in locals() else 0.0
        dh = max(0.0, min(1.0, dh))
        cdh = 1.0 - dh

        # Military power 추력
        s = gdata.B_Data[i, m] * cdh + gdata.B_Data[i + 1, m] * dh
        #s = gdata.B_Data[i_clamped, m_clamped] * cdh + gdata.B_Data[i_clamped + 1, m_clamped] * dh
        t = gdata.B_Data[i, m + 1] * cdh + gdata.B_Data[i + 1, m + 1] * dh
        tmil = s + (t - s) * dm
    

        if Pow < 50:
            # Idle ~ Military power
            s = gdata.A_Data[i, m] * cdh + gdata.A_Data[i + 1, m] * dh
            t = gdata.A_Data[i, m + 1] * cdh + gdata.A_Data[i + 1, m + 1] * dh
            tidl = s + (t - s) * dm
            return tidl + (tmil - tidl) * Pow * 0.02
        else:
            # Military ~ Maximum power
            s = gdata.C_Data[i, m] * cdh + gdata.C_Data[i + 1, m] * dh
            t = gdata.C_Data[i, m + 1] * cdh + gdata.C_Data[i + 1, m + 1] * dh
            tmax = s + (t - s) * dm
            return tmil + (tmax - tmil) * (Pow - 50) * 0.02
