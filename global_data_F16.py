"""
F-16 시뮬레이터 - 전역 데이터 관리 모듈
global_data_F16.py

전역 변수와 공력 데이터를 관리하는 클래스
"""

import numpy as np
import scipy.io as sio
import os


class F16GlobalData:
    """F-16 시뮬레이션의 모든 전역 데이터를 관리하는 클래스"""
    
    def load_Conf_data(self):
        # 물리 상수
        try:
            self.R0 = 2.377e-3  # 해수면 밀도 (slug/ft³)
            self.GD = 32.17     # 중력 가속도 (ft/s²)
            
            # 항공기 관성 모멘트 (slug-ft²)
            self.AXX = 9496.0
            self.AYY = 55814.0
            self.AZZ = 63100.0
            self.AXZ = 982.0
            
            # 관성 모멘트 조합
            AXZS = self.AXZ ** 2
            self.XPQ = self.AXZ * (self.AXX - self.AYY + self.AZZ)
            self.GAM = self.AXX * self.AZZ - AXZS
            self.XQR = self.AZZ * (self.AZZ - self.AYY) + AXZS
            self.ZPQ = (self.AXX - self.AYY) * self.AXX + AXZS
            self.YPR = self.AZZ - self.AXX
            
            # 항공기 기하학적 파라미터
            self.Weight = 20490.466  # 중량 (lb)
            self.S = 300.0           # 날개 면적 (ft²)
            self.B = 30.0            # 날개 폭 (ft)
            self.CBAR = 11.32        # 평균 공력 시위 (ft)
            self.XCGR = 0.35         # 기준 무게중심 위치
            self.HX = 160.0          # 엔진 각운동량
            self.XCG = 0.3           # 실제 무게중심 위치
            
            # 공력 데이터 테이블
            self.CL_Data = None   # 롤링 모멘트 계수
            self.CM_Data = None   # 피칭 모멘트 계수
            self.CN_Data = None   # 요잉 모멘트 계수
            self.CX_Data = None   # X축 공력 계수
            self.CZ_Data = None   # Z축 공력 계수
            self.Damp_Data = None # 감쇠 미계수
            self.DLDA_Data = None # 에일러론 롤링 미계수
            self.DLDR_Data = None # 러더 롤링 미계수
            self.DNDA_Data = None # 에일러론 요잉 미계수
            self.DNDR_Data = None # 러더 요잉 미계수
            
            # 엔진 추력 데이터
            self.A_Data = None  # THRUST idle
            self.B_Data = None  # THRUST military
            self.C_Data = None  # THRUST max
            
            # 제어 입력 [throttle, elevator, aileron, rudder]
            self.u = np.zeros(4)
            
            # 트림 계산용 변수
            self.Temp_X = np.zeros(13)   # 임시 상태 벡터
            self.SS_flag = 0             # 정상상태 플래그
            self.tol = 1e-6              # 수렴 허용오차
            self.XXD = np.zeros(13)      # 상태 미분 벡터
            
            # 트림 조건 파라미터
            self.type = 1          # 비행 타입 (1=일반, 2=선회, 3=안정축)
            self.PHI = 0.0         # 롤 각 (rad)
            self.TR = 0.0          # 턴 레이트 (rad/s)
            self.SINGAM = 0.0      # sin(flight path angle)
            self.RADGAM = 0.0      # 비행경로각 (rad)
            self.RR = 0.0          # 롤 레이트 (rad/s)
            self.PR = 0.0          # 풀업 레이트 (rad/s)
            self.CPHI = 1.0        # cos(PHI)
            self.SPHI = 0.0        # sin(PHI)

            print("✓ 형상 데이터 로드 완료!")
            return True
        
        except Exception as e:
            print(f"✗ 형상 데이터 로드 실패: {e}")
            return False
        
    def load_aero_data(self, tables_path='Tables'):
        """
        Tables 폴더에서 공력 데이터를 로드
        
        Parameters:
            tables_path (str): .mat 파일들이 있는 폴더 경로
            
        Returns:
            bool: 로드 성공 여부
        """
        try:
            # 공력 계수 데이터 로드
            self.CL_Data = sio.loadmat(os.path.join(tables_path, 'CL_Data.mat'))['CL_Data']
            self.CM_Data = sio.loadmat(os.path.join(tables_path, 'CM_Data.mat'))['CM_Data']
            self.CN_Data = sio.loadmat(os.path.join(tables_path, 'CN_Data.mat'))['CN_Data']
            self.CX_Data = sio.loadmat(os.path.join(tables_path, 'CX_Data.mat'))['CX_Data']
            self.CZ_Data = sio.loadmat(os.path.join(tables_path, 'CZ_Data.mat'))['CZ_Data']
            
            # 공력 미계수 데이터 로드
            self.Damp_Data = sio.loadmat(os.path.join(tables_path, 'Damp_Data.mat'))['Damp_Data']
            self.DLDA_Data = sio.loadmat(os.path.join(tables_path, 'DLDA_Data.mat'))['DLDA_Data']
            self.DLDR_Data = sio.loadmat(os.path.join(tables_path, 'DLDR_Data.mat'))['DLDR_Data']
            self.DNDA_Data = sio.loadmat(os.path.join(tables_path, 'DNDA_Data.mat'))['DNDA_Data']
            self.DNDR_Data = sio.loadmat(os.path.join(tables_path, 'DNDR_Data.mat'))['DNDR_Data']
            
            print("✓ 공력 데이터 로드 완료!")
            return True

        except Exception as e:
            print(f"✗ 공력 데이터 로드 실패: {e}")
            print("\nTables 폴더에 다음 파일들이 있는지 확인하세요:")
            print("  - CL_Data.mat, CM_Data.mat, CN_Data.mat")
            print("  - CX_Data.mat, CZ_Data.mat")
            print("  - Damp_Data.mat")
            print("  - DLDA_Data.mat, DLDR_Data.mat")
            print("  - DNDA_Data.mat, DNDR_Data.mat")
            print("  - THRUST_Data.mat")
            return False
    
    def load_Engine_data(self, tables_path='Tables'):
        """
        Tables 폴더에서 엔진 데이터를 로드
        
        Parameters:
            tables_path (str): .mat 파일들이 있는 폴더 경로
            
        Returns:
            bool: 로드 성공 여부
        """
        try:
            # 엔진 추력 데이터 로드
            thrust_data = sio.loadmat(os.path.join(tables_path, 'THRUST_Data.mat'))
            self.A_Data = thrust_data['A_Data']
            self.B_Data = thrust_data['B_Data']
            self.C_Data = thrust_data['C_Data']
            print("✓ 엔진 데이터 로드 완료!")
            return True

        except Exception as e:
            print(f"✗ 엔진 데이터 로드 실패: {e}")
            print("\nTables 폴더에 다음 파일들이 있는지 확인하세요:")
            print("  - THRUST_Data.mat")
            return False

    def reset_trim_variables(self):
        """트림 계산 변수 초기화"""
        self.Temp_X = np.zeros(13)
        self.XXD = np.zeros(13)
        self.SS_flag = 0


# 전역 인스턴스 생성
gdata = F16GlobalData()
