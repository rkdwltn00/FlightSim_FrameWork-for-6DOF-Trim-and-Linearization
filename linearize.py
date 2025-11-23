import numpy as np
from global_data_F16 import gdata

class Linearizer:
    """
    수치 선형화: A,B,C,D 계산
    사용법:
        A,B,C,D = Linearizer.linearize(F16Dynamics.derivs, X_trim, u_trim)
    """
    @staticmethod
    def linearize(derivs_func, X_trim, u_trim, eps_state=1e-6, eps_input=1e-6):
        # 백업
        u_backup = gdata.u.copy()
        X_backup = None
        try:
            # 보장: gdata.u를 사용하도록 u_trim 적용
            gdata.u = np.array(u_trim, dtype=float).copy()
            x0 = np.array(X_trim, dtype=float).copy()
            n = x0.size
            m = gdata.u.size

            # 기준 f0
            f0 = np.asarray(derivs_func(0.0, x0), dtype=float).reshape(n,)

            A = np.zeros((n, n), dtype=float)
            B = np.zeros((n, m), dtype=float)

            # 상태에 대해 중앙차분
            for j in range(n):
                dx = eps_state if abs(x0[j]) < 1.0 else eps_state * abs(x0[j])
                x_plus = x0.copy(); x_minus = x0.copy()
                x_plus[j] += dx
                x_minus[j] -= dx
                f_plus = np.asarray(derivs_func(0.0, x_plus), dtype=float).reshape(n,)
                f_minus = np.asarray(derivs_func(0.0, x_minus), dtype=float).reshape(n,)
                A[:, j] = (f_plus - f_minus) / (2.0 * dx)

            # 입력에 대해 중앙차분 (gdata.u 사용 inside derivs)
            for k in range(m):
                du = eps_input if abs(gdata.u[k]) < 1.0 else eps_input * abs(gdata.u[k])
                u_plus = gdata.u.copy(); u_minus = gdata.u.copy()
                u_plus[k] += du
                u_minus[k] -= du

                # 적용 및 계산
                gdata.u = u_plus.copy()
                f_plus = np.asarray(derivs_func(0.0, x0), dtype=float).reshape(n,)
                gdata.u = u_minus.copy()
                f_minus = np.asarray(derivs_func(0.0, x0), dtype=float).reshape(n,)

                B[:, k] = (f_plus - f_minus) / (2.0 * du)

            # 출력: 상태 그대로
            C = np.eye(n)
            D = np.zeros((n, m))

            # 복원
            return A, B, C, D

        finally:
            gdata.u = u_backup.copy()