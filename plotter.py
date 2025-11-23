"""
F-16 시뮬레이터 - 그래프 생성 모듈
plotter.py

13개 상태를 그래프로 시각화
"""

import numpy as np
import matplotlib.pyplot as plt


class F16Plotter:
    """F-16 시뮬레이션 결과 시각화 클래스"""
    
    @staticmethod
    def plot_states(t, states, title="F-16 6DOF Simulation Results"):
        """
        13개 상태 그래프 생성
        
        Parameters:
            t (np.ndarray): 시간 배열 (s)
            states (np.ndarray): 상태 궤적 (N x 13)
            title (str): 그래프 제목
        """
        # 13개 상태 레이블
        state_labels = [
            'VT (ft/s)',           # 0
            'Alpha (deg)',         # 1
            'Beta (deg)',          # 2
            'Phi (deg)',           # 3
            'Theta (deg)',         # 4
            'Psi (deg)',           # 5
            'P (deg/s)',           # 6
            'Q (deg/s)',           # 7
            'R (deg/s)',           # 8
            'North (ft)',          # 9
            'East (ft)',           # 10
            'Altitude (ft)',       # 11
            'Power (%)'            # 12
        ]
        
        # 단위 변환
        states_plot = states.copy()
        states_plot[:, 1:6] = np.rad2deg(states[:, 1:6])  # 각도
        states_plot[:, 6:9] = np.rad2deg(states[:, 6:9])  # 각속도
        
        # 4x4 서브플롯 생성 (13개 + 1개 여유)
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 각 상태 플롯
        for i in range(13):
            row = i // 4
            col = i % 4
            ax = axes[row, col]
            
            ax.plot(t, states_plot[:, i], 'b-', linewidth=1.5)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(state_labels[i])
            ax.set_title(f'State {i}: {state_labels[i].split("(")[0].strip()}')
        
        # 마지막 빈 서브플롯 제거
        fig.delaxes(axes[3, 3])
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_attitude(t, states, title="F-16 Attitude"):
        """
        자세각 (Phi, Theta, Psi) 그래프
        
        Parameters:
            t (np.ndarray): 시간 배열
            states (np.ndarray): 상태 궤적
            title (str): 그래프 제목
        """
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        attitude_labels = ['Roll (Phi)', 'Pitch (Theta)', 'Yaw (Psi)']
        
        for i in range(3):
            axes[i].plot(t, np.rad2deg(states[:, 3+i]), 'b-', linewidth=1.5)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylabel(f'{attitude_labels[i]} (deg)')
            axes[i].set_title(attitude_labels[i])
        
        axes[2].set_xlabel('Time (s)')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_velocity_altitude(t, states, title="F-16 Velocity and Altitude"):
        """
        속도와 고도 그래프
        
        Parameters:
            t (np.ndarray): 시간 배열
            states (np.ndarray): 상태 궤적
            title (str): 그래프 제목
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # 속도
        axes[0].plot(t, states[:, 0], 'b-', linewidth=1.5)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylabel('Velocity (ft/s)')
        axes[0].set_title('Total Velocity (VT)')
        
        # 고도
        axes[1].plot(t, states[:, 11], 'r-', linewidth=1.5)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylabel('Altitude (ft)')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_title('Altitude')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_angles(t, states, title="F-16 Alpha and Beta"):
        """
        받음각과 사이드슬립각 그래프
        
        Parameters:
            t (np.ndarray): 시간 배열
            states (np.ndarray): 상태 궤적
            title (str): 그래프 제목
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # 받음각
        axes[0].plot(t, np.rad2deg(states[:, 1]), 'b-', linewidth=1.5)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylabel('Alpha (deg)')
        axes[0].set_title('Angle of Attack')
        
        # 사이드슬립각
        axes[1].plot(t, np.rad2deg(states[:, 2]), 'r-', linewidth=1.5)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylabel('Beta (deg)')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_title('Sideslip Angle')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_trajectory_2d(states, title="F-16 Trajectory (Top View)"):
        """
        2D 궤적 플롯 (Top View)
        
        Parameters:
            states (np.ndarray): 상태 궤적
            title (str): 그래프 제목
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        north = states[:, 9]
        east = states[:, 10]
        
        ax.plot(east, north, 'b-', linewidth=2, label='Trajectory')
        ax.plot(east[0], north[0], 'go', markersize=10, label='Start')
        ax.plot(east[-1], north[-1], 'ro', markersize=10, label='End')
        
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('East (ft)')
        ax.set_ylabel('North (ft)')
        ax.set_title(title)
        ax.legend()
        ax.axis('equal')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_all(t, states):
        """
        모든 그래프를 한 번에 생성
        
        Parameters:
            t (np.ndarray): 시간 배열
            states (np.ndarray): 상태 궤적
            
        Returns:
            list: 생성된 figure 리스트
        """
        figures = []
        
        print("\n그래프 생성 중...")
        
        # 1. 전체 상태
        fig1 = F16Plotter.plot_states(t, states)
        figures.append(fig1)
        
        # 2. 자세각
        fig2 = F16Plotter.plot_attitude(t, states)
        figures.append(fig2)
        
        # 3. 속도/고도
        fig3 = F16Plotter.plot_velocity_altitude(t, states)
        figures.append(fig3)
        
        # 4. 받음각/사이드슬립각
        fig4 = F16Plotter.plot_angles(t, states)
        figures.append(fig4)
        
        # 5. 2D 궤적
        fig5 = F16Plotter.plot_trajectory_2d(states)
        figures.append(fig5)
        
        print(f"✓ {len(figures)}개 그래프 생성 완료!")
        
        return figures
    
    @staticmethod
    def save_figures(figures, prefix='f16_sim'):
        """
        그래프를 파일로 저장
        
        Parameters:
            figures (list): figure 리스트
            prefix (str): 파일 이름 접두사
        """
        filenames = [
            f'{prefix}_all_states.png',
            f'{prefix}_attitude.png',
            f'{prefix}_velocity_altitude.png',
            f'{prefix}_angles.png',
            f'{prefix}_trajectory.png'
        ]
        
        print(f"\n그래프 저장 중...")
        
        for i, (fig, filename) in enumerate(zip(figures, filenames)):
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"  ✓ {filename}")
        
        print(f"\n총 {len(figures)}개 파일 저장 완료!")
