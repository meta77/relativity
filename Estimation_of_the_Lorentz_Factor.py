import numpy as np
import matplotlib.pyplot as plt

# 定数
c = 299792458  # 光速 (m/s)
muon_lifetime = 2.2e-6  # ミューオンの静止寿命 (秒)
atmosphere_height = 10000  # 大気圏の高さ (メートル)

# ミューオンの速度 (光速の99%)
v = 0.99 * c

# ミューオンの崩壊をシミュレーションする関数
def simulate_muons(num_muons, v, muon_lifetime, atmosphere_height, gamma):
    # ミューオンが大気圏を通過する時間
    time_to_earth = atmosphere_height / v

    # 時間の遅れを考慮したミューオンの寿命
    dilated_lifetime = gamma * muon_lifetime

    # ミューオンの崩壊をランダムにシミュレーション
    decay_times = np.random.exponential(dilated_lifetime, num_muons)
    survived_muons = np.sum(decay_times > time_to_earth)

    return survived_muons

# ローレンツ因子を推定する関数
def estimate_lorentz_factor(num_muons, v, muon_lifetime, atmosphere_height, observed_muons):
    # ローレンツ因子の範囲を設定
    gamma_values = np.linspace(1, 20, 1000)

    # 各γに対してシミュレーションを実行
    survived_muons_values = np.array([
        simulate_muons(num_muons, v, muon_lifetime, atmosphere_height, gamma)
        for gamma in gamma_values
    ])

    # 観測されたミューオン数に最も近いγを選択
    gamma_estimate = gamma_values[np.argmin(np.abs(survived_muons_values - observed_muons))]

    return gamma_estimate

# シミュレーションのパラメータ
num_muons = 10000  # 初期のミューオン数
gamma_theoretical = 1 / np.sqrt(1 - (v**2 / c**2))  # 理論的なローレンツ因子
observed_muons = simulate_muons(num_muons, v, muon_lifetime, atmosphere_height, gamma_theoretical)  # 理論値に基づく観測数

# ローレンツ因子を推定
gamma_estimate = estimate_lorentz_factor(num_muons, v, muon_lifetime, atmosphere_height, observed_muons)

# 結果の表示
print(f"理論的なローレンツ因子: {gamma_theoretical:.2f}")
print(f"推定されたローレンツ因子: {gamma_estimate:.2f}")

# シミュレーション結果を可視化
gamma_values = np.linspace(1, 20, 1000)
survived_muons_values = np.array([
    simulate_muons(num_muons, v, muon_lifetime, atmosphere_height, gamma)
    for gamma in gamma_values
])

plt.figure(figsize=(10, 6))
plt.plot(gamma_values, survived_muons_values, label="Simulation Results", color="blue")
plt.axhline(observed_muons, color="red", linestyle="--", label="The Lorentz factor is at its theoretical value")
plt.axvline(gamma_estimate, color="green", linestyle="--", label=f"Estimated Lorentz Factor: {gamma_estimate:.2f}")
plt.xlabel("The Lorentz Factor (γ)")
plt.ylabel("Number of Muons Observed on the Ground")
plt.title("Estimation of the Lorentz Factor")
plt.legend()
plt.grid()
plt.show()
