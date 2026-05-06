import numpy as np
import os
import astropy.io.fits as afits
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from scipy.ndimage import gaussian_filter1d

# 设置中文字体显示（macOS 常用）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False

# 项目路径与参数配置
KNOWN_DIR = './CNstars/'           # 原始化学奇异星 FITS 数据路径
OUTPUT_DIR = './Augmented_Data/'    # 增强后矩阵文件保存位置
WAVE_GRID = np.arange(3800, 5500, 1) # 标准重采样波长网格
C_SPEED = 299792.458                 # 光速 (km/s)
AUGMENT_MULTIPLIER = 50              # 每颗种子星生成的增强样本数量

os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_continuum(flux):
    """ 利用高斯平滑进行连续谱归一化 """
    return flux / np.maximum(gaussian_filter1d(flux, sigma=10), 1e-5)

def augment_spectrum(base_flux, wave_grid):
    """ 
    物理数据增强：包含随机观测噪声、多普勒抖动及连续谱红化误差 
    """
    aug_flux = np.copy(base_flux)
    
    # 注入噪声：模拟信噪比波动 (SNR range: 5 to 30)
    snr = np.random.uniform(5.0, 30.0)
    noise = np.random.normal(0, 1.0/snr, size=len(aug_flux))
    aug_flux += noise
    
    # 模拟多普勒偏移：-30 到 30 km/s 的随机视向速度抖动
    rv_jitter = np.random.uniform(-30.0, 30.0)
    jittered_wave = wave_grid * (1.0 + rv_jitter / C_SPEED)
    
    # 插值回标准波长网格
    spl = splrep(jittered_wave, aug_flux, k=3)
    aug_flux = splev(WAVE_GRID, spl)
    
    # 模拟连续谱倾斜：模拟消光/红化误差 (slope range: +/- 1e-4)
    tilt = 1.0 + np.random.uniform(-0.0001, 0.0001) * (WAVE_GRID - np.mean(WAVE_GRID))
    aug_flux *= tilt
    
    return aug_flux

# 加载种子光谱并预处理
seed_spectra = []
files = [f for f in os.listdir(KNOWN_DIR) if f.endswith(('.fits', '.fits.gz'))]

for filename in files:
    try:
        with afits.open(os.path.join(KNOWN_DIR, filename)) as hdus:
            data = hdus[1].data[0]
            header = hdus[0].header
            
            # 获取观测流量、波长及视向速度修正值
            rv = header.get('RV', 0.0)
            w_obs, f_obs = data['WAVELENGTH'], data['FLUX']
            
            # 转换至静止系并去除无效点
            w_rest = w_obs / (1.0 + rv / C_SPEED)
            mask = np.isfinite(w_rest) & np.isfinite(f_obs)
            w, f = w_rest[mask], f_obs[mask]
            
            if len(w) < 100: continue
            
            # 排序、重采样并归一化
            idx = np.argsort(w)
            flux_interp = splev(WAVE_GRID, splrep(w[idx], f[idx], k=3))
            seed_spectra.append(normalize_continuum(flux_interp))
    except Exception:
        continue

seed_spectra = np.array(seed_spectra)
print(f"提取种子星数量: {len(seed_spectra)}")

# 批量生成增强数据集
aug_data = []
labels = []

for seed in seed_spectra:
    # 保留原始种子数据
    aug_data.append(seed)
    labels.append(1)
    
    # 生成增强样本
    for _ in range(AUGMENT_MULTIPLIER):
        aug_data.append(augment_spectrum(seed, WAVE_GRID))
        labels.append(1)

# 保存为 NumPy 矩阵供机器学习模型直接读取
np.save(os.path.join(OUTPUT_DIR, 'CP_Features.npy'), np.array(aug_data))
np.save(os.path.join(OUTPUT_DIR, 'CP_Labels.npy'), np.array(labels))

# 数据可视化对比
plt.figure(figsize=(10, 5))
plt.plot(WAVE_GRID, seed_spectra[0], 'k-', lw=1.5, label='Original Seed')
for i in range(1, 4):
    plt.plot(WAVE_GRID, aug_data[i], alpha=0.5, label=f'Augmented sample {i}')

plt.xlim(4100, 4200) # 重点展示特征波段
plt.xlabel('Rest Wavelength (Å)')
plt.ylabel('Normalized Flux')
plt.title('Spectra Data Augmentation Comparison')
plt.legend()
plt.tight_layout()
plt.show()