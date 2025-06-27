#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
from scipy.interpolate import make_interp_spline


DATA_PATH = 'c:\\Users\\김영성\\Desktop\\PeptoidGen-main\\PeptoidGen_ver2025\\1. data'
PGM_PATH= 'c:\\Users\\김영성\\Desktop\\PeptoidGen-main\\\PeptoidGen_ver2025\\2. pgm'
RESULT_PATH = 'c:\\Users\\김영성\\Desktop\\PeptoidGen-main\\PeptoidGen_ver2025\\3. result'
SAVED_WEIGHT_PATH = f'c:\\Users\\김영성\\Desktop\\PeptoidGen-main\\PeptoidGen_ver2025\\1. data\\saved_model_{20250626}'

# result_df1= pd.read_csv(os.path.join(RESULT_PATH,'TFR_with_buffer_exponential.csv'))
# result_df2= pd.read_csv(os.path.join(RESULT_PATH,'TFR_with_buffer_exponential2_from926.csv'))

#result_df = pd.concat([result_df1[:925], result_df2], ignore_index=True)
result_df = pd.read_csv(os.path.join(RESULT_PATH,'TFR_with_buffer_exponential_1500개.csv')).head(1400)

#%%
# Smoothing 함수 정의
# Smoothing 함수 정의
def smooth(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# 원본 데이터
rewards = result_df['reward'].values
reward_stds = result_df['reward_std'].values
good_samples = result_df['good_samples_num'].values

# Smoothing
rewards_smooth = smooth(rewards)
reward_stds_smooth = smooth(reward_stds)
good_samples_smooth = smooth(good_samples)

episodes_smooth = np.arange(len(rewards_smooth))
episodes_gs_smooth = np.arange(len(good_samples_smooth))


# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))
fig.suptitle('The training result of TFR (Exponential reward)', fontsize=40)

# 왼쪽: Reward + Std
ax1.plot(episodes_smooth, rewards_smooth, label='Reward', color='blue')
# ax1.fill_between(
#     episodes_smooth,
#     rewards_smooth - reward_stds_smooth,
#     rewards_smooth + reward_stds_smooth,
#     color='blue',
#     alpha=0.8,
#     label='±1 Std')
ax1.set_xlim(0, len(rewards_smooth)+10)
ax1.set_ylim(0,)
ax1.set_xlabel('Episode', fontsize=25)
ax1.set_ylabel('Reward', fontsize=25)
ax1.set_title('Reward over Episodes', fontsize=30)
ax1.grid(True)
ax1.legend(fontsize=20, loc= 'upper left')
ax1.tick_params(axis='both', labelsize=20)
for label in ax1.get_xticklabels():
    label.set_rotation(45)

# 오른쪽: Good sample 수
ax2.plot(episodes_gs_smooth, good_samples_smooth, color='green', label='Smoothed # of good samples')
ax2.set_xlim(0, len(rewards_smooth)+10)
ax2.set_ylim(0, 130)
ax2.set_xlabel('Episode', fontsize=25)
ax2.set_ylabel('# of good samples', fontsize=25)
ax2.set_title('Number of Good Samples over Episodes', fontsize=30)
ax2.grid(True)
ax2.legend(fontsize=20, loc= 'upper left')
ax2.tick_params(axis='both', labelsize=20)
for label in ax2.get_xticklabels():
    label.set_rotation(45)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(RESULT_PATH, 'TFR_with_buffer_exponential_1500개.png'), dpi=300)
plt.show()
# %%
