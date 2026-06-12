import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from pathlib import Path
import sys
import re


sonata_folder = sys.argv[1]        #'./400_4000/'
train_activity_file = sys.argv[2]  #'activity_train_65.csv'
gid_stim_file       = sys.argv[3]  #'stim_gid_400.csv'
shuffle_folder      = sys.argv[4]  #'SH_new_2'
trials_exp          = sys.argv[5]  #[10,20,30,40,45,50,55,60]
trials_exp = [int(x) for x in trials_exp.split(',')]
def safe_sigmoid(x, L, x0, k, b):
    x_clipped = np.clip(x - x0, -500/k if k != 0 else -500, 500/k if k != 0 else 500)
    return L / (1 + np.exp(-k * x_clipped)) + b

def extract_colors(colormap_name, n_colors):
    cmap = plt.get_cmap(colormap_name)
    values = np.linspace(1, 0, n_colors)
    return cmap(values)




gid_stimdf = pd.read_csv('data/'+gid_stim_file, header=0, names=['gid'], dtype=np.uint32)
gid_stim   = sorted(gid_stimdf['gid'].tolist())
gid_stim_set = set(gid_stim)
n_gid_stim = len(gid_stim)

print('n_gid_stim',n_gid_stim)
# rec   = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 200, 250, 300])
# rec_p = rec / n_gid_stim
rec_p = np.array([0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25,
                  0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.45, 0.5, 0.625, 0.75])
times = np.array([1000, 6000, 11000, 16000, 21000, 26000, 31000,36000,41000,46000,51000,
                56000,61000,66000,71000,76000,81000,86000,91000,96000])
window = 20


trials_files = {t: [] for t in trials_exp}
trials_id = {t: [] for t in trials_exp}

base_dir = Path(sonata_folder + shuffle_folder)


activity_pattern = re.compile(r'activity_(\d+)_test(\d+)\.csv')

for path in base_dir.glob('activity_*'):
    match = activity_pattern.search(path.name)
    if match:
        sh_num = int(match.group(1))   # ID shuffle (es. 7)
        trial_num = int(match.group(2)) # ID trial (es. 30)
        
        if trial_num in trials_exp:
            trials_files[trial_num].append((sh_num, str(path)))


for t in trials_exp:
    trials_files[t].sort(key=lambda x: x[0])
    trials_files[t] = [path for sh_num, path in trials_files[t]]



id_pattern = re.compile(r'id_sh_(\d+)_(\d+)\.npy')

for path in base_dir.glob('id_sh_*'):
    match = id_pattern.search(path.name)
    if match:
        sh_num = int(match.group(1))
        trial_num = int(match.group(2))
        
        if trial_num in trials_exp:
            trials_id[trial_num].append((sh_num, str(path)))


for t in trials_exp:
    trials_id[t].sort(key=lambda x: x[0])
    trials_id[t] = [path for sh_num, path in trials_id[t]]


sh_trials = len(trials_files[trials_exp[0]]) if trials_exp and trials_files[trials_exp[0]] else 0

print(f"Shuffle trials trovati: {sh_trials}  {trials_files}")
print(f"Shuffle ids trovati: {trials_id}")
act_df_tr = pd.read_csv(sonata_folder+train_activity_file, header=None, names=['id', 'spike_time'])

normal_s_all = {}
for trials in trials_exp:
    val_k = 0
    for k in range(2):
        t1 = trials * 200 - 200 * (k + 1)
        n_spkstot = np.zeros(n_gid_stim)
        for i, g in enumerate(gid_stim):
            spikes_tr, _ = np.histogram(
                act_df_tr.loc[act_df_tr['id'] == g, 'spike_time'].values,
                range(t1, t1 + window), window
            )
            n_spkstot[i] = np.sum(spikes_tr)
        val_k += np.sum(n_spkstot)
    normal_s_all[trials] = val_k / n_gid_stim / 2
    print(f"normal_s [{trials} epochs]: {normal_s_all[trials]:.4f}")

recall_val = np.zeros((len(rec_p), len(trials_exp), sh_trials))

for j, ntrains in enumerate(trials_exp):
    normal_s = normal_s_all[ntrains]

    for s in range(sh_trials):
        # Carica attività test per questo trial shuffle
        act_df_te      = pd.read_csv(trials_files[ntrains][s], header=None, names=['id', 'spike_time'])
        grouped_spikes = act_df_te.groupby('id')['spike_time'].apply(np.array).to_dict()

        # Conta spike per ogni neurone e time point
        n_spikes_tot = np.zeros((n_gid_stim, len(times)))
        for l, t in enumerate(times):
            for i, g in enumerate(gid_stim):
                if g in grouped_spikes:
                    spikes = grouped_spikes[g]
                    n_spikes_tot[i, l] = np.sum((spikes >= t) & (spikes < t + window))

        # Carica id shuffle per questo trial
        retr_id = np.load(trials_id[ntrains][s], allow_pickle=True).item()
        print('retr_id',retr_id.keys())
        
        for i, r in enumerate(rec_p*n_gid_stim):
            #neuroni stimolati 
            gid_stim_rec = retr_id[str(int(r))]

            # Indici dei neuroni NON stimolati
            idx_rec = np.setdiff1d(
                    np.arange(n_gid_stim),
                    [gid_stim.index(g) for g in gid_stim_rec if g in gid_stim_set]
            )
            n_non_stim = len(idx_rec)

            # Spike dei neuroni non stimolati al tempo i
            spikes_selected = n_spikes_tot[idx_rec, i]

            #recall index nuovo
            recall_val[i, j, s] = ((np.sum(spikes_selected) / n_non_stim / normal_s) + (np.count_nonzero(spikes_selected) / n_non_stim)) / 2

recall_avg = np.mean(recall_val, axis=2)
recall_std = np.std(recall_val,  axis=2) / np.sqrt(sh_trials)

colors = extract_colors('viridis', len(trials_exp))
fig, ax = plt.subplots(figsize=(8, 6))
bounds = (0, [np.inf, np.inf, np.inf, np.inf])

for j, ntrains in enumerate(trials_exp):
    y    = recall_avg[:, j]
    yerr = recall_std[:, j]

    ax.plot(rec_p, y, '.', color=colors[j])
    ax.fill_between(rec_p, y - yerr, y + yerr, color=colors[j], alpha=0.2)

    try:
        p0   = [max(y), np.median(rec_p), 10, min(y)]
        popt, _ = curve_fit(safe_sigmoid, rec_p, y, p0=p0, bounds=bounds)
        x_fit = np.linspace(min(rec_p), max(rec_p), 350)
        ax.plot(x_fit, safe_sigmoid(x_fit, *popt), '-', color=colors[j], label=f"{ntrains} epochs")
    except Exception as e:
        print(f"Fit fallito per {ntrains} epochs: {e}")
        ax.plot(rec_p, y, '-', color=colors[j], label=f"{ntrains} epochs")

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel('Fraction of stimulated neurons')
ax.set_ylabel('Recall Index')
ax.set_title('Recall Index')
plt.tight_layout()
plt.savefig(f'{sonata_folder}recall_shuffle_sum.eps', dpi=300, bbox_inches='tight', format="eps")

