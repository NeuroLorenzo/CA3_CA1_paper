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

recall_val = np.zeros((2, len(trials_exp), sh_trials))

for j, ntrains in enumerate(trials_exp):
    normal_s = normal_s_all[ntrains]

    for s in range(sh_trials):
        # Carica attività test per questo trial shuffle
        act_df_te      = pd.read_csv(trials_files[ntrains][s], header=None, names=['id', 'spike_time'])
        grouped_spikes = act_df_te.groupby('id')['spike_time'].apply(np.array).to_dict()

        # Conta spike per ogni neurone e time point
        n_spikes_tot = np.zeros((n_gid_stim, len(times)))
        for l, t in enumerate(times[[7,9]]):
            for i, g in enumerate(gid_stim):
                if g in grouped_spikes:
                    spikes = grouped_spikes[g]
                    n_spikes_tot[i, l] = np.sum((spikes >= t) & (spikes < t + window))

        # Carica id shuffle per questo trial
        retr_id = np.load(trials_id[ntrains][s], allow_pickle=True).item()
        print('retr_id',retr_id.keys())

        for i, r in enumerate(rec_p[[7,9]]*n_gid_stim):
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



for j, nstim in enumerate(rec_p[[7,9]]*n_gid_stim):
    colors = extract_colors('viridis', len(trials_exp))
    fig, ax = plt.subplots(figsize=(8, 6))
    bounds = (0, [np.inf, np.inf, np.inf, np.inf])


    y    = recall_avg[j, :]
    yerr = recall_std[j, :]

    # Sostituisci ax.plot e ax.fill_between con questo:
    ax.errorbar(
        trials_exp, y, yerr=yerr, 
        fmt='.',            # Disegna il punto come marcatore (sostituisce ax.plot)
        color=colors[j+1],    # Colore del punto e dell'asticella
        ecolor=colors[j+1],   # Colore specifico delle asticelle (opzionale)
        capsize=3,          # Larghezza dei trattini orizzontali in cima/fondo all'asticella
        elinewidth=1,       # Spessore della linea verticale dell'errore
        alpha=0.8           # Opzionale: rende il grafico leggermente più pul
    )
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('Training Trials')
    ax.set_ylabel('Recall Index')
    ax.set_title('Recall Index')
    plt.tight_layout()
    #plt.savefig(f'{sonata_folder}RI{nstim}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{sonata_folder}RI{nstim}.eps', dpi=300, bbox_inches='tight', format="eps")
