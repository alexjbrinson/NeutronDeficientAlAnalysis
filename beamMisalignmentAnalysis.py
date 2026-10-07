import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def randomBiasAnalysis(dataPath):
  labelsize=16
  ticklabelsize=14
  plt.figure()
  data=pd.read_csv(f'{dataPath}/CalibrationDiagnostics/calibrationFrame_afterBEC.csv')
  data['weight']=1/data['cent_uncertainty']**2
  data['weightedVal']=data['weight']*data['centroid']
  mu = np.sum(data['weightedVal'])/np.sum(data['weight'])
  tbins=np.linspace(min(data['avgScanTime'])-15000, max(data['avgScanTime']+10000), 12)
  tbins[6:10]-=8000
  for t in tbins:
    plt.gca().axvline(x=t, linestyle='--', color='k', alpha=0.25)
  tCuts=pd.cut(data.loc[:,"avgScanTime"], bins=tbins)
  grouped = data.groupby(tCuts)
  weighted_mean = grouped.apply(lambda g: (g['centroid'] * g['weight']).sum() / g['weight'].sum())
  weighted_std = grouped.apply(lambda g: np.sqrt(1 / g['weight'].sum()))
  avg_time=grouped.apply(lambda g: g['avgScanTime'].mean())
  for interval, group in grouped:
    # plt.plot(group['avgScanTime'], group['centroid'],'.',alpha=0.5)
    plt.errorbar(x=group['avgScanTime'],y=group['centroid'], yerr=group['cent_uncertainty'],marker='.', linestyle='None',alpha=0.25)
  print('np.std(weighted_mean)**2:',np.std(weighted_mean)**2)
  plt.errorbar(x=avg_time, y=weighted_mean, yerr=weighted_std, marker='.', linestyle='None', color='k')
  w_groups = 1 / weighted_std**2
  mu = np.sum(w_groups * weighted_mean) / np.sum(w_groups)
  plt.gca().axhline(y=mu)
  obs_var = np.sum(w_groups * (weighted_mean - mu)**2) / np.sum(w_groups)
  stat_var = np.sum(w_groups * weighted_std**2) / np.sum(w_groups)
  drift_var = max(0, obs_var - stat_var)
  drift_std = np.sqrt(drift_var)
  print('obs_var:',obs_var)
  print('stat_var:',stat_var)
  print('drift_std:',drift_std) #this *could* be due to relative beam misalignment
  plt.xticks(tbins, ["%d"%t for t in (tbins-tbins[0])/3600],fontsize=ticklabelsize)
  plt.yticks(fontsize=ticklabelsize)
  plt.xlabel("Time since exp start (h)", fontsize=labelsize)
  plt.ylabel("Energy corrected calibration centroid (MHz)",fontsize=labelsize)
  plt.savefig(f'{dataPath}/randomBiasAnalysis.png')
  plt.show()
  return(drift_std)

dataPath='D:/MIT Dropbox/Alex Brinson/Research/FRIB/Al/OnlineAlAnalysis/results/Voigt/equal_fwhm_False/cec_sim_toggle_False'
randomBiasAnalysis(dataPath)
plt.show()