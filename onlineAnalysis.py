import numpy as np
import pandas as pd
import pickle, os, copy
import matplotlib.pyplot as plt
import RAP.DataMunger as dm
import RAP.HelperFunctions as hf
import RAP.FittingFunctions as ff
import RAP.BeamEnergyAnalysis as bea
import RAP.SpectrumClass as spc
import RAP.SpectrumHandler as sh

wmResidualError = 3
scanTimeOffset=1716156655
# freqOffset=1129900000
absChargeRad27 = 3.0614#fm
absChargeRad27_uncertainty = 0.0029#fm
runsDictionary = {
  22:[16463,16464,16465,16478,16479,16480,16481,16482,16483,16484,16485,16486,16487,16488,16489,16490,16491,16497,16498,16499,16500,16501,16502,16503,16504,16505],#16464
  23:[16405,16406,16407,16408,16414,16415,16416,16418,16419,16420,16421], #16404
  24:[16434,16435,16436,16437,16438,16445,16446,16447,16448,16449,16450], #16445,16446 have different buncher settings
  25:[16384,16385,16386,16387,16388],
  27:[16368,16369,16370,16389,16391,16392,16395,16396,16397,16410,16412,16413,16422,16424,16425,16426,#16367, 16371,16372,16373,16374,16375,16376 are all trash #16428 not good #16366 a little weird
      16429,16430,16439,16441,16442,16451,16458,16459,16470,16473,16474,16477,16492,16494,16495,16508,16510,16512]}
jGround=0.5
jExcited=0.5
iNucDictionary={
  22:[4], #spin assignment is tentative
  23:[2.5],
  24:[4,1], #isomer is spin 1
  25:[2.5],
  26:[5,0], #isomer is spin 0
  27:[2.5]}
massDictionary = {
  22:22.01942311,#previous mass value: 22.01954000,# 
  23:23.00724440,
  24:23.99994760, #this is for ground state; isomer excited by 425.81 (10) keV
  25:24.99042831,
  #26:25.98689188,
  27:26.981538408}

mass_uncertaintyDictionary = {
  22:0.00000030,# aka .3keV old uncertainty: 400keV = 0.0004 amu,
  23:0.00000040,
  24:0.00000024, #this is for ground state; isomer excitation uncertainty = .1 keV
  25:0.00000007,
  #26:,
  27:0.00000005}

laserDictionary = {
  22:375.990796,
  23:376.004732,
  24:376.017863,
  25:376.030178,
  27:376.052850}
  
timeStepDictionary= {
  22:[450,530],
  23:[450,530],
  24:[450,530],
  25:[450,530],
  26:[450,530],
  27:[489,543]}#[485,550]}

tofDictionary={
  22: [21.50E-6,23.5E-6],
  23: [21.80E-6,24.0E-6],
  24: [22.00E-6,24.5E-6],
  25: [22.75E-6,25.0E-6],
  27: [23.45E-6,26.0E-6]}

class WhatToRun:
  def __init__(self):
    self.fitAndLogToggle_BEA =                False;#True;#
    self.exportSpectrumToggle_calibration =   False;#True;#
    self.fitAndLogToggle_calibration =        False;#True;#
    self.exportSpectrumToggle_calibration_bec=False;#True;#
    self.fitAndLogToggle_calibration_bec=     False;#True;#
    self.exportSpectrumToggle     =           False;#True;#
    self.exportSpectrumToggle_bec =           False;#True;#
    self.fitAndLogToggleDic={  22:            False,#True,#
                               23:            False,#True,#
                               24:            False,#True,#
                               25:            False,#True,#
                               27:            False}#True}#

def randomBiasAnalysis(dataPath):
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
  plt.savefig(f'{dataPath}/randomBiasAnalysis.png')
  plt.close()
  return(drift_std)

def calibrationProcedure(calibrationScans, v0, δv0, spectrumKwargs={}, fittingKwargs={}):
  calibrationFrame = pd.DataFrame()
  mass=spectrumKwargs['mass']
  laserFreq=spectrumKwargs['laserFrequency']
  for run in calibrationScans:
    print('run%d'%run)
    spec=spc.Spectrum(runs=[run], targetDirectory=f'Scan{run}', **spectrumKwargs)           
    spec.fitAndLogData(**fittingKwargs); popFrame=spec.populateFrame(prefix="iso0",index=run)
    fa= spec.resultParams['iso0_centroid'].value; δfa= spec.resultParams['iso0_centroid'].stderr
    ΔEkin =bea.calculateBeamEnergyCorrectionFromv0vc(mass, laserFreq, fa, v0)
    δΔEkin=bea.propagateBeamEnergyCorrectionUncertainties([mass,0], [laserFreq,1], [fa, δfa], [v0,δv0])
    popFrame['ΔEkin']=ΔEkin; popFrame['ΔEkin_uncertainty']=δΔEkin
    calibrationFrame=pd.concat([calibrationFrame, popFrame])
  calibrationScanTimes=np.array(calibrationFrame['avgScanTime'])
  calibrationVsScanNumber = bea.getCalibrationFunction(v0, δv0, calibrationFrame, np.array(calibrationFrame.index),mass, laserFreq)
  calibrationVsScanTime   = bea.getCalibrationFunction(v0, δv0, calibrationFrame, calibrationScanTimes, mass, laserFreq)
  directoryPrefix=spectrumKwargs['directoryPrefix']
  energyCorrected=spectrumKwargs['energyCorrection'] if 'energyCorrection' in spectrumKwargs.keys() else False
  '''exporting calibration results for analysis comparision purposes'''
  exportsPrefix='./'+directoryPrefix+'/CalibrationDiagnostics/'
  if energyCorrected==False:
    if not os.path.isdir(exportsPrefix): os.makedirs(exportsPrefix)
    with open(exportsPrefix+'calibrationVsScanNumber_fit_report.txt','w') as file: file.write(calibrationVsScanNumber.fit_report()); file.close()
    with open(exportsPrefix+'calibrationVsScanTime_fit_report.txt','w') as file: file.write(calibrationVsScanTime.fit_report()); file.close()
    with open(exportsPrefix+'calibrationRelevantConstants.txt','w',encoding='utf-8') as file: file.write(f'scanTimeOffset: {scanTimeOffset}\nv0: {v0}\nδv0: {δv0}' );file.close()
    calibrationFrame[['centroid','cent_uncertainty', 'ΔEkin','ΔEkin_uncertainty','avgScanTime']].to_csv(exportsPrefix+'calibrationFunctionData.csv')
    plt.title('calibration run beam energy corrections'); plt.xlabel('run'); plt.ylabel('energy corrections (eV)')
    plt.errorbar(calibrationFrame.index, y=calibrationFrame['ΔEkin'], yerr=calibrationFrame['ΔEkin_uncertainty'], fmt='k.', label='individual runs')
    #plt.gca().axhline(y=allIsotopesFrame.loc[stableIndex]['centroid'], label='all runs combined')~
    plt.plot(calibrationFrame.index, calibrationVsScanNumber.best_fit, 'b-', label = 'linear correction as function of run number')
    plt.legend(loc=2)
    plt.savefig(exportsPrefix+'/energy_correctionsVsRunNumber.png');plt.close()

    plt.title('calibration run beam energy corrections'); plt.xlabel('time (s)'); plt.ylabel('energy corrections (eV)')
    plt.errorbar(calibrationFrame['avgScanTime'], y=calibrationFrame['ΔEkin'], yerr=calibrationFrame['ΔEkin_uncertainty'], fmt='k.', label='individual runs')
    #plt.gca().axhline(y=allIsotopesFrame.loc[stableIndex]['centroid'], label='all runs combined')
    plt.plot(calibrationFrame['avgScanTime'], calibrationVsScanTime.best_fit, 'b-', label = 'linear correction as function of time')
    plt.legend(loc=2)
    plt.savefig(exportsPrefix+'/energy_correctionsVsRunTime.png');plt.close()
  else:
    plt.title('calibration run A Ratios'); plt.xlabel('run'); plt.ylabel(r'$A_{Lower}/A_{Upper}$')
    plt.errorbar(calibrationFrame.index, y=calibrationFrame['aRatio'], yerr=calibrationFrame['aRatio_uncertainty'], fmt='k.', label='individual runs')
    plt.gca().axhline(y=np.mean(calibrationFrame['aRatio']), label='avg A Ratio')
    plt.legend(loc='best')
    plt.savefig(exportsPrefix+'/aRatioVsRun_energyCorrected.png');plt.close()

    plt.title('calibration run centroids'); plt.xlabel('run'); plt.ylabel('centroid (MHz)')#plt.ylabel('centroid (MHz) - %.2f THz'%(freqOffset/1E6))
    plt.errorbar(calibrationFrame.index, y=calibrationFrame['centroid'], yerr=calibrationFrame['cent_uncertainty'], fmt='k.', label='individual runs')
    plt.gca().axhline(y=np.mean(calibrationFrame['centroid']), label='avg centroid')
    plt.legend(loc='best')
    plt.savefig(exportsPrefix+'/centroidVsRun_energyCorrected.png');plt.close()
  return(calibrationFrame, calibrationVsScanNumber, calibrationVsScanTime)

def unstableFitRoutine(spec, fittingKwargs, aRatioSamples, prefix="iso0"):
  spec=copy.deepcopy(spec)
  fittingKwargs=copy.deepcopy(fittingKwargs)
  if "constrainAmplitudes" in fittingKwargs.keys():pass
  else: fittingKwargs["constrainAmplitudes"] = False
  massNumber=round(spec.mass)
  energyCorrectionToLoad=spec.energyCorrection
  fixed_Aratio=fittingKwargs['fixed_Aratio']
  spec.fitAndLogData(**fittingKwargs); popFrame=spec.populateFrame(prefix=prefix); v_iso=spec.resultParams[f"{prefix}_centroid"]-spec.frequencyOffset
  popFrame['avgEnergyCorrection']=energyCorrectionToLoad.eval(x=popFrame["avgScanTime"]);
  popFrame['avgEnergyCorrection_uncertainty']=energyCorrectionToLoad.eval_uncertainty(x=popFrame["avgScanTime"])
  if fixed_Aratio:
    aLowerSamples=[]; aLowerSampleErrs=[]
    aUpperSamples=[]; aUpperSampleErrs=[]
    centroidSamples=[]; centroidSampleErrs=[]
    for ratio in aRatioSamples:
      fittingKwargs['fixed_Aratio']=ratio; res,_=spec.fitDat(**fittingKwargs)
      aLowerSamples+=[res.params[f"{prefix}_Alower"].value]; aLowerSampleErrs+=[res.params[f"{prefix}_Alower"].stderr]
      aUpperSamples+=[res.params[f"{prefix}_Aupper"].value]; aUpperSampleErrs+=[res.params[f"{prefix}_Aupper"].stderr]
      centroidSamples+=[res.params[f"{prefix}_centroid"].value]; centroidSampleErrs+=[res.params[f"{prefix}_centroid"].stderr]
    fittingKwargs['fixed_Aratio']=fixed_Aratio
    plt.plot(aRatioSamples, centroidSamples,'.')
    plt.plot(fixed_Aratio, v_iso,'.')
    plt.title(f"Centroid variation from A ratio uncertainty for {massNumber}Al")
    plt.savefig(f'{spec.resultsPath}Extrapolating A_Ratio_Uncertainty.png'); plt.close()
    cent_unc_Arat = max([vv - v_iso for vv in centroidSamples], key=abs)
    # popFrame['cent_uncertainty'] = np.sqrt(popFrame['cent_uncertainty']**2+cent_unc_Arat**2)
    popFrame['aUpper_uncertainty']=np.sqrt(popFrame['aUpper_uncertainty']**2+(aUpperSamples[0]-aUpperSamples[-1])**2)
    popFrame['aLower_uncertainty']=np.sqrt(popFrame['aLower_uncertainty']**2+(aLowerSamples[0]-aLowerSamples[-1])**2)
    popFrame['centroid_sys_Arat']=cent_unc_Arat

  return(popFrame)

def fullAnalysis(a_ratio_fixed = True, equal_fwhm = False, cec_sim_toggle = False, peakModel='pseudoVoigt',whatToRun=False):
  massNumber=27
  mass=massDictionary[massNumber]
  cec_path = f'{massNumber}Al_CEC_peaks.csv' if cec_sim_toggle else False
  directoryPrefix='results/'+peakModel+'/equal_fwhm_'+str(equal_fwhm)+'/cec_sim_toggle_'+str(cec_sim_toggle!=False)
  if not os.path.exists(f'{directoryPrefix}/CalibrationDiagnostics'):  os.makedirs(f'{directoryPrefix}/CalibrationDiagnostics')
  wtr = whatToRun if whatToRun else WhatToRun() #determines whether to run spectrum construction and fits from scratch or not.
  fittingKwargs ={'cec_sim_data_path':cec_path,'equal_fwhm':equal_fwhm, 'peakModel':peakModel,'transitionLabel':'P12-S12'}

  '''Anti/colinear Analysis to determine v_0'''
  if wtr.fitAndLogToggle_BEA or not(os.path.exists(directoryPrefix+'/CalibrationDiagnostics/ReferenceCentroid.csv')):
    scanDirec='Anti_Colinear_Data'; logDirec=scanDirec; bea.updateLaserDic(logDirec)
    with open('laserDic.pkl','rb') as file: laserDic=pickle.load(file); file.close()
    for key in laserDic.keys(): laserDic[key]*=3E6 #conversion to MHz, and frequency tripled output

    fittingFunction = lambda x, y, yErr, mass=massDictionary[massNumber],\
        iList=iNucDictionary[massNumber], jGround=jGround, jExcited=jExcited, **kwargs:\
        ff.fitData(x, y, yErr, mass, iList, jGround, jExcited, **kwargs)

    targetDirectoryName='beamEnergy_analysis'
    spectrumKwargsBEA={'mass':mass,'jGround':jGround, 'jExcited':jExcited, 'nuclearSpinList':iNucDictionary[massNumber],
                    'directoryPrefix':directoryPrefix,'targetDirectory':targetDirectoryName, 'scanDirectory':scanDirec,
                    'windowToF':[489,543],'cuttingColumn':'time_step', 'constructSpectrum':False, 'fittingFunction':fittingFunction}
    anticolinearRuns = [16253,16254,16255,16263,16264,16265]
    colinearRuns     = [16258,16259,16260,16268,16269,16270]
    correctedCentroidEstimate,\
    compiledColinearParmResults,\
    compiledAnticolinearParmResults,\
    compiledColinearParmResults_Corrected,\
    compiledAnticolinearParmResults_Corrected = bea.getEnergyCorrectedResults(colinearRuns, anticolinearRuns, laserDic,
                                                                              spectrumKwargs=spectrumKwargsBEA, fittingKwargs=fittingKwargs,
                                                                              redoFits=False,redoFitWithEnergyCorrection=True)
    compiledColinearParmResults_Corrected.to_csv(f'{directoryPrefix}/CalibrationDiagnostics/BEA_ColinearFrame_Corrected.csv')
    compiledAnticolinearParmResults_Corrected.to_csv(f'{directoryPrefix}/CalibrationDiagnostics/BEA_AnticolinearFrame_Corrected.csv')
    print(correctedCentroidEstimate)
    v0 = correctedCentroidEstimate[0]; δv0 = np.sqrt(correctedCentroidEstimate[1]**2+correctedCentroidEstimate[2]**2)
    with open(directoryPrefix+'/CalibrationDiagnostics/ReferenceCentroid.csv','w') as file: file.write(f'{v0},{δv0}' );file.close()
  else:
    referenceCentroidData = np.loadtxt(directoryPrefix+'/CalibrationDiagnostics/ReferenceCentroid.csv', delimiter=',')
    v0 = referenceCentroidData[0]; δv0 = referenceCentroidData[1]
  print(f'v0={v0}+/-{δv0}')
  keV2amu=0.000001073544664258
  colinearity = False
  allIsotopesFrame = pd.DataFrame()
  '''Now for energy calibration runs'''
  massNumber=27
  fittingFunction = lambda x, y, yErr, mass=massDictionary[massNumber],\
      iList=iNucDictionary[massNumber], jGround=jGround, jExcited=jExcited, **kwargs:\
      ff.fitData(x, y, yErr, mass, iList, jGround, jExcited, **kwargs)
  spectrumKwargs={'mass':massDictionary[massNumber],'mass_uncertainty':mass_uncertaintyDictionary[massNumber], 'jGround':jGround, 'jExcited':jExcited, 'nuclearSpinList':iNucDictionary[massNumber],
                    'laserFrequency':3E6*laserDictionary[massNumber],'colinearity':colinearity, 'directoryPrefix':directoryPrefix,'scanDirectory':str(massNumber)+'Al/',
                    'timeOffset':scanTimeOffset,'windowToF':tofDictionary[massNumber], 'cuttingColumn':'ToF', 'constructSpectrum':wtr.exportSpectrumToggle_calibration, 'fittingFunction':fittingFunction}
  if wtr.fitAndLogToggle_calibration_bec or not(os.path.exists(f'{directoryPrefix}/CalibrationDiagnostics/calibrationFrame_beforeBEC.csv')):
    calibrationFrame_beforeBEC, calibrationVsScanNumber, calibrationVsScanTime = calibrationProcedure(runsDictionary[27],v0,δv0, spectrumKwargs=spectrumKwargs, fittingKwargs=fittingKwargs)
    spectrumKwargs['energyCorrection']=calibrationVsScanTime; spectrumKwargs['constructSpectrum']=wtr.exportSpectrumToggle_calibration_bec
    calibrationFrame, _, _ = calibrationProcedure(runsDictionary[27],v0,δv0, spectrumKwargs=spectrumKwargs, fittingKwargs=fittingKwargs)
    calibrationFrame_beforeBEC.to_csv(f'{directoryPrefix}/CalibrationDiagnostics/calibrationFrame_beforeBEC.csv')
    calibrationFrame.to_csv(f'{directoryPrefix}/CalibrationDiagnostics/calibrationFrame_afterBEC.csv')
  else:
    calibrationFrame_beforeBEC=pd.read_csv(f'{directoryPrefix}/CalibrationDiagnostics/calibrationFrame_beforeBEC.csv')
    calibrationScanTimes=np.array(calibrationFrame_beforeBEC['avgScanTime'])
    mass = massDictionary[massNumber]; laserFreq=3E6*laserDictionary[massNumber]
    calibrationVsScanNumber = bea.getCalibrationFunction(v0, δv0, calibrationFrame_beforeBEC, np.array(calibrationFrame_beforeBEC.index),mass, laserFreq)
    calibrationVsScanTime   = bea.getCalibrationFunction(v0, δv0, calibrationFrame_beforeBEC, calibrationScanTimes, mass, laserFreq)
    calibrationFrame=pd.read_csv(f'{directoryPrefix}/CalibrationDiagnostics/calibrationFrame_afterBEC.csv')
    print("you are here")

  aRatio,uncertainty_Aratio1, uncertainty_Aratio2 = bea.weightedStats(calibrationFrame['aRatio'],calibrationFrame['aRatio_uncertainty'])
  uncertainty_Aratio=(uncertainty_Aratio1**2+uncertainty_Aratio2**2)**0.5; print(aRatio, uncertainty_Aratio);
  if uncertainty_Aratio<0.01*aRatio: uncertainty_Aratio = 0.01*aRatio
  aRatioSamples=aRatio+np.linspace(-uncertainty_Aratio, uncertainty_Aratio,2)
  
  print(aRatio, uncertainty_Aratio)
  fixed_Aratio=aRatio if a_ratio_fixed else False
  print('calibration frame aRatio, before energy corrections:',np.mean(calibrationFrame_beforeBEC['aRatio']))
  print('calibration frame aRatio, after energy corrections:', np.mean(calibrationFrame['aRatio']));
  misalignmentError = randomBiasAnalysis(directoryPrefix)
  print('misalignmentError:',misalignmentError)


  def logEnergyCorrectionsVsRun(path, massNumber, runs, corrections):
    frame=pd.DataFrame({'runs':runs, 'correction':corrections})
    frame.to_csv(path+f"/calibrationVsRunNumberForMass{massNumber}.csv", index=False)


  print(calibrationFrame.keys());print(calibrationFrame)
  stable_aLower, unc1, unc2= bea.weightedStats(calibrationFrame['aLower'], calibrationFrame['aLower_uncertainty']); stable_aLower_uncertainty = np.sqrt(unc1**2+unc2**2)
  stable_aUpper, unc1, unc2= bea.weightedStats(calibrationFrame['aUpper'], calibrationFrame['aUpper_uncertainty']); stable_aUpper_uncertainty = np.sqrt(unc1**2+unc2**2)
  stable_aRatio=stable_aLower/stable_aUpper
  stable_aRatio_uncertainty = (stable_aRatio**2) *( (stable_aLower_uncertainty/stable_aLower)**2 + (stable_aUpper_uncertainty/stable_aUpper)**2)
  stable_centroid, stable_centroid_uncertainty = v0, δv0
  stable_uncorrectedCentroid, unc1, unc2= bea.weightedStats(calibrationFrame['uncorrectedCentroid'], calibrationFrame['uncorrectedCentroid_uncertainty']); stable_uncorrectedCentroid_uncertainty = np.sqrt(unc1**2+unc2**2)
  avgScanTime=np.mean(calibrationFrame['avgScanTime'])
  stableDict={"massNumber":massNumber,"I":[iNucDictionary[massNumber][0]],
              'mass':[massDictionary[massNumber]], "mass_uncertainty":[mass_uncertaintyDictionary[massNumber]],
              "aLower":[stable_aLower],"aLower_uncertainty":[stable_aLower_uncertainty],
              "aUpper":[stable_aUpper],"aUpper_uncertainty":[stable_aUpper_uncertainty],
              "aRatio":[stable_aRatio],"aRatio_uncertainty":[stable_aRatio_uncertainty],
              "centroid":[stable_centroid],"cent_uncertainty":[stable_centroid_uncertainty],
              "uncorrectedCentroid":[stable_uncorrectedCentroid],"uncorrectedCentroid_uncertainty":[stable_uncorrectedCentroid_uncertainty],
              "avgScanTime":avgScanTime, "avgEnergyCorrection_uncertainty": calibrationVsScanTime.params['slope'].value*3600,
              "centroid_sys_Arat":0}
  allIsotopesFrame = pd.concat([allIsotopesFrame, pd.DataFrame(stableDict)], ignore_index=True)
  '''now for all the other isotopes'''
  f_laser27=3E6*laserDictionary[27]
  beta27 = (v0**2-f_laser27**2)/(v0**2+f_laser27**2); gamma27 = 1/np.sqrt(1-beta27**2)
  factor27=beta27*gamma27*f_laser27
  print(factor27)
  exportsPrefix='./'+directoryPrefix+'/CalibrationDiagnostics/'
  directoryPrefix='results/'+peakModel+'/equal_fwhm_'+str(equal_fwhm)+'/cec_sim_toggle_'+str(cec_sim_toggle!=False)+'/fixed_Aratio_'+str(a_ratio_fixed)
  for massNumber in list(massDictionary.keys())[::-1][1:]:
    if wtr.fitAndLogToggleDic[massNumber]==False:continue
    spScaleable=False
    cec_path = f'{massNumber}Al_CEC_peaks.csv' if cec_sim_toggle else False
    runs = runsDictionary[massNumber]
    energyCorrectionToLoad_time = calibrationVsScanTime 
    energyCorrectionToLoad_runs = [float(calibrationVsScanNumber.eval(x=runNumber)) for runNumber in runs]
    logEnergyCorrectionsVsRun(exportsPrefix, massNumber, runs, energyCorrectionToLoad_runs)
    energyCorrectionToLoad=energyCorrectionToLoad_time
    mass=massDictionary[massNumber]
    targetDirectory = 'allScans_GroundMass/'
    targetDirectoryName=targetDirectory
    print('mass%d'%massNumber)
    if not os.path.exists(directoryPrefix): os.makedirs(directoryPrefix)

    if runsDictionary[massNumber]==[16446]:
      print('will try single nuclear species fit')
      iNucDictionary[massNumber] = [4]

    fittingFunction = lambda x, y, yErr, mass=massDictionary[massNumber],\
      iList=iNucDictionary[massNumber], jGround=jGround, jExcited=jExcited, **kwargs:\
      ff.fitData(x, y, yErr, mass, iList, jGround, jExcited, **kwargs)
    spectrumKwargs={'runs':runsDictionary[massNumber],'mass':massDictionary[massNumber],'mass_uncertainty':mass_uncertaintyDictionary[massNumber], 'jGround':jGround, 'jExcited':jExcited, 'nuclearSpinList':iNucDictionary[massNumber],
                    'laserFrequency':3E6*laserDictionary[massNumber],'colinearity':colinearity, 'directoryPrefix':directoryPrefix,'scanDirectory':str(massNumber)+'Al/', 'targetDirectory':'allScans_GroundMass/',
                    'timeOffset':scanTimeOffset,'windowToF':tofDictionary[massNumber], 'cuttingColumn':'ToF','keepLessIntegratedBins':False if massNumber==22 else True, 'fittingFunction':fittingFunction}
    fittingKwargs={'colinearity':colinearity, 'cec_sim_data_path':cec_path,'equal_fwhm':equal_fwhm, 'peakModel':peakModel,
                   'spScaleable':spScaleable, 'transitionLabel':'P12-S12','fixed_Aratio':fixed_Aratio}
     
    if massNumber==24:
      '''first analyze data transformed wrt ground state nuclear mass'''
      if len(runsDictionary[massNumber])==1: spectrumKwargs['targetDirectory'] =  f'{runsDictionary[massNumber][0]}_GroundMass/'
      spec=spc.Spectrum(constructSpectrum=wtr.exportSpectrumToggle, energyCorrection=energyCorrectionToLoad, **spectrumKwargs)         
      popFrame=unstableFitRoutine(spec, fittingKwargs, aRatioSamples, prefix="iso0")
      allIsotopesFrame = pd.concat([allIsotopesFrame, popFrame], ignore_index=True)
      '''and now for isomer'''
      if runsDictionary[massNumber]!=[16446]:
        spectrumKwargs['mass']=massDictionary[massNumber]+keV2amu*425.81
        spectrumKwargs['mass_uncertainty']=np.sqrt(mass_uncertaintyDictionary[massNumber]**2+(keV2amu*0.1)**2)  #isomer excited by 425.81 (10) keV
        spectrumKwargs['targetDirectory'] = 'allScans_IsomerMass/' if len(runsDictionary[massNumber])>1 else f'{runsDictionary[massNumber][0]}_IsomerMass/'
        spec=spc.Spectrum(constructSpectrum=wtr.exportSpectrumToggle, energyCorrection=energyCorrectionToLoad, **spectrumKwargs)           
        popFrame=unstableFitRoutine(spec, fittingKwargs, aRatioSamples, prefix="iso1")
        allIsotopesFrame = pd.concat([allIsotopesFrame, popFrame], ignore_index=True)

    else:                
      spec=spc.Spectrum(constructSpectrum=wtr.exportSpectrumToggle, energyCorrection=energyCorrectionToLoad, **spectrumKwargs)           
      popFrame=unstableFitRoutine(spec, fittingKwargs, aRatioSamples, prefix="iso0")
      allIsotopesFrame = pd.concat([allIsotopesFrame, popFrame], ignore_index=True)

  stableIndex = allIsotopesFrame.loc[allIsotopesFrame['massNumber']==27].index[0]
  if not fixed_Aratio: allIsotopesFrame['centroid_sys_Arat'] = 0
  for ii in allIsotopesFrame.index:
    massNumber = allIsotopesFrame.loc[ii,"massNumber"]
    allIsotopesFrame.loc[ii,"labLaser"] = 3E6*laserDictionary[massNumber]
  allIsotopesFrame['resVoltage'] = hf.freqToVoltage(*[*allIsotopesFrame[['mass','labLaser', 'centroid']].to_numpy().T])
  allIsotopesFrame['rel_beta'] = (allIsotopesFrame['centroid']**2 - allIsotopesFrame['labLaser']**2)/(allIsotopesFrame['centroid']**2 + allIsotopesFrame['labLaser']**2)
  # beta = (popFrame['centroid']**2-f_laser**2)/(popFrame['centroid']**2+f_laser**2)
  allIsotopesFrame['rel_gamma'] = 1/np.sqrt(1-allIsotopesFrame['rel_beta']**2)
  allIsotopesFrame['centroid_sys_align']=allIsotopesFrame['rel_gamma']*allIsotopesFrame['rel_beta']*allIsotopesFrame['labLaser']/factor27*misalignmentError
  allIsotopesFrame['centroid_sys_BEC']=np.abs(hf.voltageShiftToFrequencyShift(*[*allIsotopesFrame[['mass', 'labLaser', 'resVoltage', 'avgEnergyCorrection_uncertainty']].to_numpy().T]))
  allIsotopesFrame['centroid_sys_WM'] = wmResidualError
  allIsotopesFrame.loc[stableIndex,'centroid_sys_WM']=wmResidualError/np.sqrt(2)
  cent_sys_keys = [key for key in allIsotopesFrame.keys() if 'centroid_sys' in key]; shift_sys_keys=[]
  xData =  np.array(calibrationFrame['avgScanTime']).astype(float) ; yData = np.array(calibrationFrame['centroid']).astype(float)
  allIsotopesFrame['interpolatedReference']=np.interp(allIsotopesFrame['avgScanTime'],xData, yData)
  # allIsotopesFrame['shift']=allIsotopesFrame['centroid']-allIsotopesFrame['interpolatedReference']
  allIsotopesFrame['shift']=allIsotopesFrame['centroid']-allIsotopesFrame.loc[stableIndex]['centroid']
  allIsotopesFrame['shift_uncertainty_stat'] = np.sqrt(allIsotopesFrame['cent_uncertainty']**2+allIsotopesFrame.loc[stableIndex]['cent_uncertainty']**2)
  allIsotopesFrame.loc[stableIndex,'shift_uncertainty_stat']=0
  allIsotopesFrame.loc[stableIndex,'centroid_sys_fittingMethod']=0
  for key in cent_sys_keys:
    shift_key = 'shift'+key.lstrip('centroid'); shift_sys_keys+=[shift_key]
    allIsotopesFrame[shift_key]=np.sqrt(allIsotopesFrame[key]**2+allIsotopesFrame.loc[stableIndex,key]**2)
    allIsotopesFrame.loc[stableIndex,shift_key]=0
  allIsotopesFrame['shift_systematic_error']=np.sqrt(np.sum(allIsotopesFrame[shift_sys_keys]**2, axis=1))
  allIsotopesFrame['shift_uncertainty_exp']=np.sqrt(allIsotopesFrame['shift_uncertainty_stat']**2+allIsotopesFrame['shift_systematic_error']**2)
  print("wot")
  def extractChargeRadii(K, F,σK,σF, δν, stableIndex):
    # δν['shift']=δν['centroid']-δν['interpolatedReference']
    electronMass = sh.electronRestEnergy/sh.amu2eV
    for index in δν.index:
      mi=δν.loc[index]['mass']-13*electronMass; mRef=δν.loc[stableIndex]['mass']-13*electronMass
      δmi=δν.loc[index]['mass_uncertainty']; δmRef=δν.loc[stableIndex]['mass_uncertainty']
      massFactor=1/mi-1/mRef
      δν.loc[index,'δrsq']=(δν.loc[index]['shift']-K*massFactor)/F
      uncert1=(1/F**2) * δν.loc[index]['shift_uncertainty_stat']**2 
      uncert2=(1/F**2) * δν.loc[index]['shift_systematic_error']**2
      uncert3=((massFactor/F)**2) * σK**2
      uncert4=(((δν.loc[index]['shift']-K*massFactor)/F**2)**2) * σF**2
      uncert5=((K/F)**2)*((δmi/(mi**2))**2 + (δmRef/(mRef**2))**2)
      if index==stableIndex:
        δν.loc[index,'δrsq_uncertainty_stat']=0
        δν.loc[index,'δrsq_systematic_error']=0
        δν.loc[index,'δrsq_uncertainty_mass_factor']=0
        δν.loc[index,'δrsq_uncertainty_field_factor']=0
        δν.loc[index,'δrsq_uncertainty_mass_measure']=0
        δν.loc[index,'δrsq_uncertainty_exp']=0
        δν.loc[index,'δrsq_uncertainty_theory']=0
        δν.loc[index,'δrsq_uncertainty_total']=0
        δν.loc[index,'abs_charge_rad']=absChargeRad27
        δν.loc[index,'abs_charge_rad_uncertainty']=absChargeRad27_uncertainty
      else:
        δν.loc[index,'δrsq_uncertainty_stat']=np.sqrt(uncert1)
        δν.loc[index,'δrsq_systematic_error']=np.sqrt(uncert2)
        δν.loc[index,'δrsq_uncertainty_mass_factor']=np.sqrt(uncert3)
        δν.loc[index,'δrsq_uncertainty_field_factor']=np.sqrt(uncert4)
        δν.loc[index,'δrsq_uncertainty_mass_measure']=np.sqrt(uncert5)
        δν.loc[index,'δrsq_uncertainty_exp']=np.sqrt(uncert1+uncert2)
        δν.loc[index,'δrsq_uncertainty_theory']=np.sqrt(uncert3+uncert4+uncert5)
        δν.loc[index,'δrsq_uncertainty_total']=np.sqrt(uncert1+uncert2+uncert3+uncert4+uncert5)
        δν.loc[index,'abs_charge_rad']=np.sqrt(absChargeRad27**2+δν.loc[index,'δrsq'])
        δν.loc[index,'abs_charge_rad_uncertainty']=(1/δν.loc[index,'abs_charge_rad'])*np.sqrt((absChargeRad27*absChargeRad27_uncertainty)**2+(1/4)*(δν.loc[index,'δrsq_uncertainty_total']**2))
    return()

  kα_Skrip=-0.7*1000; σK_Skrip=2.1*1000 #Total MassShift sensitivity in MHz*amu
  Fα_Skrip=70.11;     σF_Skrip=0.13 #FieldShift sensitivity in MHz/fm^2
  extractChargeRadii(kα_Skrip,Fα_Skrip,σK_Skrip,σF_Skrip, allIsotopesFrame, stableIndex)
  
  I27=allIsotopesFrame.loc[stableIndex]['I'];
  aLow27=allIsotopesFrame.loc[stableIndex]['aLower']; δaLow27=allIsotopesFrame.loc[stableIndex]['aLower_uncertainty'];
  μ27Al= 3.64070#(2) #Οther value?: 3.6415069#(7)
  μ27Al_uncertainty= 0.00002
  μCommonFactor = μ27Al/(I27*allIsotopesFrame.loc[stableIndex]['aLower'])
  μCommonFactor_uncertainty = np.sqrt((μ27Al_uncertainty/aLow27)**2
                                     +( (μ27Al*δaLow27)/(aLow27**2) )**2 )/I27
  allIsotopesFrame['μ']=allIsotopesFrame['aLower']*allIsotopesFrame['I']*μCommonFactor
  allIsotopesFrame['μ_uncertainty']=allIsotopesFrame['I']*np.sqrt((allIsotopesFrame['aLower_uncertainty']*μCommonFactor)**2 +
                                                                  (allIsotopesFrame['aLower']*μCommonFactor_uncertainty)**2)

  nameTag=peakModel+'-eqFWHM_'+str(equal_fwhm)+'-CECsim'+str(cec_sim_toggle!=False)+'-fix_Arat_'+str(a_ratio_fixed)+'_'
  allIsotopesFrame.to_csv(directoryPrefix+'/'+nameTag+'CompiledAnalysisResults.csv')#, header='#cec_sim: '+str(cec_sim_toggle)+'; equal_fwhm: '+str(equal_fwhm)+'; fixed_Aratio:'+str(fixed_Aratio))
  return(allIsotopesFrame)

if __name__ == '__main__':
  #If you don't already have the pre-processed data files, this step will be incredibly slow, but only once 
  dm.processMDA_Directory('Anti_Colinear_Data')
  for massNum in [27,25,24,23,22]: dm.processMDA_Directory(f'{massNum}Al')
  
  wtr=WhatToRun()
  wtr.fitAndLogToggle_BEA =                True;#False;#
  wtr.exportSpectrumToggle_calibration =   True;#False;#
  wtr.fitAndLogToggle_calibration =        True;#False;#
  wtr.exportSpectrumToggle_calibration_bec=True;#False;#
  wtr.fitAndLogToggle_calibration_bec=     True;#False;#
  wtr.exportSpectrumToggle     =           True;#False;#
  wtr.exportSpectrumToggle_bec =           True;#False;#
  wtr.fitAndLogToggleDic={  22:            True,#False,#
                            23:            True,#False,#
                            24:            True,#False,#
                            25:            True,#False,#
                            27:            True}#False}#
  peak_model_list=['Voigt']#,'pseudoVoigt']
  equal_fwhm_toggle_list = [False]# ,True]
  cec_sim_toggle_list = [False]# ,True]
  a_ratio_fixed_list  = [True]#, False]
  i=0
  allFramesDic={}
  steps_factor=len(a_ratio_fixed_list)*len(cec_sim_toggle_list)
  total_steps = 0;
  if 'Voigt' in peak_model_list: total_steps+=steps_factor
  if 'pseudoVoigt' in peak_model_list: total_steps+=len(equal_fwhm_toggle_list)*steps_factor
  for peakModel in peak_model_list:
    for equal_fwhm_toggle in equal_fwhm_toggle_list:
      if equal_fwhm_toggle==True and peakModel=='Voigt':continue
      for cec_sim_toggle in cec_sim_toggle_list:
        print(equal_fwhm_toggle,cec_sim_toggle)
        for a_ratio_toggle in a_ratio_fixed_list:
          allIsotopesFrame=fullAnalysis(a_ratio_fixed = a_ratio_toggle, equal_fwhm = equal_fwhm_toggle, cec_sim_toggle = cec_sim_toggle, peakModel=peakModel, whatToRun=wtr)
          allFramesDic[('fwhm_'+str(equal_fwhm_toggle),'cec_'+str(cec_sim_toggle), 'aRatio_'+str(a_ratio_toggle))]=allIsotopesFrame
          i+=1
          print('i=',i)
  print(allIsotopesFrame.keys())
  print(allIsotopesFrame[['massNumber', 'I', 'shift','shift_uncertainty_exp', 'δrsq',
                          'δrsq_uncertainty_exp','abs_charge_rad', 'abs_charge_rad_uncertainty']])