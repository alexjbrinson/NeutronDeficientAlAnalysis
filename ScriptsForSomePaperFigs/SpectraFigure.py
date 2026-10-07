import numpy as np
import RAP.FittingFunctions as ff
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# import SpectrumClass as spc
scanTimeOffset=1716156655
runsDictionary = {
  22:[16463,16464,16465,16478,16479,16480,16481,16482,16483,16484,16485,16486,16487,16488,16489,16490,16491,16497,16498,16499,16500,16501,16502,16503,16504,16505],#16464
  23:[16405,16406,16407,16408,16414,16415,16416,16418,16419,16420,16421], #16404
  24:[16434,16435,16436,16437,16438,16445,16446,16447,16448,16449,16450], #16445,16446 have different buncher settings
  25:[16384,16385,16386,16387,16388],
  27:[16368,16369,16370,16389,16391,16392,16395,16396,16397,16410,16412,16413,16422,16424,16425,16426,#16367, 16371,16372,16373,16374,16375,16376 are all trash #16428 not good #16366 a little weird
      16429,16430,16439,16441,16442,16451,16458,16459,16470,16473,16474,16477,16492,16494,16495,16508,16510,16512]}

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

laserDictionary = {
  22:375.990796,
  23:376.004732,
  24:376.017863,
  25:376.030178,
  27:376.052850}

axesLabelFontSize=18
tickSize=16
labelfontSize=12
colorList=["orange","lime","cyan","magenta"]

colorDic={'22' :'#ffa600',
          '23' :'#dd5182',
          '24' :'#955196',
          '24m':'#ff6e54',
          '25' :'#444e86',
          '27' :'#000000'}#'#003f5c'}

pd.set_option('display.max_columns', None)

absChargeRad27 = 3.0610#fm
absChargeRad27_uncertainty = 0.0031#fm

def pullCalibrationScan(scan, peakModel='Voigt', equal_fwhm=False, cec_sim_toggle=False, energyCorrection=False):
  subPath=''
  mass=27
  directoryPrefix='../results/'+str(peakModel)+'/equal_fwhm_'+str(equal_fwhm)+'/cec_sim_toggle_'+str(cec_sim_toggle!=False)
  targetDirectoryName='Scan'+str(scan)
  runPath = './'+directoryPrefix+'/mass'+str(round(mass))+'/'+str(targetDirectoryName)+'/'+subPath #this is where I saved all outputs of the earlier function
  if energyCorrection==False:
    resultsPath = runPath+'fit_result.pkl'
    statsPath   = runPath+'fit_statistics.pkl'
  else:
    runPath += '/energyCorrected/'
    resultsPath = runPath+'fit_result_energyCorrected.pkl' #this is where I'll save all outputs of this function
    statsPath   = runPath+'fit_statistics_energyCorrected.pkl'
  with open(resultsPath,'rb') as file:
    paramResults=pickle.load(file)
  with open(statsPath,'rb') as file:
    statsResults=pickle.load(file)
  tempDict={}
  for parm in paramResults.keys(): tempDict[parm+'_val'] = paramResults[parm].value; tempDict[parm+'_unc'] = paramResults[parm].stderr
  tempDict['redchi'] = statsResults['redchi']
  return(tempDict)

massNumber=27
mass=massDictionary[massNumber]
laserFreq=3E6*laserDictionary[massNumber]
iNucList=iNucDictionary[massNumber]
jGround=0.5; jExcited=0.5;
equal_fwhm=False
peakModel='Voigt'

# fig2, axs = plt.subplots(nrows=5, figsize=(8,8), sharex=True, gridspec_kw={'hspace': 0})
# fig3 = plt.figure(figsize=(8,2))
fig = plt.figure(figsize=(8,10), constrained_layout=True)
gs = fig.add_gridspec(
  nrows=7,
  ncols=1,
  height_ratios=[1,1,1,1,1,.2,0.65],
  hspace=0.03
)
axs = [fig.add_subplot(gs[i,0]) for i in range(5)]
ax_shift = fig.add_subplot(gs[6,0])

pathPrefix = "../results/Voigt/equal_fwhm_False/cec_sim_toggle_False/"
resultsFrame=pd.read_csv(pathPrefix+"fixed_Aratio_True/Voigt-eqFWHM_False-CECsimFalse-fix_Arat_True_CompiledAnalysisResults.csv")
pd.set_option('display.max_columns', None)
print(resultsFrame.keys())
print(resultsFrame[['mass','shift','shift_uncertainty_stat']])
# quit()
print(resultsFrame[['δrsq_uncertainty_exp','δrsq_uncertainty_theory','δrsq_uncertainty_total']])
# quit()
resultsFrame['key']=['27','25','24','24m','23','22']
print(resultsFrame[['key','shift','δrsq']][-1:0:-1])


reference=resultsFrame['centroid'][0]
runsList=runsDictionary[27]
redchiList=np.array([pullCalibrationScan(scan, peakModel=peakModel, equal_fwhm=equal_fwhm, cec_sim_toggle=False, energyCorrection=False)['redchi'] for scan in runsList])
bestScan=runsList[np.argmin(redchiList)]
print(bestScan)

ax_shift.tick_params(axis='both', direction='in')
ax_shift.set_xlabel('IsotopeShift (MHz)', fontsize=axesLabelFontSize)
ax_shift.set_ylabel('A', fontsize=axesLabelFontSize)

plt.xticks(fontsize=tickSize); ax_shift.set_xticks(ticks=[4*i for i in range(-1,4)], labels=[4*i for i in range(-1,4)], fontsize=tickSize)
ax_shift.set_yticks(ticks=[i for i in range(6)], labels=[22+i for i in range(6)], fontsize=tickSize)
# Get fig 3 y-axis tick labels, hide every other
for i, label in enumerate(ax_shift.get_yticklabels()):
    if i % 2 == 0: label.set_visible(False)

# plt.plot(0, 5, '.', color=colorDic['27'])
for ii,index in enumerate(resultsFrame.index[::-1]):
  print('index:', index)
  y = resultsFrame['mass'][index]-22
  # if index==3: y-=0.8
  # elif ii<5: y-=1
  key=str(resultsFrame['massNumber'][index])
  if key=="24" and resultsFrame["I"][index]==1: key='24m'
  print('key:',key)
  ax_shift.plot(resultsFrame['shift'][index], y, '.', color=colorDic[key],)
  expError = np.sqrt(resultsFrame['shift_uncertainty_stat'][index]**2+resultsFrame['shift_systematic_error'][index]**2)
  ax_shift.errorbar(x=resultsFrame['shift'][index], xerr=expError,
                    y=y, label=r'${}^{%s}$Al'%key, alpha=0.5, color=colorDic[key])
# fig3.legend()
# plt.tight_layout()

for ii,mass in enumerate([27,25,24,23,22]):
  axIndex=ii
  print(f'mass: {mass}')
  if mass==27: path=f'{pathPrefix}mass27/Scan{bestScan}/energyCorrected/'
  else: path=f'{pathPrefix}fixed_Aratio_True/mass{mass}/allScans_GroundMass/energyCorrected/'
  # with open(f'{path}fitEvaluator/bgParams.pkl','rb') as file: bgParms=pickle.load(file)
  with open(f'{path}fitEvaluator/iso0_evalParms.pkl','rb') as file: evalParms=pickle.load(file)
  with open(f'{path}fitEvaluator/evalKwargs.pkl','rb') as file: evalKwargs=pickle.load(file)
  evalParms['spProp']=0; mainPeakInterp = lambda x: ff.hyperFinePredictionFreeAmps_pseudoVoigt(x, **evalKwargs, **evalParms)
  interpolator=lambda x: ff.hyperfinePredictorGREAT

  with open(path+'fit_params_energyCorrected.pkl','rb') as file: fitParams = pickle.load(file)
  spShift=fitParams['iso0_spShift'].value
  data=np.loadtxt(path+'spectralData_energyCorrected.csv', dtype=float, delimiter=',',skiprows=1)
  xData= data[:,2]-reference ; yData= data[:,3] ; yUncertainty=data[:,4]; yMax=np.max(yData)
  
  interp_data=np.loadtxt(f'{path}_bg_energyCorrected.csv', delimiter=','); x_interp=interp_data[:,0]-np.min(interp_data[:,0])+xData[0]; bg_interp=interp_data[:,1]
  iso0_interp=np.loadtxt(f'{path}_iso0_energyCorrected.csv', delimiter=',')[:,1]; y_interp = bg_interp+iso0_interp
  cent0=resultsFrame[resultsFrame['key']==str(mass)]['centroid'].iloc[0]-reference# + spShift
  axs[axIndex].axvline(cent0, linestyle='--', color=colorDic[str(mass)]); print(f'{mass}: {cent0}')
  
  if mass==24:
    axs[axIndex].text(.99,.8,r'--$\,^{%d}$Al'%mass, transform=axs[axIndex].transAxes, color=colorDic[str(mass)], fontsize=tickSize, horizontalalignment='right')
    axs[axIndex].text(.99,.6,r'$\cdots\,^{24m}$Al', transform=axs[axIndex].transAxes, color=colorDic['24m'], fontsize=tickSize, horizontalalignment='right')
    axs[axIndex].plot(x_interp, iso0_interp, color=colorDic[str(mass)], linestyle='dashed')
    iso1_interp=np.loadtxt(f'{path}_iso1_energyCorrected.csv', delimiter=',')[:,1]; y_interp += iso1_interp
    axs[axIndex].plot(x_interp, iso1_interp, color=colorDic[str(mass)+'m'],linestyle=(0,(1,1))) 
    cent1=resultsFrame[resultsFrame['key']=='24m']['centroid'].iloc[0]-reference
    axs[axIndex].axvline(cent1, linestyle='--', color=colorDic['24m']); print(f'{mass}m: {cent1}')
    color0=np.array([int(colorDic['24'].lstrip('#')[i:i+2],16) for i in (0,2,4)])
    color1=np.array([int(colorDic['24m'].lstrip('#')[i:i+2],16) for i in (0,2,4)])
    color_interp=np.array([iso0_interp/(iso0_interp+iso1_interp)*color0[i] + iso1_interp/(iso0_interp+iso1_interp)*color1[i] for i in range(3)]).T
    scatterColors=np.array([np.interp(xData, x_interp, color_interp[:,i]) for i in range(3)]).T
    for jj in range(len(x_interp)-1):
      axs[axIndex].plot(x_interp[jj:jj+2], y_interp[jj:jj+2], color=color_interp[jj]/256, alpha=0.5)
    for jj in range(len(xData)):
      axs[axIndex].errorbar(xData[jj],yData[jj],yerr=yUncertainty[jj], fmt='.', color=scatterColors[jj]/256)
  else:
    axs[axIndex].text(.99,.8,r'$-\,^{%d}$Al'%mass, transform=axs[axIndex].transAxes, color=colorDic[str(mass)], fontsize=tickSize, horizontalalignment='right')
    axs[axIndex].plot(x_interp, y_interp, color=colorDic[str(mass)], alpha=0.5)
    axs[axIndex].errorbar(xData,yData,yerr=yUncertainty, fmt='.', color=colorDic[str(mass)])
  if mass==27: 
    yMax27=yMax
    # axs[axIndex].plot(x_interp, mainPeakInterp(interp_data[:,0]), color=colorDic[str(mass)], alpha=0.5)
  yData*=(yMax27/yMax); yUncertainty*=(yMax27/yMax)
  axs[axIndex].tick_params(axis='both', direction='in', labelbottom=False, labelleft=True, labelsize=tickSize)
for ax in axs:
  ax.set_xlim([-1200,1475])


axs[0].tick_params(axis='both', direction='in', top=True, labeltop=True, bottom=True)
plt.xticks(fontsize=tickSize); axs[0].set_xticks(ticks=[500*i for i in range(-2,3)], labels=[500*i for i in range(-2,3)], fontsize=tickSize)
axs[0].set_xlabel(r'Frequency - $\nu_0^{27}$ (MHz)', fontsize=axesLabelFontSize)
axs[0].xaxis.set_label_position('top')
axs[2].set_ylabel('Count rate (ions/s)', fontsize=axesLabelFontSize)

kwargs = dict(
    transform=fig.transFigure,
    color='black',
    clip_on=False,
    linewidth=1,
    linestyle='--'
)

fig.lines.extend([
    plt.Line2D((0.49, 0.085), (0.193, .152), **kwargs),
    plt.Line2D((0.5, .996), (0.193, 0.152), **kwargs),
])

# plt.show()
# plt.close()
# savePath='../../ChargeRadPaper/Figures/'
plt.savefig("spectrumFig_FullExpError.pdf")