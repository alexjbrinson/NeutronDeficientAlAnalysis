import pandas as pd
import numpy as np
import RAP.SpectrumHandler as sh
import RAP.HelperFunctions as hf


kα_Skrip=-0.7*1000; σK_Skrip=2.1*1000 #Total MassShift sensitivity in MHz*amu
Fα_Skrip=70.11;     σF_Skrip=0.13 #FieldShift sensitivity in MHz/fm^2

def extractChargeRadii(K, F,σK,σF, δν, referenceIndex, correlation = 0):
  δν['shift']=δν['centroid']-δν.loc[referenceIndex]['centroid']
  # δν['shift']=δν['centroid']-δν['interpolatedReference']
  δν['shift_uncertainty_stat'] = np.sqrt(δν['cent_uncertainty']**2+δν.loc[referenceIndex]['cent_uncertainty']**2
                                        -2*correlation*δν['cent_uncertainty']*δν.loc[referenceIndex]['cent_uncertainty']) #fit correlation between 24 and 24m (~0.22) might actually reduce this
  δν.loc[referenceIndex,'shift_uncertainty_stat']=0
  electronMass = sh.electronRestEnergy/sh.amu2eV
  for index in δν.index:
    mi=δν.loc[index]['mass']-13*electronMass; mRef=δν.loc[referenceIndex]['mass']-13*electronMass
    δmi=δν.loc[index]['mass_uncertainty']; δmRef=δν.loc[referenceIndex]['mass_uncertainty']
    massFactor=1/mi-1/mRef
    δν.loc[index,'δrsq']=(δν.loc[index]['shift']-K*massFactor)/F
    uncert1=(1/F**2) * δν.loc[index]['shift_uncertainty_stat']**2 
    uncert2=(1/F**2) * δν.loc[index]['shift_systematic_error']**2
    uncert3=((massFactor/F)**2) * σK**2
    uncert4=(((δν.loc[index]['shift']-K*massFactor)/F**2)**2) * σF**2
    uncert5=((K/F)**2)*((δmi/(mi**2))**2 + (δmRef/(mRef**2))**2)
    if index==referenceIndex:
      δν.loc[index,'δrsq_uncertainty_stat']=0
      δν.loc[index,'δrsq_systematic_error']=0
      δν.loc[index,'δrsq_uncertainty_mass_factor']=0
      δν.loc[index,'δrsq_uncertainty_field_factor']=0
      δν.loc[index,'δrsq_uncertainty_mass_measure']=0
      δν.loc[index,'δrsq_uncertainty_exp']=0
      δν.loc[index,'δrsq_uncertainty_theory']=0
      δν.loc[index,'δrsq_uncertainty_total']=0
      # δν.loc[index,'abs_charge_rad']=absChargeRad27
      # δν.loc[index,'abs_charge_rad_uncertainty']=absChargeRad27_uncertainty
    else:
      δν.loc[index,'δrsq_uncertainty_stat']=np.sqrt(uncert1)
      δν.loc[index,'δrsq_systematic_error']=np.sqrt(uncert2)
      δν.loc[index,'δrsq_uncertainty_mass_factor']=np.sqrt(uncert3)
      δν.loc[index,'δrsq_uncertainty_field_factor']=np.sqrt(uncert4)
      δν.loc[index,'δrsq_uncertainty_mass_measure']=np.sqrt(uncert5)
      δν.loc[index,'δrsq_uncertainty_exp']=np.sqrt(uncert1+uncert2)
      δν.loc[index,'δrsq_uncertainty_theory']=np.sqrt(uncert3+uncert4+uncert5)
      δν.loc[index,'δrsq_uncertainty_total']=np.sqrt(uncert1+uncert2+uncert3+uncert4+uncert5)
      # δν.loc[index,'abs_charge_rad']=np.sqrt(absChargeRad27**2+δν.loc[index,'δrsq'])
      # δν.loc[index,'abs_charge_rad_uncertainty']=(1/δν.loc[index,'abs_charge_rad'])*np.sqrt((absChargeRad27*absChargeRad27_uncertainty)**2+(1/4)*(δν.loc[index,'δrsq_uncertainty_total']**2))
  return()

if __name__ == '__main__':
  pd.set_option('display.max_columns', None)
  laserFreq=3E6*376.017863
  peak_model_list=['Voigt']#,'pseudoVoigt']
  equal_fwhm_toggle_list = [False]#[True, False]#
  cec_sim_toggle_list = [False]
  a_ratio_fixed_list = [True]#,False]

  for peakModel in peak_model_list:
    for equal_fwhm_toggle in equal_fwhm_toggle_list:
      if equal_fwhm_toggle==True and peakModel=='Voigt':continue
      for cec_sim_toggle in cec_sim_toggle_list:
        partial_path = f'results/{peakModel}/equal_fwhm_{equal_fwhm_toggle}/'\
                       f'cec_sim_toggle_{cec_sim_toggle}'
        print(equal_fwhm_toggle,cec_sim_toggle)
        calibrationData = np.loadtxt(f'{partial_path}/CalibrationDiagnostics/calibrationvsRunNumberForMass24.csv', delimiter=',', skiprows=1)[:,1]
        print(calibrationData)
        ΔE = np.mean(calibrationData)
        δΔE = np.std(calibrationData)/np.sqrt(len(calibrationData));
        print(f'ΔE={ΔE}+/-{δΔE}')

        for a_ratio_toggle in a_ratio_fixed_list:
          path =f'{partial_path}/fixed_Aratio_{a_ratio_toggle}'
          fname=f'{peakModel}-eqFWHM_{equal_fwhm_toggle}-CECsim{cec_sim_toggle}-'\
                f'fix_Arat_{a_ratio_toggle}_CompiledAnalysisResults.csv'
          allIsotopesFrame=pd.read_csv(f'{path}/{fname}')
          al24Frame = allIsotopesFrame.copy()
          al24Frame=al24Frame[al24Frame["massNumber"]==24]
          print(al24Frame.keys())
          keys=['mass','I','shift','centroid', 'cent_uncertainty', 'centroid_sys_Arat','centroid_sys_align', 'centroid_sys_BEC','shift',
                'shift_uncertainty_stat', 'shift_sys_Arat', 'shift_sys_align','shift_sys_BEC', 'shift_sys_WM', 'shift_systematic_error',
                'centroid_sys_WM','δrsq', 'δrsq_uncertainty_exp', 'δrsq_uncertainty_theory', 'δrsq_uncertainty_total']
          # hf.freqToVoltage
          print('ref = 27:')
          print(al24Frame[keys])

          referenceIndex = al24Frame.loc[al24Frame['I']==4].index[0]
          isomerIndex = al24Frame.loc[al24Frame['I']==1].index[0]
          v0 = al24Frame.loc[referenceIndex]['centroid']; v1  = al24Frame.loc[isomerIndex]['centroid']
          m0 = al24Frame.loc[referenceIndex]['mass']    ; m1 = al24Frame.loc[isomerIndex]['mass']
          cent_sys_keys = [key for key in al24Frame.keys() if 'centroid_sys' in key]; shift_sys_keys=[]
          al24Frame['shift']=al24Frame['centroid']-al24Frame.loc[referenceIndex]['centroid']
          al24Frame['shift_uncertainty_stat'] = (al24Frame['cent_uncertainty']**2+al24Frame.loc[referenceIndex]['cent_uncertainty']**2)
          al24Frame.loc[referenceIndex,'shift_uncertainty_stat']=0
          for key in cent_sys_keys:
            shift_key = 'shift'+key.lstrip('centroid'); shift_sys_keys+=[shift_key]
            al24Frame[shift_key]=np.abs(al24Frame[key]-al24Frame.loc[referenceIndex,key])
            al24Frame.loc[referenceIndex,shift_key]=0
          al24Frame['shift_systematic_error']=np.sqrt(np.sum(al24Frame[shift_sys_keys]**2, axis=1))
          extractChargeRadii(kα_Skrip,Fα_Skrip,σK_Skrip,σF_Skrip, al24Frame, referenceIndex, correlation=0.22)
          print('\nref = 24:')
          
          keys=['mass','I','centroid','cent_uncertainty', 'shift','shift_uncertainty_stat']+cent_sys_keys+shift_sys_keys+\
               ['shift_systematic_error','δrsq', 'δrsq_uncertainty_stat', 'δrsq_systematic_error','δrsq_uncertainty_mass_measure',
                'δrsq_uncertainty_mass_factor','δrsq_uncertainty_field_factor','δrsq_uncertainty_exp','δrsq_uncertainty_theory', 'δrsq_uncertainty_total']
          print(al24Frame[keys])
          al24Frame[keys].to_csv(f'{path}/isomerShift.csv')