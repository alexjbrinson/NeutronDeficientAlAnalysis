# RISE Neutron Deficient Al Repo
This is a snapshot of the analysis code used for the RISE paper "Nuclear charge radii of aluminum isotopes at the proton drip line". The raw data is too large to include on github, but will be uploaded to Zenodo. All necessary steps for reproducing the analysis yourself are given below. For this analysis, I used **64-bit python 3.13 on Windows**, with the following dependency list:
|Package|        Version I used|
|-|-|
|numpy|         2.2.5 |
|sympy|         1.13.1|
|scipy|         1.15.2|
|pandas|        2.2.3 |
|polars|        1.27.1|
|matplotlib|    3.10.1|
|lmfit|         1.3.3 |
|numba|         0.61.2|
|openpyxl|      3.1.5 |
|pyarrow|       19.0.1|

Any missing dependencies should automatically install when you pip install the RISE Analysis Package in step 2.
## Reproducing the analysis:

1. **mda2ascii**
This step is not necessary if you elect to start from the provided "scanXXXXXDataFrame.csv" files, which have already converted the mda into individual time-stamped ascii files, and then concatenated all of those together. But if for whatever reason you wish to process the raw mda files yourself, the first thing you need is the mda2ascii program. Downloads for these used to be hosted at ANL.gov, but the webpage seems to have been taken down. MDA_Utilities can still be built locally using the github repo https://github.com/BCDA-APS/MDA_Utilities. I have also included the Windows executable in the "RISEAnalysisPackage" directory. This program should be added to your path.

2. **RISE Analysis Package (RAP)**
The RISE Analysis Package is a (somewhat) abstracted library for analyzing data produced by BECOLA/RISE. The most current version can be found at https://github.com/alexjbrinson/RISEAnalysisPackage, but a snapshot of the package has been embedded in this repo, for posterity. You can install the package with:
    >\>pip install path/to/repo/RISEAnalysisPackage

    Note that the namespace of the package is RAP, e.g. 
    ``` python
     import RAP.HelperFunctions as hf
    ```
3. **Analysis Script**
On the top level of this directory, you should see `onlineAnalysis.py`. This script will take the raw .MDA files, run the ASCII conversion(if you don't already have the pre-processed scan DataFrame.csv's), construct and export spectra, and fit them. It will also run the anti/collinear analysis to determine $\nu_0$, find the best-fit energy correction calibration function, and then redo the spectrum construction and fitting on the energy-corrected calibration spectra. All of these resulting spectra and fit results will be saved in a subdirectory called "results/".
Note that the initial ascii conversion and dataframe construction is quite slow. The good new is those dataframes only need to be constructed once, so these steps will be skipped over if you ever wish to re-run this analysis script.

4. **Generating Figure Plots**
In the directory "ScriptsForSomePaperFigs/", you will find additional scripts which reproduce figures 1, 3, and supplement 6 from the paper.
The data used to produce fig 2 and Table 1 can be found in the "results/" subdirectory. If you follow the path down all of the selected fit settings, you will see a dataframe titled "Voigt-eqFWHM_False-CECsimFalse-fix_Arat_True_CompiledAnalysisResults.csv", which compiles best fit results from all isotopes, and extracts the isotope shift and associated charge radii.

Note: The scripts in steps 3 and 4 make use of relative paths to import data. They have been written with the expectation that you will run them from the respective directories in which they are saved, e.g.
> path/to/repo>python onlineAnalysis\.py
> path/to/repo/ScriptsForSomePaperFigs>python spectraFigure\.py
