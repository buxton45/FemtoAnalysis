#ifndef COMPAREFITTINGMETHODSWSYSERRS_H_
#define COMPAREFITTINGMETHODSWSYSERRS_H_


#include "FitValuesWriterwSysErrs.h"
#include "FitValuesLatexTableHelperWriter.h"
#include "CompareFittingMethods.cxx"


  extern TCanvas* CompareImF0vsReF0(vector<FitValWriterInfo> &aFitValWriterInfo, TString aSystematicsFileLocation_LamKch, TString aSystematicsFileLocation_LamK0, bool aDrawPredictions=false, TString aCanNameMod="", bool aSuppressDescs=false, bool aSuppressAnStamps=false, bool aDrawStatOnly=false);
  extern TCanvas* CompareLambdavsRadius(vector<FitValWriterInfo> &aFitValWriterInfo, TString aSystematicsFileLocation_LamKch, TString aSystematicsFileLocation_LamK0, CentralityType aCentType, TString aCanNameMod="", bool aSuppressDescs=false, bool aSuppressAnStamps=false, bool aDrawStatOnly=false);
  extern TCanvas* CompareAll(vector<FitValWriterInfo> &aFitValWriterInfo, TString aSystematicsFileLocation_LamKch, TString aSystematicsFileLocation_LamK0, bool aDrawPredictions=false, TString aCanNameMod="", bool aDrawStatOnly=false);
  extern TCanvas* CompareAll2Panel(vector<FitValWriterInfo> &aFitValWriterInfo, TString aSystematicsFileLocation_LamKch, TString aSystematicsFileLocation_LamK0, bool aDrawPredictions=false, TString aCanNameMod="", bool aDrawStatOnly=false);



#endif
