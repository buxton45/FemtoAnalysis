/* FitValuesWriter.h */
/* NOTE: This is designed for the most typical case of a pair with its conjugate
         across all 3 centrality bins, for a total of 6 FitPairAnalysis */

#ifndef FITVALUESWRITER_H
#define FITVALUESWRITER_H

//includes and any constant variable declarations
#include <iostream>
#include <fstream>
#include <iomanip>
#include <complex>
#include <math.h>
#include <vector>
#include <assert.h>

#include "TSystem.h"
#include "TObjString.h"
#include "TObjArray.h"
#include "TGraphAsymmErrors.h"
#include "TPad.h"

using std::cout;
using std::endl;
using std::vector;

#include "Types.h"
#include "Types_LambdaValues.h"
#include "Types_ThermBgdParams.h"

#include "AnalysisInfo.h"
class AnalysisInfo;

#include "FitParameter.h"
#include "LednickyFitter.h"

class FitValuesWriter {

public:

  //Constructor, destructor, copy constructor, assignment operator
  FitValuesWriter();
  virtual ~FitValuesWriter();

  static TString BuildFitInfoTString(bool aApplyMomResCorrection, bool aApplyNonFlatBackgroundCorrection, NonFlatBgdFitType aNonFlatBgdFitType, 
                                     IncludeResidualsType aIncludeResidualsType, ResPrimMaxDecayType aResPrimMaxDecayType=k5fm, 
                                     ChargedResidualsType aChargedResidualsType=kUseXiDataAndCoulombOnlyInterp, bool aFixD0=false,
                                     bool aUseStavCf=false, bool aFixAllLambdaTo1=false, bool aFixAllNormTo1=false, bool aFixRadii=false, bool aFixAllScattParams=false, 
                                     bool aShareLambdaParams=false, bool aAllShareSingleLambdaParam=false, bool aUsemTScalingOfResidualRadii=false, bool aIsDualie=false, 
                                     bool aDualieShareLambda=false, bool aDualieShareRadii=false);

  static TString GetFitInfoTString(TString aLine);
  static AnalysisType GetAnalysisType(TString aLine);
  static CentralityType GetCentralityType(TString aLine);
  static ParameterType GetParamTypeFromName(TString aName);
  static td1dVec ReadParameterValue(TString aLine);
  static vector<vector<FitParameter*> > InterpretFitParamsTStringVec(vector<TString> &aTStringVec);


  static vector<vector<TString> > ConvertMasterTo2dVec(TString aFileLocation);
  static void WriteToMaster(TString aFileLocation, vector<TString> &aFitParamsTStringVec, TString &aFitInfoTString);

  static vector<vector<FitParameter*> > GetAllFitResults(TString aFileLocation, TString aFitInfoTString, AnalysisType aPairAnType);
  static vector<FitParameter*> GetFitResults(TString aFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType);
  static FitParameter* GetFitParameter(TString aFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType, ParameterType aParamType);

  static TGraphAsymmErrors* GetYvsXGraph(TString aFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType, ParameterType aParamTypeY, ParameterType aParamTypeX);
  static void DrawYvsXGraph(TPad* aPad, TString aFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType, ParameterType aParamTypeY, ParameterType aParamTypeX, int aMarkerColor, int aMarkerStyle, double aMarkerSize=0.75, TString aDrawOption = "epsame");

  static TGraphAsymmErrors* GetImF0vsReF0Graph(TString aFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType);
  static void DrawImF0vsReF0Graph(TPad* aPad, TString aFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType, int aMarkerColor, int aMarkerStyle, double aMarkerSize=0.75, TString aDrawOption = "epsame");

  static TGraphAsymmErrors* GetLambdavsRadiusGraph(TString aFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType);
  static void DrawLambdavsRadiusGraph(TPad* aPad, TString aFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType, int aMarkerColor, int aMarkerStyle, double aMarkerSize=0.75, TString aDrawOption = "epsame");

  static TGraphAsymmErrors* GetD0Graph(TString aFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType, double aXOffset=0.5);
  static void DrawD0Graph(TPad* aPad, TString aFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType, double aXOffset, int aMarkerColor, int aMarkerStyle, double aMarkerSize=0.75, TString aDrawOption = "epsame");

  //inline (i.e. simple) functions


private:


#ifdef __ROOT__
  ClassDef(FitValuesWriter, 1)
#endif
};


//inline


#endif
