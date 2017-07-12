///////////////////////////////////////////////////////////////////////////
// FitGenerator:                                                         //
//                                                                       //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef FITGENERATOR_H
#define FITGENERATOR_H

#include <TColor.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>

#include "FitSharedAnalyses.h"
#include "LednickyFitter.h"

#include "CanvasPartition.h"
class CanvasPartition;


class FitGenerator {

public:
  FitGenerator(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, const vector<CentralityType> &aCentralityTypes, AnalysisRunType aRunType=kTrain, int aNPartialAnalysis=2, FitGeneratorType aGeneratorType=kPairwConj, bool aShareLambdaParams=false, bool aAllShareSingleLambdaParam=false, TString aDirNameModifier="");

  FitGenerator(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, CentralityType aCentralityType=kMB, AnalysisRunType aRunType=kTrain, int aNPartialAnalysis=2, FitGeneratorType aGeneratorType=kPairwConj, bool aShareLambdaParams=false, bool aAllShareSingleLambdaParam=false, TString aDirNameModifier="");
  virtual ~FitGenerator();

  void SetNAnalyses();
  void SetupAxis(TAxis* aAxis, double aMin, double aMax, TString aTitle, 
                 float aTitleSize=0.05, float aTitleOffset=0.9, bool aCenterTitle=false, float aLabelSize=0.03, float aLabelOffset=0.005, int aNdivisions=510);
  void SetupAxis(TAxis* aAxis, TString aTitle, 
                 float aTitleSize=0.05, float aTitleOffset=0.9, bool aCenterTitle=false, float aLabelSize=0.03, float aLabelOffset=0.005, int aNdivisions=510);

  void CreateParamInitValuesText(CanvasPartition *aCanPart, int aNx, int aNy, double aTextXmin=0.75, double aTextYmin=0.75, double aTextWidth=0.15, double aTextHeight=0.10, double aTextFont=63, double aTextSize=15);
  void CreateParamFinalValuesText(CanvasPartition *aCanPart, int aNx, int aNy, double aTextXmin=0.75, double aTextYmin=0.75, double aTextWidth=0.15, double aTextHeight=0.10, double aTextFont=63, double aTextSize=15);
  void CreateParamFinalValuesText(CanvasPartition *aCanPart, int aNx, int aNy, TF1* aFit, const double* aSysErrors, double aTextXmin=0.75, double aTextYmin=0.75, double aTextWidth=0.15, double aTextHeight=0.10, double aTextFont=63, double aTextSize=15, bool aDrawAll=true);
  void CreateParamFinalValuesTextTwoColumns(CanvasPartition *aCanPart, int aNx, int aNy, TF1* aFit, const double* aSysErrors, double aText1Xmin=0.75, double aText1Ymin=0.75, double aText1Width=0.15, double aText1Height=0.10, bool aDrawText1=true, double aText2Xmin=0.50, double aText2Ymin=0.75, double aText2Width=0.15, double aText2Height=0.10, bool aDrawText2=true, double aTextFont=63, double aTextSize=15);
  void AddTextCorrectionInfo(CanvasPartition *aCanPart, int aNx, int aNy, bool aMomResCorrect, bool aNonFlatCorrect, double aTextXmin=0.75, double aTextYmin=0.75, double aTextWidth=0.15, double aTextHeight=0.10, double aTextFont=63, double aTextSize=15);

  void DrawSingleKStarCf(TPad* aPad, int aPairAnNumber, double aYmin=0.9, double aYmax=1.1, double aXmin=0.0, double aXmax=0.5, int aMarkerColor=1, TString aOption = "", int aMarkerStyle=20);
  void DrawSingleKStarCfwFit(TPad* aPad, int aPairAnNumber, double aYmin=0.9, double aYmax=1.1, double aXmin=0.0, double aXmax=0.5, int aMarkerColor=1, TString aOption = "", int aMarkerStyle=20);
  virtual TCanvas* DrawKStarCfs(bool aSaveImage=false, bool aDrawSysErrors=true);
  virtual TCanvas* DrawKStarCfswFits(bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType=kLinear, bool aSaveImage=false, bool aDrawSysErrors=true, bool aZoomROP=true);
  virtual TCanvas* DrawResiduals(int aAnalysisNumber, CentralityType aCentralityType=k0010, TString aCanvasName="tCan");
  virtual TCanvas* DrawPrimaryWithResiduals(int aAnalysisNumber, CentralityType aCentralityType=k0010, TString aCanvasName="tCan");

  virtual TCanvas* DrawModelKStarCfs(bool aSaveImage=false);  //TODO add option to choose true, fake, no weight, etc.

  void SetUseLimits(vector<FitParameter> &aVec, bool aUse);  //Internal use only

  void SetRadiusStartValue(double aRad, int aIndex=0);
  void SetRadiusStartValues(const vector<double> &aStartValues);
  void SetRadiusLimits(double aMin, double aMax, int aIndex=0);  //set for particular R parameter
  void SetRadiusLimits(const td2dVec &aMinMax2dVec);  //set unique values for each R parameter

  void SetScattParamStartValue(double aVal, ParameterType aParamType);
  void SetScattParamStartValues(double aReF0, double aImF0, double aD0);
  void SetScattParamLimits(double aMin, double aMax, ParameterType aParamType);
  void SetScattParamLimits(const td2dVec &aMinMax2dVec);

  int GetLambdaBinNumber(bool tConjPair=false, CentralityType aCentType=k0010);  //ex. tConjPair=false for kLamK0 and tConjPair=true for kALamK0
  void SetLambdaParamStartValue(double aLam, bool tConjPair=false, CentralityType aCentType=kMB);
  void SetLambdaParamLimits(double aMin, double aMax, bool tConjPair=false, CentralityType aCentType=kMB);

  void SetDefaultSharedParameters(bool aSetAllUnbounded=false);

  void SetAllParameters();
  void DoFit(bool aApplyMomResCorrection=false, bool aApplyNonFlatBackgroundCorrection=false, bool aIncludeResiduals=false, NonFlatBgdFitType aNonFlatBgdFitType=kLinear, double aMaxFitKStar=0.3);
  void WriteAllFitParameters(ostream &aOut=std::cout);
  vector<TString> GetAllFitParametersTStringVector();

  void FindGoodInitialValues(bool aApplyMomResCorrection=false, bool aApplyNonFlatBackgroundCorrection=false);

  void SetSaveLocationBase(TString aBase, TString aSaveNameModifier="");
  void ExistsSaveLocationBase();

  //inline 
  TString GetSaveLocationBase();

  void SetSharedParameter(ParameterType aParamType);  //share amongst all
  void SetSharedParameter(ParameterType aParamType, double aStartValue, double aLowerBound=0., double aUpperBound=0.);  //share amongst all

  void SetSharedParameter(ParameterType aParamType, const vector<int> &aSharedAnalyses); //share amongst analyses selected in aSharedAnalyses
  void SetSharedParameter(ParameterType aParamType, const vector<int> &aSharedAnalyses, double aStartValue, double aLowerBound=0., double aUpperBound=0.);

  void SetSharedAndFixedParameter(ParameterType aParamType, double aFixedValue);

  void SetParameter(ParameterType aParamType, int aAnalysisNumber, double aStartValue, double aLowerBound=0., double aUpperBound=0.);

  void SetUseRadiusLimits(bool aUse);
  void SetUseScattParamLimits(bool aUse);
  void SetUseLambdaLimits(bool aUse);

  void SetFitType(FitType aFitType);

  double GetChi2();

  TH1* GetKStarCf(int aAnalysisNumber);

protected:
  TString fSaveLocationBase;
  TString fSaveNameModifier;
  bool fContainsMC;
  int fNAnalyses;  //should be 1, 2, 3 or 6
  FitGeneratorType fGeneratorType;
  AnalysisType fPairType, fConjPairType;
  CentralityType fCentralityType;  //Note kMB means include all
  vector<CentralityType> fCentralityTypes;

  vector<FitParameter> fRadiusFitParams;  //size depends on centralities being fit
  vector<FitParameter> fScattFitParams;  //size = 3; [ReF0,ImF0,D0]
  vector<FitParameter> fLambdaFitParams; //size depends on centralities being fit and option chosen for Lambda parameter sharing
  bool fShareLambdaParams; //If true, I will still only share across like centralities
  bool fAllShareSingleLambdaParam;  //If true, only one lambda parameter for all analyses
  vector<vector<FitParameter> > fFitParamsPerPad; //Each 1d Vector = [Lambda,Radius,ReF0,ImF0,D0]

  FitSharedAnalyses* fSharedAn;
  LednickyFitter* fLednickyFitter;



#ifdef __ROOT__
  ClassDef(FitGenerator, 1)
#endif
};

inline TString FitGenerator::GetSaveLocationBase() {return fSaveLocationBase;}

inline void FitGenerator::SetSharedParameter(ParameterType aParamType) 
  {fSharedAn->SetSharedParameter(aParamType);}
inline void FitGenerator::SetSharedParameter(ParameterType aParamType, double aStartValue, double aLowerBound, double aUpperBound) 
  {fSharedAn->SetSharedParameter(aParamType,aStartValue,aLowerBound,aUpperBound);}

inline void FitGenerator::SetSharedParameter(ParameterType aParamType, const vector<int> &aSharedAnalyses) 
  {fSharedAn->SetSharedParameter(aParamType,aSharedAnalyses);}
inline void FitGenerator::SetSharedParameter(ParameterType aParamType, const vector<int> &aSharedAnalyses, double aStartValue, double aLowerBound, double aUpperBound) 
  {fSharedAn->SetSharedParameter(aParamType,aSharedAnalyses,aStartValue,aLowerBound,aUpperBound);}

inline void FitGenerator::SetSharedAndFixedParameter(ParameterType aParamType, double aFixedValue)
  {fSharedAn->SetSharedAndFixedParameter(aParamType,aFixedValue);}

inline void FitGenerator::SetParameter(ParameterType aParamType, int aAnalysisNumber, double aStartValue, double aLowerBound, double aUpperBound)
  {fSharedAn->SetParameter(aParamType,aAnalysisNumber,aStartValue,aLowerBound,aUpperBound);}

inline void FitGenerator::SetUseRadiusLimits(bool aUse) {SetUseLimits(fRadiusFitParams,aUse);}
inline void FitGenerator::SetUseScattParamLimits(bool aUse) {SetUseLimits(fScattFitParams,aUse);}
inline void FitGenerator::SetUseLambdaLimits(bool aUse) {SetUseLimits(fLambdaFitParams,aUse);}

inline void FitGenerator::SetFitType(FitType aFitType) {fSharedAn->SetFitType(aFitType);}

inline double FitGenerator::GetChi2() {return fLednickyFitter->GetChi2();}

inline TH1* FitGenerator::GetKStarCf(int aAnalysisNumber) {return fSharedAn->GetKStarCfHeavy(aAnalysisNumber)->GetHeavyCfClone();}
#endif

