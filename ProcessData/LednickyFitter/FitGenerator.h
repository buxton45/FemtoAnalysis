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

#include "Types_FitParamValues.h"

#include "FitValuesWriter.h"

class FitGenerator {
friend class DualieFitGenerator;  //So DualieFitGenerator can access private FitGenerator members/functions
friend class TripleFitGenerator;  //So TripleFitGenerator can access private FitGenerator members/functions

public:
  FitGenerator(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, const vector<CentralityType> &aCentralityTypes, AnalysisRunType aRunType=kTrain, int aNPartialAnalysis=2, FitGeneratorType aGeneratorType=kPairwConj, bool aShareLambdaParams=false, bool aAllShareSingleLambdaParam=false, TString aDirNameModifier="", bool aUseStavCf=false);

  FitGenerator(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, CentralityType aCentralityType=kMB, AnalysisRunType aRunType=kTrain, int aNPartialAnalysis=2, FitGeneratorType aGeneratorType=kPairwConj, bool aShareLambdaParams=false, bool aAllShareSingleLambdaParam=false, TString aDirNameModifier="", bool aUseStavCf=false);
  virtual ~FitGenerator();

  void SetNAnalyses();

  void SetUseLimits(vector<FitParameter> &aVec, bool aUse);  //Internal use only

  void SetRadiusStartValue(double aRad, int aIndex=0);
  void SetRadiusStartValues(const vector<double> &aStartValues);
  void SetRadiusLimits(double aMin, double aMax, int aIndex=0);  //set for particular R parameter
  void SetRadiusLimits(const td2dVec &aMinMax2dVec);  //set unique values for each R parameter
  void SetAllRadiiLimits(double aMin, double aMax);

  void SetScattParamStartValue(double aVal, ParameterType aParamType, bool aIsFixed=false);
  void SetScattParamStartValues(double aReF0, double aImF0, double aD0, bool aAreFixed=false);
  void SetScattParamLimits(double aMin, double aMax, ParameterType aParamType);
  void SetScattParamLimits(const td2dVec &aMinMax2dVec);

  int GetLambdaBinNumber(bool tConjPair=false, CentralityType aCentType=k0010);  //ex. tConjPair=false for kLamK0 and tConjPair=true for kALamK0
  void SetLambdaParamStartValue(double aLam, bool tConjPair=false, CentralityType aCentType=kMB, bool aIsFixed=false);
  void SetAllLambdaParamStartValues(const vector<double> &aLams, bool aAreFixed=false);
  void SetLambdaParamLimits(double aMin, double aMax, bool tConjPair=false, CentralityType aCentType=kMB);
  void SetAllLambdaParamLimits(double aMin, double aMax);

  void SetDefaultSharedParameters(bool aSetAllUnbounded=false);
  void SetDefaultLambdaParametersWithResiduals(double aMinLambda=0., double aMaxLambda=1.0);

  void SetAllParameters();
  void InitializeGenerator(double aMaxFitKStar=0.3);  //Called withith DoFit
  void DoFit(double aMaxFitKStar=0.3);

  TCanvas* GenerateContourPlots(int aNPoints, const vector<double> &aParams, const vector<double> &aErrVals={4,1}, TString aSaveNameModifier="", bool aFixAllOthers=false, double aMaxFitKStar=0.3);  //1=1sigma, 4=2sigma
  TCanvas* GenerateContourPlots(int aNPoints, CentralityType aCentType, const vector<double> &aErrVals={4,1}, bool aFixAllOthers=false, double aMaxFitKStar=0.3);  //1=1sigma, 4=2sigma

  void WriteAllFitParameters(ostream &aOut=std::cout);
  vector<TString> GetAllFitParametersTStringVector();
  void WriteToMasterFitValuesFile(TString aFileLocation, TString aResultsDate);

  void FindGoodInitialValues(bool aApplyMomResCorrection=false, bool aApplyNonFlatBackgroundCorrection=false);

  void SetSaveLocationBase(TString aBase, TString aSaveNameModifier="");
  void ExistsSaveLocationBase();

  //inline 
  TString GetSaveLocationBase();
  TString GetSaveNameModifier();

  void SetSharedParameter(ParameterType aParamType, bool aIsFixed=false);  //share amongst all
  void SetSharedParameter(ParameterType aParamType, double aStartValue, double aLowerBound=0., double aUpperBound=0., bool aIsFixed=false);  //share amongst all

  void SetSharedParameter(ParameterType aParamType, const vector<int> &aSharedAnalyses, bool aIsFixed=false); //share amongst analyses selected in aSharedAnalyses
  void SetSharedParameter(ParameterType aParamType, const vector<int> &aSharedAnalyses, double aStartValue, double aLowerBound=0., double aUpperBound=0., bool aIsFixed=false);

  void SetSharedAndFixedParameter(ParameterType aParamType, double aFixedValue);

  void SetParameter(ParameterType aParamType, int aAnalysisNumber, double aStartValue, double aLowerBound=0., double aUpperBound=0., bool aIsFixed=false);

  void SetUseRadiusLimits(bool aUse);
  void SetUseScattParamLimits(bool aUse);
  void SetUseLambdaLimits(bool aUse);

  void SetFitType(FitType aFitType);

  double GetChi2();
  int GetNDF();

  CfHeavy* GetKStarCfHeavy(int aAnalysisNumer);
  TH1* GetKStarCf(int aAnalysisNumber);
  void SetKStarMinMaxNorm(double aMin, double aMax);
  void SetMinMaxBgdFit(double aMin, double aMax);
  FitSharedAnalyses* GetFitSharedAnalyses();

  void SetApplyNonFlatBackgroundCorrection(bool aApply);
  void SetNonFlatBgdFitType(NonFlatBgdFitType aNonFlatBgdFitType);
  void SetUseNewBgdTreatment(bool aUse);
  void SetApplyMomResCorrection(bool aApplyMomResCorrection);
  virtual void SetIncludeResidualCorrelationsType(IncludeResidualsType aIncludeResidualsType, double aMinLambda=0., double aMaxLambda=1.0);
  void SetChargedResidualsType(ChargedResidualsType aChargedResidualsType);
  void SetResPrimMaxDecayType(ResPrimMaxDecayType aResPrimMaxDecayType);

  void SetUsemTScalingOfResidualRadii(bool aUse=true, double aPower=-0.5);

  void SetSaveFileType(TString aType);
  TString GetSaveFileType();

  int GetNAnalyses();
  FitGeneratorType GetGeneratorType();
  const vector<CentralityType> GetCentralityTypes();
  FitSharedAnalyses* GetSharedAn();

  void SetFixNormParams(bool aFixNormParams);

  void SetMasterFileLocation(TString aLocation);
  void SetSystematicsFileLocation(TString aLocation);

  TString GetMasterFileLocation();
  TString GetSystematicsFileLocation();

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

  bool fApplyNonFlatBackgroundCorrection; //TODO eliminate this in favor of fSharedAn->GetApplyNonFlatBackgroundCorrection
  NonFlatBgdFitType fNonFlatBgdFitType;  //TODO eliminate this in favor of fSharedAn->GetNonFlatBgdType()
                                         //TODO can probably eliminate other similar types of redundancies
  bool fApplyMomResCorrection;

  IncludeResidualsType fIncludeResidualsType;
  ChargedResidualsType fChargedResidualsType;
  ResPrimMaxDecayType fResPrimMaxDecayType;

  bool fUsemTScalingOfResidualRadii;
  double fmTScalingPowerOfResidualRadii;

  FitSharedAnalyses* fSharedAn;
  LednickyFitter* fLednickyFitter;

  TString fSaveFileType;

  TString fMasterFileLocation;
  TString fSystematicsFileLocation;

#ifdef __ROOT__
  ClassDef(FitGenerator, 1)
#endif
};

inline TString FitGenerator::GetSaveLocationBase() {return fSaveLocationBase;}
inline TString FitGenerator::GetSaveNameModifier() {return fSaveNameModifier;}

inline void FitGenerator::SetSharedParameter(ParameterType aParamType, bool aIsFixed) 
  {fSharedAn->SetSharedParameter(aParamType,aIsFixed);}
inline void FitGenerator::SetSharedParameter(ParameterType aParamType, double aStartValue, double aLowerBound, double aUpperBound, bool aIsFixed) 
  {fSharedAn->SetSharedParameter(aParamType,aStartValue,aLowerBound,aUpperBound,aIsFixed);}

inline void FitGenerator::SetSharedParameter(ParameterType aParamType, const vector<int> &aSharedAnalyses, bool aIsFixed) 
  {fSharedAn->SetSharedParameter(aParamType,aSharedAnalyses,aIsFixed);}
inline void FitGenerator::SetSharedParameter(ParameterType aParamType, const vector<int> &aSharedAnalyses, double aStartValue, double aLowerBound, double aUpperBound, bool aIsFixed) 
  {fSharedAn->SetSharedParameter(aParamType,aSharedAnalyses,aStartValue,aLowerBound,aUpperBound,aIsFixed);}

inline void FitGenerator::SetSharedAndFixedParameter(ParameterType aParamType, double aFixedValue)
  {fSharedAn->SetSharedAndFixedParameter(aParamType,aFixedValue);}

inline void FitGenerator::SetParameter(ParameterType aParamType, int aAnalysisNumber, double aStartValue, double aLowerBound, double aUpperBound, bool aIsFixed)
  {fSharedAn->SetParameter(aParamType,aAnalysisNumber,aStartValue,aLowerBound,aUpperBound,aIsFixed);}

inline void FitGenerator::SetUseRadiusLimits(bool aUse) {SetUseLimits(fRadiusFitParams,aUse);}
inline void FitGenerator::SetUseScattParamLimits(bool aUse) {SetUseLimits(fScattFitParams,aUse);}
inline void FitGenerator::SetUseLambdaLimits(bool aUse) {SetUseLimits(fLambdaFitParams,aUse);}

inline void FitGenerator::SetFitType(FitType aFitType) {fSharedAn->SetFitType(aFitType);}

inline double FitGenerator::GetChi2() {return fLednickyFitter->GetChi2();}
inline int FitGenerator::GetNDF() {return fLednickyFitter->GetNDF();}

inline CfHeavy* FitGenerator::GetKStarCfHeavy(int aAnalysisNumber) {return fSharedAn->GetKStarCfHeavy(aAnalysisNumber);}
inline TH1* FitGenerator::GetKStarCf(int aAnalysisNumber) {return fSharedAn->GetKStarCfHeavy(aAnalysisNumber)->GetHeavyCfClone();}
inline void FitGenerator::SetKStarMinMaxNorm(double aMin, double aMax) {fSharedAn->SetKStarMinMaxNorm(aMin, aMax);}
inline void FitGenerator::SetMinMaxBgdFit(double aMin, double aMax) {fSharedAn->SetMinMaxBgdFit(aMin, aMax);}

inline FitSharedAnalyses* FitGenerator::GetFitSharedAnalyses() {return fSharedAn;}

inline void FitGenerator::SetApplyNonFlatBackgroundCorrection(bool aApply) {fApplyNonFlatBackgroundCorrection = aApply; fSharedAn->SetApplyNonFlatBackgroundCorrection(aApply);}
inline void FitGenerator::SetNonFlatBgdFitType(NonFlatBgdFitType aNonFlatBgdFitType) {fNonFlatBgdFitType = aNonFlatBgdFitType; fSharedAn->SetNonFlatBgdFitType(aNonFlatBgdFitType);}
inline void FitGenerator::SetUseNewBgdTreatment(bool aUse) {fSharedAn->SetUseNewBgdTreatment(aUse);}
inline void FitGenerator::SetApplyMomResCorrection(bool aApplyMomResCorrection) {fApplyMomResCorrection = aApplyMomResCorrection;}
inline void FitGenerator::SetIncludeResidualCorrelationsType(IncludeResidualsType aIncludeResidualsType, double aMinLambda, double aMaxLambda) {fIncludeResidualsType = aIncludeResidualsType; if(aIncludeResidualsType != kIncludeNoResiduals) SetDefaultLambdaParametersWithResiduals(aMinLambda, aMaxLambda);}
inline void FitGenerator::SetChargedResidualsType(ChargedResidualsType aChargedResidualsType) {fChargedResidualsType = aChargedResidualsType;}
inline void FitGenerator::SetResPrimMaxDecayType(ResPrimMaxDecayType aResPrimMaxDecayType) {fResPrimMaxDecayType = aResPrimMaxDecayType;}

inline void FitGenerator::SetUsemTScalingOfResidualRadii(bool aUse, double aPower) {fUsemTScalingOfResidualRadii = aUse; fmTScalingPowerOfResidualRadii = aPower;}

inline void FitGenerator::SetSaveFileType(TString aType) {fSaveFileType = aType;}
inline TString FitGenerator::GetSaveFileType() {return fSaveFileType;}

inline int FitGenerator::GetNAnalyses() {return fNAnalyses;}
inline FitGeneratorType FitGenerator::GetGeneratorType() {return fGeneratorType;}
inline const vector<CentralityType> FitGenerator::GetCentralityTypes() {return fCentralityTypes;}
inline FitSharedAnalyses* FitGenerator::GetSharedAn() {return fSharedAn;}

inline void FitGenerator::SetFixNormParams(bool aFixNormParams) {fSharedAn->SetFixNormParams(aFixNormParams);}

inline void FitGenerator::SetMasterFileLocation(TString aLocation) {fMasterFileLocation = aLocation;}
inline void FitGenerator::SetSystematicsFileLocation(TString aLocation) {fSystematicsFileLocation = aLocation;}

inline TString FitGenerator::GetMasterFileLocation() {return fMasterFileLocation;}
inline TString FitGenerator::GetSystematicsFileLocation() {return fSystematicsFileLocation;}

#endif

