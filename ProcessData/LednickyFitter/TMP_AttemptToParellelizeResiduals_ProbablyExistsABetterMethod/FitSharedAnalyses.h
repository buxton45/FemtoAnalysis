///////////////////////////////////////////////////////////////////////////
// FitSharedAnalyses:                                                    //
///////////////////////////////////////////////////////////////////////////

#ifndef FITSHAREDANALYSES_H
#define FITSHAREDANALYSES_H

//includes and any constant variable declarations
#include <iostream>
#include <iomanip>
#include <complex>
#include <math.h>
#include <vector>
#include <assert.h>

#include "Faddeeva.hh"

#include "TF1.h"
#include "TH1F.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TPaveText.h"
#include "TStyle.h"
#include "TAxis.h"
#include "TGraph.h"
#include "TMath.h"

#include "TSystem.h"
#include "TMinuit.h"
#include "TVirtualFitter.h"

using std::cout;
using std::cin;
using std::endl;
using std::vector;

#include "FitPairAnalysis.h"
class FitPairAnalysis;

#include "FitChi2Histograms.h"
class FitChi2Histograms;

#include "ParallelSharedResidualCollection.h"
class ParallelSharedResidualCollection;

class FitSharedAnalyses {

public:
  //Constructor, destructor, copy constructor, assignment operator
  FitSharedAnalyses(vector<FitPairAnalysis*> &aVecOfFitPairAnalyses);
  FitSharedAnalyses(vector<FitPairAnalysis*> &aVecOfFitPairAnalyses, vector<ParameterType> &aVecOfSharedParameterTypes);
  virtual ~FitSharedAnalyses();

  void SetParameter(ParameterType aParamType, int aAnalysisNumber, double aStartValue, double aLowerBound=0., double aUpperBound=0., bool aIsFixed=false);

  void CompareParameters(TString aAnalysisName1, FitParameter* aParam1, TString aAnalysisName2, FitParameter* aParam2);

  void SetSharedParameter(ParameterType aParamType, bool aIsFixed=false);  //share amongst all
  void SetSharedParameter(ParameterType aParamType, double aStartValue, double aLowerBound=0., double aUpperBound=0., bool aIsFixed=false);  //share amongst all

  void SetSharedParameter(ParameterType aParamType, const vector<int> &aSharedAnalyses, bool aIsFixed=false); //share amongst analyses selected in aSharedAnalyses
  void SetSharedParameter(ParameterType aParamType, const vector<int> &aSharedAnalyses, double aStartValue, double aLowerBound=0., double aUpperBound=0., bool aIsFixed=false);

  void SetSharedAndFixedParameter(ParameterType aParamType, double aFixedValue);

  vector<FitParameter*> GetDistinctParamsOfCommonType(ParameterType aParamType);  
  void CreateMinuitParametersMatrix();  //call after all parameters shared!!!!!

  void CreateMinuitParameter(int aMinuitParamNumber, FitParameter* aParam);
  void CreateMinuitParameters();

  void ReturnFitParametersToAnalyses();

  double GetKStarMinNorm();
  double GetKStarMaxNorm();

  double GetMinBgdFit();
  double GetMaxBgdFit();

  BinInfoTransformMatrix BuildBinInfoTransformMatrix(AnalysisType aDaughterAnType, AnalysisType aParentResType, CentralityType aCentType, TH2D* aTransformMatrix);
  td2dVec BuildTransformVec(TH2D* aTransformMatrix);
  void BuildParallelSharedResidualCollection();

  //inline (i.e. simple) functions

  int GetNMinuitParams();

  void SetFitType(FitType aFitType);
  FitType GetFitType();

  void SetApplyNonFlatBackgroundCorrection(bool aApply);
  bool GetApplyNonFlatBackgroundCorrection();

  void SetNonFlatBgdFitType(NonFlatBgdFitType aFitType);
  NonFlatBgdFitType GetNonFlatBgdFitType();

  TMinuit* GetMinuitObject();
  int GetNFitPairAnalysis();

  int GetNFitParamsPerAnalysis();
  int GetNFitNormParamsPerAnalysis();

  void SetMinuitMinParams(vector<double> &aMinuitMinParams);
  void SetMinuitParErrors(vector<double> &aMinuitParErrors);

  vector<double> GetMinuitMinParams();
  vector<double> GetMinuitParErrors();

  FitPairAnalysis* GetFitPairAnalysis(int aPairAnalysisNumber);
  CfHeavy* GetKStarCfHeavy(int aPairAnalysisNumber);
  CfHeavy* GetModelKStarCfHeavy(int aPairAnalysisNumber);

  void DrawFit(int aAnalysisNumber, const char* aTitle);

  void RebinAnalyses(int aRebin);

  void SetKStarMinMaxNorm(double aMin, double aMax);
  void SetMinMaxBgdFit(double aMin, double aMax);

  FitChi2Histograms* GetFitChi2Histograms();

  void SetFixNormParams(bool aFixNormParams);

private:
  TMinuit* fMinuit;
  FitType fFitType; //kChi2PML = default, or kChi2;
  bool fApplyNonFlatBackgroundCorrection;
  NonFlatBgdFitType fNonFlatBgdFitType; //kLinear = default
  int fNFitPairAnalysis;

  int fNFitParamsPerAnalysis;
  int fNFitNormParamsPerAnalysis;
  bool fFixNormParams;

  vector<FitPairAnalysis*> fFitPairAnalysisCollection;

  int fNMinuitParams;
  vector<double> fMinuitMinParams;
  vector<double> fMinuitParErrors;

  vector<vector<FitParameter*> > fMinuitFitParametersMatrix;  //The number of rows will always be equal to fNFitParamsPerAnalysis + fNFitPairAnalysis
  //The number of columns depends on how many analyses share the parameter
  //If parameter is distinct for each analysis, there will be fNAnalyses columns
  //If parameter is shared by all, there will be 1 column 

  FitChi2Histograms* fFitChi2Histograms;

  ParallelSharedResidualCollection* fParallelSharedResidualCollection;

#ifdef __ROOT__
  ClassDef(FitSharedAnalyses, 1)
#endif
};

//inline stuff

inline int FitSharedAnalyses::GetNMinuitParams() {return fNMinuitParams;}

inline void FitSharedAnalyses::SetFitType(FitType aFitType) {fFitType = aFitType;}
inline FitType FitSharedAnalyses::GetFitType() {return fFitType;}

inline void FitSharedAnalyses::SetApplyNonFlatBackgroundCorrection(bool aApply) {fApplyNonFlatBackgroundCorrection = aApply;}
inline bool FitSharedAnalyses::GetApplyNonFlatBackgroundCorrection() {return fApplyNonFlatBackgroundCorrection;}

inline void FitSharedAnalyses::SetNonFlatBgdFitType(NonFlatBgdFitType aFitType) {fNonFlatBgdFitType = aFitType;}
inline NonFlatBgdFitType FitSharedAnalyses::GetNonFlatBgdFitType() {return fNonFlatBgdFitType;}

inline TMinuit* FitSharedAnalyses::GetMinuitObject() {return fMinuit;}
inline int FitSharedAnalyses::GetNFitPairAnalysis() {return fNFitPairAnalysis;}

inline int FitSharedAnalyses::GetNFitParamsPerAnalysis() {return fNFitParamsPerAnalysis;}
inline int FitSharedAnalyses::GetNFitNormParamsPerAnalysis() {return fNFitNormParamsPerAnalysis;}

inline void FitSharedAnalyses::SetMinuitMinParams(vector<double> &aMinuitMinParams) {fMinuitMinParams = aMinuitMinParams;}
inline void FitSharedAnalyses::SetMinuitParErrors(vector<double> &aMinuitParErrors) {fMinuitParErrors = aMinuitParErrors;}

inline vector<double> FitSharedAnalyses::GetMinuitMinParams() {return fMinuitMinParams;}
inline vector<double> FitSharedAnalyses::GetMinuitParErrors() {return fMinuitParErrors;}


inline FitPairAnalysis* FitSharedAnalyses::GetFitPairAnalysis(int aPairAnalysisNumber) {return fFitPairAnalysisCollection[aPairAnalysisNumber];}
inline CfHeavy* FitSharedAnalyses::GetKStarCfHeavy(int aPairAnalysisNumber) {return fFitPairAnalysisCollection[aPairAnalysisNumber]->GetKStarCfHeavy();}
inline CfHeavy* FitSharedAnalyses::GetModelKStarCfHeavy(int aPairAnalysisNumber) {return fFitPairAnalysisCollection[aPairAnalysisNumber]->GetModelKStarHeavyCf();}

inline void FitSharedAnalyses::DrawFit(int aAnalysisNumber, const char* aTitle) {fFitPairAnalysisCollection[aAnalysisNumber]->DrawFit(aTitle);}

inline void FitSharedAnalyses::RebinAnalyses(int aRebin) {for(int i=0; i<fNFitPairAnalysis; i++) fFitPairAnalysisCollection[i]->RebinKStarCfHeavy(aRebin);}
inline void FitSharedAnalyses::SetKStarMinMaxNorm(double aMin, double aMax) {for(int i=0; i<fNFitPairAnalysis; i++) fFitPairAnalysisCollection[i]->SetKStarMinMaxNorm(aMin, aMax);}
inline void FitSharedAnalyses::SetMinMaxBgdFit(double aMin, double aMax) {for(int i=0; i<fNFitPairAnalysis; i++) fFitPairAnalysisCollection[i]->SetMinMaxBgdFit(aMin, aMax);}

inline FitChi2Histograms* FitSharedAnalyses::GetFitChi2Histograms() {return fFitChi2Histograms;}

inline void FitSharedAnalyses::SetFixNormParams(bool aFixNormParams) {fFixNormParams = aFixNormParams;}

#endif


