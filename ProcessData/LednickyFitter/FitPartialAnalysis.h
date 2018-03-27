///////////////////////////////////////////////////////////////////////////
// FitPartialAnalysis:                                                   //
//     This object represents a particular pair system (LamKchP),        //
//     in a particular centrality bin (i.e. 0010),                       //
//     and a particular division of the dataset (i.e. Bp1)               //
//                                                                       //
//     ex. FitPartialAnalysis = LamKchP0010_Bp1                          //
//                                                                       //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef FITPARTIALANALYSIS_H
#define FITPARTIALANALYSIS_H

//includes and any constant variable declarations
#include <iostream>
#include <iomanip>
#include <complex>
#include <math.h>
#include <vector>
#include <assert.h>

#include "Faddeeva.hh"

#include "TF1.h"
#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TPaveText.h"
#include "TStyle.h"
#include "TAxis.h"
#include "TGraph.h"
#include "TMath.h"
#include "TDirectoryFile.h"

#include "TSystem.h"
#include "TMinuit.h"
#include "TVirtualFitter.h"

using std::cout;
using std::endl;
using std::vector;

#include "FitParameter.h"
class FitParameter;

#include "BackgroundFitter.h"
class BackgroundFitter;

#include "CfLite.h"
class CfLite;

#include "Types.h"
#include "Types_LambdaValues.h"
#include "Types_ThermBgdParams.h"

#include "AnalysisInfo.h"
class AnalysisInfo;

class FitPartialAnalysis {

public:

  //Constructor, destructor, copy constructor, assignment operator
  FitPartialAnalysis(TString aFileLocation, TString aAnalysisName, AnalysisType aAnalysisType, CentralityType aCentralityType, BFieldType aBFieldType, AnalysisRunType aRunType=kTrain, TString aDirNameModifier="", bool aIncludeSingletAndTriplet=false);
  FitPartialAnalysis(TString aFileLocation, TString aFileLocationMC, TString aAnalysisName, AnalysisType aAnalysisType, CentralityType aCentralityType, BFieldType aBFieldType, AnalysisRunType aRunType=kTrain, TString aDirNameModifier="", bool aIncludeSingletAndTriplet=false);
  virtual ~FitPartialAnalysis();

  static double GetLednickyF1(double z);
  static double GetLednickyF2(double z);
  static double LednickyEq(double *x, double *par);
  static double LednickyEqWithNorm(double *x, double *par);

  TObjArray* ConnectAnalysisDirectory(TString aFileLocation, TString aDirectoryName);

  void SetParticleTypes();

  TH1* Get1dHisto(TString aHistoName, TString aNewName);
  TH1* Get1dHisto(TString aFileLocation, TString aHistoName, TString aNewName);
  TH2* Get2dHisto(TString aHistoName, TString aNewName);
  TH2* Get2dHisto(TString aFileLocation, TString aDirectoryName, TString aHistoName, TString aNewName);

  void BuildKStarCf(double aKStarMinNorm=0.32, double aKStarMaxNorm=0.4);
  void RebinKStarCf(int aRebinFactor, double aKStarMinNorm=0.32, double aKStarMaxNorm=0.4);

  void CreateFitFunction(bool aApplyNorm, IncludeResidualsType aIncResType, ResPrimMaxDecayType aResPrimMaxDecayType, double aChi2, int aNDF, 
                         double aKStarMin=0.0, double aKStarMax=1.0, TString aBaseName="Fit");

  //----------- Used when fitting background first and separate from everything else (old method)
  static TF1* FitNonFlatBackground(TH1* aNum, TH1* aDen, TH1* aCf, NonFlatBgdFitType aBgdFitType, FitType aFitType, bool aNormalizeFitToCf, 
                                   double aMinBgdFit=0.60, double aMaxBgdFit=0.90, double aMaxBgdBuild=2.0, double aKStarMinNorm=0.32, double aKStarMaxNorm=0.40);
  static TF1* FitNonFlatBackground(TH1* aCf, NonFlatBgdFitType aBgdFitType, 
                                   double aMinBgdFit=0.6, double aMaxBgdFit=0.9, double aMaxBgdBuild=2.0, double aKStarMinNorm=0.32, double aKStarMaxNorm=0.40);

  TF1* GetThermNonFlatBackground();
  TF1* GetNonFlatBackground(NonFlatBgdFitType aBgdFitType, FitType aFitType, bool aNormalizeFitToCf);


  //---------------------------------------------------------------------------------------------
  TF1* GetNewNonFlatBackground(NonFlatBgdFitType aBgdFitType);
  void InitializeBackgroundParams(NonFlatBgdFitType aNonFlatBgdType);
  void SetBgdParametersSharedLocal(bool aIsShared, vector<int> &aSharedAnalyses);
  void SetBgdParametersSharedGlobal(bool aIsShared, vector<int> &aSharedAnalyses);

  void SetFitParameterShallow(FitParameter* aParam);
  void SetBgdParametersShallow(vector<FitParameter*> &aBgdParameters);

  CfLite* GetModelKStarCf(double aKStarMinNorm=0.32, double aKStarMaxNorm=0.40, int aRebin=1);
  CfLite* GetModelKStarCfFake(double aKStarMinNorm, double aKStarMaxNorm, int aRebin=1);
  CfLite* GetModelKStarCfFakeIdeal(double aKStarMinNorm, double aKStarMaxNorm, int aRebin=1);

  //inline (i.e. simple) functions
  AnalysisType GetAnalysisType();
  BFieldType GetBFieldType();
  CentralityType GetCentralityType();

  void SetFitPartialAnalysisNumber(int aAnalysisNumber);  //will be set by the FitPairAnalysis object
  int GetFitPartialAnalysisNumber();

  CfLite* GetKStarCfLite();
  TH1* GetKStarCf();
  TH1* GetNumKStarCf();
  TH1* GetDenKStarCf();

  double GetKStarMinNorm();
  double GetKStarMaxNorm();
  void SetKStarMinMaxNorm(double aMin, double aMax);

  double GetMinBgdFit();
  double GetMaxBgdFit();
  void SetMinMaxBgdFit(double aMin, double aMax);
  void SetMaxBgdBuild(double aMaxBuild);

  double GetKStarNumScale();
  double GetKStarDenScale();

  vector<ParticleType> GetParticleTypes();

  int GetNFitParams();
  vector<FitParameter*> GetFitParameters();
  FitParameter* GetFitNormParameter();
  FitParameter* GetFitParameter(ParameterType aParamType);

  vector<FitParameter*> GetBgdParameters();
  FitParameter* GetBgdParameter(int aIdx);

  void SetRejectOmega(bool aRejectOmega);
  bool RejectOmega();

  TH2* GetModelKStarTrueVsRecMixed();

  void SetCorrectedFitVec(td1dVec &aVec);
  td1dVec GetCorrectedFitVec();

  TF1* GetPrimaryFit();

private:
  AnalysisRunType fAnalysisRunType;
  TString fFileLocation;
  TString fFileLocationMC;
  TString fAnalysisName;
  TString fDirectoryName;

  AnalysisType fAnalysisType;
  BFieldType fBFieldType;
  CentralityType fCentralityType;

  int fFitPartialAnalysisNumber;  //to help identify in FitPairAnalysis (set in FitPairAnalysis);

  vector<ParticleType> fParticleTypes;

  CfLite *fKStarCfLite;
  double fMinBgdFit, fMaxBgdFit;
  double fMaxBgdBuild;
  bool fNormalizeBgdFitToCf;

  int fNFitParams;
  FitParameter* fLambda;
  FitParameter* fRadius;
  FitParameter* fRef0;
  FitParameter* fImf0;
  FitParameter* fd0;
  FitParameter* fRef02;
  FitParameter* fImf02;
  FitParameter* fd02;
  vector<FitParameter*> fFitParameters;
  FitParameter* fNorm;
  vector<FitParameter*> fBgdParameters;

  bool fRejectOmega;

  TH2* fModelKStarTrueVsRecMixed;
  CfLite* fModelKStarCfFake;
  CfLite* fModelKStarCfFakeIdeal;

  TF1* fPrimaryFit;
  TF1* fNonFlatBackground;
  td1dVec fCorrectedFitVec;

#ifdef __ROOT__
  ClassDef(FitPartialAnalysis, 1)
#endif
};


//inline stuff
inline AnalysisType FitPartialAnalysis::GetAnalysisType() {return fAnalysisType;}
inline BFieldType FitPartialAnalysis::GetBFieldType() {return fBFieldType;}
inline CentralityType FitPartialAnalysis::GetCentralityType() {return fCentralityType;}

inline void FitPartialAnalysis::SetFitPartialAnalysisNumber(int aAnalysisNumber) {fFitPartialAnalysisNumber = aAnalysisNumber;}
inline int FitPartialAnalysis::GetFitPartialAnalysisNumber() {return fFitPartialAnalysisNumber;}

inline CfLite* FitPartialAnalysis::GetKStarCfLite() {return fKStarCfLite;}
inline TH1* FitPartialAnalysis::GetKStarCf() {return fKStarCfLite->Cf();}
inline TH1* FitPartialAnalysis::GetNumKStarCf() {return fKStarCfLite->Num();}
inline TH1* FitPartialAnalysis::GetDenKStarCf() {return fKStarCfLite->Den();}

inline double FitPartialAnalysis::GetKStarMinNorm() {return fKStarCfLite->GetMinNorm();}
inline double FitPartialAnalysis::GetKStarMaxNorm() {return fKStarCfLite->GetMaxNorm();}
inline void FitPartialAnalysis::SetKStarMinMaxNorm(double aMin, double aMax) {RebinKStarCf(1, aMin, aMax);}

inline double FitPartialAnalysis::GetMinBgdFit() {return fMinBgdFit;}
inline double FitPartialAnalysis::GetMaxBgdFit() {return fMaxBgdFit;}
inline void FitPartialAnalysis::SetMinMaxBgdFit(double aMin, double aMax) {fMinBgdFit=aMin; fMaxBgdFit=aMax;}
inline void FitPartialAnalysis::SetMaxBgdBuild(double aMaxBuild) {fMaxBgdBuild=aMaxBuild;}

inline double FitPartialAnalysis::GetKStarNumScale() {return fKStarCfLite->GetNumScale();}
inline double FitPartialAnalysis::GetKStarDenScale() {return fKStarCfLite->GetDenScale();}

inline vector<ParticleType> FitPartialAnalysis::GetParticleTypes() {return fParticleTypes;}

inline int FitPartialAnalysis::GetNFitParams() {return fNFitParams;}
inline vector<FitParameter*> FitPartialAnalysis::GetFitParameters() {return fFitParameters;}
inline FitParameter* FitPartialAnalysis::GetFitNormParameter() {return fNorm;}
inline FitParameter* FitPartialAnalysis::GetFitParameter(ParameterType aParamType) {return fFitParameters[aParamType];}

inline vector<FitParameter*> FitPartialAnalysis::GetBgdParameters() {return fBgdParameters;}
inline FitParameter* FitPartialAnalysis::GetBgdParameter(int aIdx) {return fBgdParameters[aIdx];}

inline void FitPartialAnalysis::SetRejectOmega(bool aRejectOmega) {fRejectOmega = aRejectOmega;}
inline bool FitPartialAnalysis::RejectOmega() {return fRejectOmega;}

inline TH2* FitPartialAnalysis::GetModelKStarTrueVsRecMixed() {return fModelKStarTrueVsRecMixed;}

inline void FitPartialAnalysis::SetCorrectedFitVec(td1dVec &aVec) {fCorrectedFitVec = aVec;}
inline td1dVec FitPartialAnalysis::GetCorrectedFitVec() {return fCorrectedFitVec;}

inline TF1* FitPartialAnalysis::GetPrimaryFit() {return fPrimaryFit;}

#endif
