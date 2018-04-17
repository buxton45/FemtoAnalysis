///////////////////////////////////////////////////////////////////////////
// FitPairAnalysis:                                                      //
//     This object represents a particular pair system (LamKchP),        //
//     in a particular centrality bin (i.e. 0010)                        //
//                                                                       //
//     i.e. this contains all (5) divisions of the dataset (Bp1, Bp2,    //
//     Bm1, Bm2, and Bm3) together                                       //
//                                                                       //
//     ex. LamKchP0010                                                   //
///////////////////////////////////////////////////////////////////////////

#ifndef FITPAIRANALYSIS_H
#define FITPAIRANALYSIS_H

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
using std::endl;
using std::vector;

#include "FitPartialAnalysis.h"
class FitPartialAnalysis;

#include "ResidualCollection.h"
class ResidualCollection;

class FitPairAnalysis {

public:

  //Constructor, destructor, copy constructor, assignment operator
  FitPairAnalysis(TString aAnalysisName, vector<FitPartialAnalysis*> &aFitPartialAnalysisCollection, bool aIncludeSingletAndTriplet=false);
  FitPairAnalysis(TString aFileLocationBase, AnalysisType aAnalysisType, CentralityType aCentralityType, AnalysisRunType aRunType=kTrain, int aNFitPartialAnalysis=2, TString aDirNameModifier="", bool aUseNumRotPar2InsteadOfDen=false, bool aIncludeSingletAndTriplet=false);
  FitPairAnalysis(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, CentralityType aCentralityType, AnalysisRunType aRunType=kTrain, int aNFitPartialAnalysis=2, TString aDirNameModifier="", bool aUseNumRotPar2InsteadOfDen=false, bool aIncludeSingletAndTriplet=false);
  virtual ~FitPairAnalysis();

  void BuildModelKStarTrueVsRecMixed(int aRebinFactor=1);
  void BuildKStarCfHeavy(double aKStarMinNorm=0.32, double aKStarMaxNorm=0.4);
  void RebinKStarCfHeavy(int aRebinFactor, double aKStarMinNorm=0.32, double aKStarMaxNorm=0.4);
  void DrawKStarCfHeavy(TPad* aPad, int aMarkerColor=1, TString aOption = "", int aMarkerStyle=20);

  void CreateFitFunction(IncludeResidualsType aIncResType, ResPrimMaxDecayType aResPrimMaxDecayType, double aChi2, int aNDF, 
                         double aKStarMin=0.0, double aKStarMax=1.0, TString aBaseName="Fit");

  TH1* GetThermNonFlatBackground(bool aCombineConj=true, bool aCombineLamKchPM=false, ThermEventsType aThermEventsType=kMeAndAdam);
  void DivideCfByThermBgd(bool aCombineConj=true, bool aCombineLamKchPM=false, ThermEventsType aThermEventsType=kMeAndAdam);
  //----------- Used when fitting background first and separate from everything else (old method)
  TF1* GetNonFlatBackground_FitCombinedPartials(NonFlatBgdFitType aBgdFitType, FitType aFitType, bool aNormalizeFitToCf);
  TF1* GetNonFlatBackground_CombinePartialFits(NonFlatBgdFitType aBgdFitType, FitType aFitType, bool aNormalizeFitToCf);
  TF1* GetNonFlatBackground(NonFlatBgdFitType aBgdFitType, FitType aFitType, bool aNormalizeFitToCf, bool aCombinePartialFits);
  //---------------------------------------------------------------------------------------------
  TF1* GetNewNonFlatBackground(NonFlatBgdFitType aBgdFitType, bool aShareAmongstPartials);
  void InitializeBackgroundParams(NonFlatBgdFitType aNonFlatBgdType, bool aShareAmongstPartials);
  void SetBgdParametersShallow(vector<FitParameter*> &aBgdParameters);
  void SetBgdParametersSharedGlobal(bool aIsShared, vector<int> &aSharedAnalyses);

  void CreateFitNormParameters();
  void ShareFitParameters(bool aIncludeSingletAndTriplet=false);
  void WriteFitParameters(ostream &aOut=std::cout);
  vector<TString> GetFitParametersTStringVector();
  vector<double> GetFitParametersVector();

  void SetFitParameterShallow(FitParameter* aParam);

  CfHeavy* GetModelKStarHeavyCf(double aKStarMinNorm=0.32, double aKStarMaxNorm=0.40, int aRebin=4);
  void BuildModelKStarHeavyCfFake(double aKStarMinNorm, double aKStarMaxNorm, int aRebin=1);
  void BuildModelKStarHeavyCfFakeIdeal(double aKStarMinNorm, double aKStarMaxNorm, int aRebin=1);
  void BuildModelCfFakeIdealCfFakeRatio(double aKStarMinNorm=0.32, double aKStarMaxNorm=0.4, int aRebinFactor=1);

  TH1F* GetCorrectedFitHisto(bool aMomResCorrection=true, bool aNonFlatBgdCorrection=true, bool aIncludeResiduals=false, NonFlatBgdFitType aNonFlatBgdFitType=kLinear, FitType aFitType=kChi2PML);

  void LoadTransformMatrices(IncludeResidualsType aIncludeResidualsType, int aRebin=2, TString aFileLocation="");
  vector<TH2D*> GetTransformMatrices(IncludeResidualsType aIncludeResidualsType, int aRebin=2, TString aFileLocation="");
  TH2D* GetTransformMatrix(IncludeResidualsType aIncludeResidualsType, int aIndex, int aRebin=2, TString aFileLocation="");
  TH2D* GetTransformMatrix(IncludeResidualsType aIncludeResidualsType, AnalysisType aResidualType, int aRebin=2, TString aFileLocation="");

  TH1* GetCfwSysErrors();

  td1dVec GetCorrectedFitVec();
  TH1F* GetCorrectedFitHistv2(double aMaxDrawKStar=1.0); //TODO
  void InitiateResidualCollection(td1dVec &aKStarBinCenters, IncludeResidualsType aIncludeResidualsType, ChargedResidualsType aChargedResidualsType, ResPrimMaxDecayType aResPrimMaxDecayType, TString aInterpCfsDirectory="/home/jesse/Analysis/FemtoAnalysis/ProcessData/CoulombFitter/");

  //inline (i.e. simple) functions
  TString GetAnalysisName();
  TString GetAnalysisDirectoryName();

  vector<FitPartialAnalysis*> GetFitPartialAnalysisCollection();
  FitPartialAnalysis* GetFitPartialAnalysis(int aPartialAnalysisNumber);

  int GetNFitPartialAnalysis();

  AnalysisType GetAnalysisType();
  CentralityType GetCentralityType();

  void SetFitPairAnalysisNumber(int aAnalysisNumber);  //will be set by the FitSharedAnalysis object
  int GetFitPairAnalysisNumber();

  int GetNFitParams();
  int GetNFitNormParams();

  vector<FitParameter*> GetFitParameters();
  FitParameter* GetFitParameter(ParameterType aParamType);

  vector<vector<FitParameter*> > Get2dBgdParameters();
  vector<FitParameter*> Get1dBgdParameters(int aPartAn);

  vector<FitParameter*> GetFitNormParameters();
  FitParameter* GetFitNormParameter(int aFitPartialAnalysisNumber);

  CfHeavy* GetKStarCfHeavy();
  TH1* GetKStarCf();

  double GetKStarMinNorm();
  double GetKStarMaxNorm();
  void SetKStarMinMaxNorm(double aMin, double aMax);

  double GetMinBgdFit();
  double GetMaxBgdFit();
  void SetMinMaxBgdFit(double aMin, double aMax);
  void SetMaxBgdBuild(double aMaxBuild);

  void SetPrimaryFit(TF1* aFit);
  TF1* GetPrimaryFit();

  void SetupAxis(TAxis* aAxis, double aMin, double aMax, TString aTitle, float aTitleSize, float aTitleOffset, bool aCenterTitle, float aLabelSize, float aLabelOffset, int aNdivisions);
  void SetupAxis(TAxis* aAxis, TString aTitle, float aTitleSize, float aTitleOffset, bool aCenterTitle, float aLabelSize, float aLabelOffset, int aNdivisions);

  void DrawFit(const char* aTitle);

  TH2* GetModelKStarTrueVsRecMixed();
  TH1* GetModelCfFakeIdealCfFakeRatio();

  bool AreTrainResults();

  void SetPrimaryWithResiduals(td1dVec &aPrimWithResid);

  vector<AnalysisType> GetTransformStorageMapping();
  td1dVec GetNeutralResidualCorrelation(AnalysisType aResidualType, double *aParentCfParams);
  td1dVec GetTransformedNeutralResidualCorrelation(AnalysisType aResidualType, double *aParentCfParams);
  td1dVec CombinePrimaryWithResiduals(double *aCfParams, td1dVec &aPrimaryCf);
  ResidualCollection* GetResidualCollection();

private:
  AnalysisRunType fAnalysisRunType;
  TString fAnalysisName;
  TString fAnalysisDirectoryName;
  vector<FitPartialAnalysis*> fFitPartialAnalysisCollection;
  int fNFitPartialAnalysis;

  AnalysisType fAnalysisType;
  CentralityType fCentralityType;

  int fFitPairAnalysisNumber;  //to help identify in FitSharedAnalysis (set in FitSharedAnalysis);

  vector<ParticleType> fParticleTypes;

  CfHeavy *fKStarCfHeavy;
  double fMinBgdFit, fMaxBgdFit;
  double fMaxBgdBuild;
  bool fNormalizeBgdFitToCf;

  TF1* fPrimaryFit;
  TF1* fNonFlatBackground;
  TH1* fThermNonFlatBgdHist;


  int fNFitParams;
  int fNFitParamsToShare;
  int fNFitNormParams;

  vector<FitParameter*> fFitNormParameters;        //typical case, there will be 5 (one for each Bp1, Bp2, etc.)
  vector<FitParameter*> fFitParameters;            //typical case, there will be 5(Lambda,Radius,Ref0,Imf0,d0)
  vector<vector<FitParameter*> > f2dBgdParameters;   //If sharing between partials, f2dBgdParameters.size()==1
                                                     //  otherwise, f2dBgdParameters.size() == fNFitPartialAnalysis

  TH2* fModelKStarTrueVsRecMixed;
  CfHeavy* fModelKStarHeavyCfFake;
  CfHeavy* fModelKStarHeavyCfFakeIdeal;
  TH1* fModelCfFakeIdealCfFakeRatio;

  vector<TH2D*> fTransformMatrices;
  vector<AnalysisType> fTransformStorageMapping;

  ResidualCollection *fResidualCollection;
  td1dVec fPrimaryWithResiduals;

  bool fUseNumRotPar2InsteadOfDen;

#ifdef __ROOT__
  ClassDef(FitPairAnalysis, 1)
#endif
};


//inline stuff
inline TString FitPairAnalysis::GetAnalysisName() {return fAnalysisName;}
inline TString FitPairAnalysis::GetAnalysisDirectoryName() {return fAnalysisDirectoryName;}

inline vector<FitPartialAnalysis*> FitPairAnalysis::GetFitPartialAnalysisCollection() {return fFitPartialAnalysisCollection;}
inline FitPartialAnalysis* FitPairAnalysis::GetFitPartialAnalysis(int aPartialAnalysisNumber) {return fFitPartialAnalysisCollection[aPartialAnalysisNumber];}

inline int FitPairAnalysis::GetNFitPartialAnalysis() {return fNFitPartialAnalysis;}

inline AnalysisType FitPairAnalysis::GetAnalysisType() {return fAnalysisType;}
inline CentralityType FitPairAnalysis::GetCentralityType() {return fCentralityType;}

inline void FitPairAnalysis::SetFitPairAnalysisNumber(int aAnalysisNumber) {fFitPairAnalysisNumber = aAnalysisNumber;}
inline int FitPairAnalysis::GetFitPairAnalysisNumber() {return fFitPairAnalysisNumber;}

inline int FitPairAnalysis::GetNFitParams() {return fNFitParams;}
inline int FitPairAnalysis::GetNFitNormParams() {return fNFitNormParams;}

inline vector<FitParameter*> FitPairAnalysis::GetFitParameters() {return fFitParameters;}
inline FitParameter* FitPairAnalysis::GetFitParameter(ParameterType aParamType) {return fFitParameters[aParamType];}

inline vector<vector<FitParameter*> > FitPairAnalysis::Get2dBgdParameters() {return f2dBgdParameters;}
inline vector<FitParameter*> FitPairAnalysis::Get1dBgdParameters(int aPartAn) {return f2dBgdParameters[aPartAn];}

inline vector<FitParameter*> FitPairAnalysis::GetFitNormParameters() {return fFitNormParameters;}
inline FitParameter* FitPairAnalysis::GetFitNormParameter(int aFitPartialAnalysisNumber) {return fFitNormParameters[aFitPartialAnalysisNumber];}

inline CfHeavy* FitPairAnalysis::GetKStarCfHeavy() {return fKStarCfHeavy;}
inline TH1* FitPairAnalysis::GetKStarCf() {return fKStarCfHeavy->GetHeavyCf();}

inline double FitPairAnalysis::GetKStarMinNorm() {return fKStarCfHeavy->GetMinNorm();}
inline double FitPairAnalysis::GetKStarMaxNorm() {return fKStarCfHeavy->GetMaxNorm();}
inline void FitPairAnalysis::SetKStarMinMaxNorm(double aMin, double aMax) {RebinKStarCfHeavy(1, aMin, aMax);}  //Propagated through to CfLite objects, no need to for loop through
                                                                                                               //fFitPartialAnalysisCollection and call SetKStarMinMaxNorm on each

inline double FitPairAnalysis::GetMinBgdFit() {return fMinBgdFit;}
inline double FitPairAnalysis::GetMaxBgdFit() {return fMaxBgdFit;}
inline void FitPairAnalysis::SetMinMaxBgdFit(double aMin, double aMax) {fMinBgdFit=aMin; fMaxBgdFit=aMax; for(int i=0; i<fNFitPartialAnalysis; i++) fFitPartialAnalysisCollection[i]->SetMinMaxBgdFit(aMin, aMax);}
inline void FitPairAnalysis::SetMaxBgdBuild(double aMaxBuild) {fMaxBgdBuild=aMaxBuild; for(int i=0; i<fNFitPartialAnalysis; i++) fFitPartialAnalysisCollection[i]->SetMaxBgdBuild(aMaxBuild);}

inline void FitPairAnalysis::SetPrimaryFit(TF1* aFit) {fPrimaryFit = aFit;}
inline TF1* FitPairAnalysis::GetPrimaryFit() {return fPrimaryFit;}

inline TH2* FitPairAnalysis::GetModelKStarTrueVsRecMixed() {return fModelKStarTrueVsRecMixed;}
inline TH1* FitPairAnalysis::GetModelCfFakeIdealCfFakeRatio() {return fModelCfFakeIdealCfFakeRatio;}

inline bool FitPairAnalysis::AreTrainResults() {if(fAnalysisRunType==kTrain || fAnalysisRunType==kTrainSys) return true;}

inline void FitPairAnalysis::SetPrimaryWithResiduals(td1dVec &aPrimWithResid) {fPrimaryWithResiduals=aPrimWithResid;}

inline vector<AnalysisType> FitPairAnalysis::GetTransformStorageMapping() {return fTransformStorageMapping;}
inline td1dVec FitPairAnalysis::GetNeutralResidualCorrelation(AnalysisType aResidualType, double *aParentCfParams) {return fResidualCollection->GetNeutralResidualCorrelation(aResidualType, aParentCfParams);}
inline td1dVec FitPairAnalysis::GetTransformedNeutralResidualCorrelation(AnalysisType aResidualType, double *aParentCfParams) {return fResidualCollection->GetTransformedNeutralResidualCorrelation(aResidualType, aParentCfParams);}
inline td1dVec FitPairAnalysis::CombinePrimaryWithResiduals(double *aCfParams, td1dVec &aPrimaryCf) {return fResidualCollection->CombinePrimaryWithResiduals(aCfParams, aPrimaryCf);}
inline ResidualCollection* FitPairAnalysis::GetResidualCollection() {return fResidualCollection;}

#endif





