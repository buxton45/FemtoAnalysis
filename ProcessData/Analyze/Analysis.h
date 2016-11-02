///////////////////////////////////////////////////////////////////////////
// Analysis:                                                             //
//                                                                       //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef ANALYSIS_H
#define ANALYSIS_H

#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

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
#include "TLegend.h"

using std::cout;
using std::cin;
using std::endl;
using std::vector;

#include "PartialAnalysis.h"
class PartialAnalysis;

#include "CfHeavy.h"
class CfHeavy;

#include "Cf2dHeavy.h"
class Cf2dHeavy;

#include "Purity.h"
class Purity;

class Analysis {

public:

  Analysis(TString aAnalysisName, vector<PartialAnalysis*> &aPartialAnalysisCollection, bool aCombineConjugates=false);

  Analysis(TString aFileLocationBase, AnalysisType aAnalysisType, CentralityType aCentralityType, AnalysisRunType aRunType=kTrain, int aNPartialAnalysis=2, TString aDirNameModifier="");
  virtual ~Analysis();

  TH1* SimpleAddTH1Collection(TString tHistosName);
  vector<ParticleType> GetCorrectDaughterParticleTypes(DaughterPairType aDaughterPairType);
  TString GetDaughtersHistoTitle(DaughterPairType aDaughterPairType);
  //-----
  void BuildKStarHeavyCf(double aMinNorm=0.32, double aMaxNorm=0.4, int aRebin=1);
  void DrawKStarHeavyCf(TPad* aPad, int aMarkerColor=1, TString aOption = "", int aMarkerStyle=20);
  void SaveAllKStarHeavyCf(TFile* aFile);
  //-----
  void BuildKStarHeavyCfMCTrue(double aMinNorm=0.32, double aMaxNorm=0.4);
  void DrawKStarHeavyCfMCTrue(TPad* aPad, int aMarkerColor=1, TString aOption = "", int aMarkerStyle=20);
  //-----
  void BuildModelKStarHeavyCfTrue(double aMinNorm=0.32, double aMaxNorm=0.4);
  void DrawModelKStarHeavyCfTrue(TPad* aPad, int aMarkerColor=1, TString aOption = "", int aMarkerStyle=20);
  void BuildModelKStarHeavyCfTrueIdeal(double aMinNorm=0.32, double aMaxNorm=0.4);
  void BuildModelKStarHeavyCfFake(double aMinNorm=0.32, double aMaxNorm=0.4);
  void BuildModelKStarHeavyCfFakeIdeal(double aMinNorm=0.32, double aMaxNorm=0.4);
  void BuildModelKStarHeavyCfFakeIdealSmeared(TH2* aMomResMatrix, double aMinNorm=0.32, double aMaxNorm=0.4, int aRebinFactor=1);

  void BuildModelCfTrueIdealCfTrueRatio(double aMinNorm=0.32, double aMaxNorm=0.4, int aRebinFactor=1);
  void BuildModelCfFakeIdealCfFakeRatio(double aMinNorm=0.32, double aMaxNorm=0.4, int aRebinFactor=1);
  void FitModelCfTrueIdealCfTrueRatio();
  void FitModelCfFakeIdealCfFakeRatio();

  void BuildModelKStarHeavyCfTrueUnitWeights(double aMinNorm=0.32, double aMaxNorm=0.4, int aRebin=1);
  void BuildModelKStarHeavyCfTrueIdealUnitWeights(double aMinNorm=0.32, double aMaxNorm=0.4, int aRebin=1);
  //-----
//  void BuildModelKStarTrueVsRecTotal();
  void NormalizeTH2ByTotalEntries(TH2* aHist);
  void NormalizeTH2EachColumn(TH2* aHist);
  void NormalizeTH2EachRow(TH2* aHist);

  void BuildModelKStarTrueVsRecTotal(KStarTrueVsRecType aType);
  void BuildAllModelKStarTrueVsRecTotal();
  TH2* GetModelKStarTrueVsRecTotal(KStarTrueVsRecType aType);

  void BuildMomResMatrixFit(KStarTrueVsRecType aType, double aKStarLow, double aKStarHigh);
  TH2* GetMomResMatrixFit(KStarTrueVsRecType aType);

  //-----
  void BuildAvgSepHeavyCf(DaughterPairType aDaughterPairType, double aMinNorm=14.99, double aMaxNorm=19.99);
  void BuildAllAvgSepHeavyCfs(double aMinNorm=14.99, double aMaxNorm=19.99);
  CfHeavy* GetAvgSepHeavyCf(DaughterPairType aDaughterPairType);
  void DrawAvgSepHeavyCf(DaughterPairType aDaughterPairType, TPad* aPad);
  void SaveAllAvgSepHeavyCfs(TFile* aFile);
  //-----
  void BuildKStar2dHeavyCfs(double aMinNorm=0.32, double aMaxNorm=0.40);
  void RebinKStar2dHeavyCfs(int aRebinFactor);
  void DrawKStar2dHeavyCfKStarOutRatio(TPad* aPad, int aMarkerColor=1, TString aOption = "", int aMarkerStyle=20);
  void DrawKStar2dHeavyCfKStarSideRatio(TPad* aPad, int aMarkerColor=1, TString aOption = "", int aMarkerStyle=20);
  void DrawKStar2dHeavyCfKStarLongRatio(TPad* aPad, int aMarkerColor=1, TString aOption = "", int aMarkerStyle=20);
  void DrawKStar2dHeavyCfRatios(TPad* aPad);

  //-----
  void BuildSepHeavyCfs(DaughterPairType aDaughterPairType, double aMinNorm=14.99, double aMaxNorm=19.99);
  void BuildAllSepHeavyCfs(double aMinNorm=14.99, double aMaxNorm=19.99);
  Cf2dHeavy* GetSepHeavyCfs(DaughterPairType aDaughterPairType);
  void DrawSepHeavyCfs(DaughterPairType aDaughterPairType, TPad* aPad);
  //-----
  void BuildAvgSepCowSailHeavyCfs(DaughterPairType aDaughterPairType, double aMinNorm=14.99, double aMaxNorm=19.99);
  void BuildAllAvgSepCowSailHeavyCfs(double aMinNorm=14.99, double aMaxNorm=19.99);
  Cf2dHeavy* GetAvgSepCowSailHeavyCfs(DaughterPairType aDaughterPairType);
  void DrawAvgSepCowSailHeavyCfs(DaughterPairType aDaughterPairType, TPad* aPad);
  //-----
  void BuildPurityCollection();
  void DrawAllPurityHistos(TPad* aPad);
  double GetPurity(ParticleType aV0Type);

  void OutputPassFailInfo();

  void DrawPart1MassFail(TPad* aPad, bool aDrawWideRangeToo=false);

  //-----
  TH1* GetMassAssumingK0ShortHypothesis();
  TH1* GetMassAssumingLambdaHypothesis();
  TH1* GetMassAssumingAntiLambdaHypothesis();

  //inline
  TString GetAnalysisName();
  AnalysisType GetAnalysisType();
  CentralityType GetCentralityType();

  vector<PartialAnalysis*> GetPartialAnalysisCollection();

  CfHeavy* GetKStarHeavyCf();
  CfHeavy* GetKStarHeavyCfCopy();
  CfHeavy* GetKStarHeavyCfMCTrue();

  CfHeavy* GetModelKStarHeavyCfTrue();
  CfHeavy* GetModelKStarHeavyCfTrueIdeal();
  CfHeavy* GetModelKStarHeavyCfFake();
  CfHeavy* GetModelKStarHeavyCfFakeIdeal();
  CfHeavy* GetModelKStarHeavyCfFakeIdealSmeared();
  TH1* GetModelCfTrueIdealCfTrueRatio();
  TF1* GetMomResFit();
  TH1* GetModelCfFakeIdealCfFakeRatio();
  TF1* GetMomResFitFake();

  CfHeavy* GetModelKStarHeavyCfTrueUnitWeights();
  CfHeavy* GetModelKStarHeavyCfTrueIdealUnitWeights();

//  TH2* GetModelKStarTrueVsRecTotal();

  vector<CfHeavy*> GetAllAvgSepHeavyCfs();

  TH1* GetKStar1dHeavyKStarOutPosNegRatio();
  TH1* GetKStar1dHeavyKStarSidePosNegRatio();
  TH1* GetKStar1dHeavyKStarLongPosNegRatio();

  vector<Purity*> GetPurityCollection();

  Cf2dHeavy* GetKStar2dHeavyCfKStarOut();
  Cf2dHeavy* GetKStar2dHeavyCfKStarSide();
  Cf2dHeavy* GetKStar2dHeavyCfKStarLong();

  double GetNEventsPass();
  double GetNEventsFail();
  double GetNPart1Pass();
  double GetNPart1Fail();
  double GetNPart2Pass();
  double GetNPart2Fail();
  double GetNKStarNumEntries();

  TH1* GetPart1MassFail();

  void GetMCKchPurity(bool aBeforePairCut);
  bool AreTrainResults();

private:
  AnalysisRunType fAnalysisRunType;
  bool fCombineConjugates;

  TString fAnalysisName;
  vector<PartialAnalysis*> fPartialAnalysisCollection;
  
  AnalysisType fAnalysisType;
  CentralityType fCentralityType;

  vector<ParticleType> fParticleTypes;
  vector<vector<ParticleType> > fDaughterParticleTypes;

  //----------


  int fNPartialAnalysis;

  double fNEventsPass, fNEventsFail;
  double fNPart1Pass, fNPart1Fail;
  double fNPart2Pass, fNPart2Fail;
  double fNKStarNumEntries;

  CfHeavy *fKStarHeavyCf;
  CfHeavy *fKStarHeavyCfMCTrue;

  CfHeavy* fModelKStarHeavyCfTrue;
  CfHeavy* fModelKStarHeavyCfTrueIdeal;
  CfHeavy* fModelKStarHeavyCfFake;
  CfHeavy* fModelKStarHeavyCfFakeIdeal;
  CfHeavy* fModelKStarHeavyCfFakeIdealSmeared;
  TH1* fModelCfTrueIdealCfTrueRatio;
  TF1* fMomResFit;
  TH1* fModelCfFakeIdealCfFakeRatio;
  TF1* fMomResFitFake;

  CfHeavy* fModelKStarHeavyCfTrueUnitWeights;
  CfHeavy* fModelKStarHeavyCfTrueIdealUnitWeights;

//  TH2* fModelKStarTrueVsRecTotal;
  TH2* fModelKStarTrueVsRecSameTot;
  TH2* fModelKStarTrueVsRecRotSameTot;  //Rotated version of above.  x=(KRec+KTrue)/2 and y=(KRec-KTrue)/sqrt(2)
  TH2* fModelKStarTrueVsRecMixedTot;
  TH2* fModelKStarTrueVsRecRotMixedTot;  //Rotated version of above.

  TH2* fMomResMatrixFitSame;
  TH2* fMomResMatrixFitMixed;

  vector<DaughterPairType> fDaughterPairTypes;

  vector<CfHeavy*> fAvgSepHeavyCfs;

  vector<Cf2dHeavy*> fSepHeavyCfs;
  vector<Cf2dHeavy*> fAvgSepCowSailHeavyCfs;

/*
  Cf2dHeavy* fKStar2dHeavyCf;
  CfHeavy *fKStar1dHeavyCfPos, *fKStar1dHeavyCfNeg;
  TH1* fKStar1dHeavyPosNegRatio;
*/

  Cf2dHeavy* fKStar2dHeavyCfKStarOut;
  CfHeavy *fKStar1dHeavyCfKStarOutPos, *fKStar1dHeavyCfKStarOutNeg;
  TH1* fKStar1dHeavyKStarOutPosNegRatio;

  Cf2dHeavy* fKStar2dHeavyCfKStarSide;
  CfHeavy *fKStar1dHeavyCfKStarSidePos, *fKStar1dHeavyCfKStarSideNeg;
  TH1* fKStar1dHeavyKStarSidePosNegRatio;

  Cf2dHeavy* fKStar2dHeavyCfKStarLong;
  CfHeavy *fKStar1dHeavyCfKStarLongPos, *fKStar1dHeavyCfKStarLongNeg;
  TH1* fKStar1dHeavyKStarLongPosNegRatio;


  vector<Purity*> fPurityCollection;

  TH1* fPart1MassFail;


#ifdef __ROOT__
  ClassDef(Analysis, 1)
#endif
};

inline TString Analysis::GetAnalysisName() {return fAnalysisName;}
inline AnalysisType Analysis::GetAnalysisType() {return fAnalysisType;}
inline CentralityType Analysis::GetCentralityType() {return fCentralityType;}

inline vector<PartialAnalysis*> Analysis::GetPartialAnalysisCollection() {return fPartialAnalysisCollection;}

inline CfHeavy* Analysis::GetKStarHeavyCf() {return fKStarHeavyCf;}
inline CfHeavy* Analysis::GetKStarHeavyCfCopy() {CfHeavy* tKStarHeavyCfCopy = new CfHeavy(*fKStarHeavyCf); return tKStarHeavyCfCopy;}
inline CfHeavy* Analysis::GetKStarHeavyCfMCTrue() {return fKStarHeavyCfMCTrue;}

inline CfHeavy* Analysis::GetModelKStarHeavyCfTrue() {return fModelKStarHeavyCfTrue;}
inline CfHeavy* Analysis::GetModelKStarHeavyCfTrueIdeal() {return fModelKStarHeavyCfTrueIdeal;}
inline CfHeavy* Analysis::GetModelKStarHeavyCfFake() {return fModelKStarHeavyCfFake;}
inline CfHeavy* Analysis::GetModelKStarHeavyCfFakeIdeal() {return fModelKStarHeavyCfFakeIdeal;}
inline CfHeavy* Analysis::GetModelKStarHeavyCfFakeIdealSmeared() {return fModelKStarHeavyCfFakeIdealSmeared;}
inline TH1* Analysis::GetModelCfTrueIdealCfTrueRatio() {return fModelCfTrueIdealCfTrueRatio;}
inline TF1* Analysis::GetMomResFit() {return fMomResFit;}
inline TH1* Analysis::GetModelCfFakeIdealCfFakeRatio() {return fModelCfFakeIdealCfFakeRatio;}
inline TF1* Analysis::GetMomResFitFake() {return fMomResFitFake;}

inline CfHeavy* Analysis::GetModelKStarHeavyCfTrueUnitWeights() {return fModelKStarHeavyCfTrueUnitWeights;}
inline CfHeavy* Analysis::GetModelKStarHeavyCfTrueIdealUnitWeights() {return fModelKStarHeavyCfTrueIdealUnitWeights;}

//inline TH2* Analysis::GetModelKStarTrueVsRecTotal() {return fModelKStarTrueVsRecTotal;}

inline vector<CfHeavy*> Analysis::GetAllAvgSepHeavyCfs() {return fAvgSepHeavyCfs;}

inline TH1* Analysis::GetKStar1dHeavyKStarOutPosNegRatio() {return fKStar1dHeavyKStarOutPosNegRatio;}
inline TH1* Analysis::GetKStar1dHeavyKStarSidePosNegRatio() {return fKStar1dHeavyKStarSidePosNegRatio;}
inline TH1* Analysis::GetKStar1dHeavyKStarLongPosNegRatio() {return fKStar1dHeavyKStarLongPosNegRatio;}

inline vector<Purity*> Analysis::GetPurityCollection() {return fPurityCollection;}

inline Cf2dHeavy* Analysis::GetKStar2dHeavyCfKStarOut() {return fKStar2dHeavyCfKStarOut;}
inline Cf2dHeavy* Analysis::GetKStar2dHeavyCfKStarSide() {return fKStar2dHeavyCfKStarSide;}
inline Cf2dHeavy* Analysis::GetKStar2dHeavyCfKStarLong() {return fKStar2dHeavyCfKStarLong;}

inline double Analysis::GetNEventsPass() {return fNEventsPass;}
inline double Analysis::GetNEventsFail() {return fNEventsFail;}
inline double Analysis::GetNPart1Pass() {return fNPart1Pass;}
inline double Analysis::GetNPart1Fail() {return fNPart1Fail;}
inline double Analysis::GetNPart2Pass() {return fNPart2Pass;}
inline double Analysis::GetNPart2Fail() {return fNPart2Fail;}
inline double Analysis::GetNKStarNumEntries() {return fNKStarNumEntries;}

inline TH1* Analysis::GetPart1MassFail() {return fPart1MassFail;}

inline bool Analysis::AreTrainResults() {if(fAnalysisRunType==kTrain || fAnalysisRunType==kTrainSys) return true;}

#endif

