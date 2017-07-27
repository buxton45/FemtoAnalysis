///////////////////////////////////////////////////////////////////////////
// PartialAnalysis:                                                      //
//                                                                       //
//  A partial analysis is essentially a representation of the contents   //
//  of a specific directory returned in an analysis ROOT file            //
//                                                                       //
//  i.e. = specific pair type (ex LamKo),                                //
//         B field configuration (ex Bp1)                                //
//         Centrality (ex 0010)                                          //
//                                                                       //
//  This class should be able to BuildCf, CalculatePurity, Rebin, etc    //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef PARTIALANALYSIS_H
#define PARTIALANALYSIS_H

#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

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

using std::cout;
using std::endl;
using std::vector;

#include "CfLite.h"
class CfLite;

#include "Cf2dLite.h"
class Cf2dLite;

#include "Types.h"

#include "AnalysisInfo.h"
class AnalysisInfo;


class PartialAnalysis {

public:

  PartialAnalysis(TString aFileLocation, TString aAnalysisName, AnalysisType aAnalysisType, BFieldType aBFieldType, CentralityType aCentralityType, AnalysisRunType aRunType=kTrain, TString aDirNameModifier="");
  virtual ~PartialAnalysis();

  TObjArray* ConnectAnalysisDirectory(TString aFileLocation, TString aDirectoryName);

  void SetParticleTypes();
  void SetDaughterParticleTypes();

  TH1* Get1dHisto(TString aHistoName, TString aNewName);
  TH2* Get2dHisto(TString aHistoName, TString aNewName);

  void BuildKStarCf(double aMinNorm=0.32, double aMaxNorm=0.4);
  void BuildKStarCfMCTrue(double aMinNorm=0.32, double aMaxNorm=0.4);

  void BuildModelKStarCfTrue(double aMinNorm=0.32, double aMaxNorm=0.4);  //same event numerators, binned in KStar with Lednicky weight (generated using KStarTrue)
  void BuildModelKStarCfTrueIdeal(double aMinNorm=0.32, double aMaxNorm=0.4);  //same event numerators, binned in KStarTrue with Lednicky weight (generated using KStarTrue)
  void BuildModelKStarCfFake(double aMinNorm=0.32, double aMaxNorm=0.4);  //mixed events numerators, binned in KStar with Lednicky weight (generated using KStarTrue)
  void BuildModelKStarCfFakeIdeal(double aMinNorm=0.32, double aMaxNorm=0.4); //mixed events numerators, binned in KStar with Lednicky weight (generated using KStarTrue)
  void BuildModelKStarCfFakeIdealSmeared(TH2* aMomResMatrix, double aMinNorm=0.32, double aMaxNorm=0.4, int aRebinFactor=1); 

  void BuildModelKStarCfTrueUnitWeights(double aMinNorm=0.32, double aMaxNorm=0.4);
  void BuildModelKStarCfTrueIdealUnitWeights(double aMinNorm=0.32, double aMaxNorm=0.4);

  void BuildModelKStarTrueVsRec(KStarTrueVsRecType aType);
  void BuildAllModelKStarTrueVsRec();
  TH2* GetModelKStarTrueVsRec(KStarTrueVsRecType aType);

  void BuildAvgSepCf(DaughterPairType aDaughterPairType, double aMinNorm=14.99, double aMaxNorm=19.99);
  void BuildAllAvgSepCfs(double aMinNorm=14.99, double aMaxNorm=19.99);
  CfLite* GetAvgSepCf(DaughterPairType aDaughterPairType);

  void BuildSepCfs(DaughterPairType aDaughterPairType, double aMinNorm=14.99, double aMaxNorm=19.99);
  void BuildAllSepCfs(double aMinNorm=14.99, double aMaxNorm=19.99);
  Cf2dLite* GetSepCfs(DaughterPairType aDaughterPairType);

  void BuildKStar2dCfs(double aMinNorm=0.32, double aMaxNorm=0.40);

  void BuildAvgSepCowSailCfs(DaughterPairType aDaughterPairType, double aMinNorm=14.99, double aMaxNorm=19.99);
  void BuildAllAvgSepCowSailCfs(double aMinNorm=14.99, double aMaxNorm=19.99);
  Cf2dLite* GetAvgSepCowSailCfs(DaughterPairType aDaughterPairType);

  TH1* GetPurityHisto(ParticleType aParticleType);

  void SetNEventsPassFail();
  void SetNPart1PassFail();
  void SetNPart2PassFail();
  void SetNKStarNumEntries();

  void SetPart1MassFail();

  TH1* GetMCKchPurityHisto(bool aBeforePairCut);

  //inline
  AnalysisType GetAnalysisType();
  BFieldType GetBFieldType();
  CentralityType GetCentralityType();

  CfLite* GetKStarCf();
  CfLite* GetKStarCfMCTrue();

  CfLite* GetModelKStarCfTrue();
  CfLite* GetModelKStarCfTrueIdeal();
  CfLite* GetModelKStarCfFake();
  CfLite* GetModelKStarCfFakeIdeal();
  CfLite* GetModelKStarCfFakeIdealSmeared();

  CfLite* GetModelKStarCfTrueUnitWeights();
  CfLite* GetModelKStarCfTrueIdealUnitWeights();

//  TH2* GetModelKStarTrueVsRec();
  TH2* GetModelKStarTrueVsRecSame();
  TH2* GetModelKStarTrueVsRecRotSame();
  TH2* GetModelKStarTrueVsRecMixed();
  TH2* GetModelKStarTrueVsRecRotMixed();

  vector<DaughterPairType> GetDaughterPairTypes();

  vector<CfLite*> GetAllAvgSepCfs();
  vector<Cf2dLite*> GetAllSepCfs();

  Cf2dLite* GetKStar2dCfKStarOut();
  Cf2dLite* GetKStar2dCfKStarSide();
  Cf2dLite* GetKStar2dCfKStarLong();

  vector<ParticleType> GetParticleTypes();
  vector<vector<ParticleType> > GetDaughterParticleTypes();

  double GetNEventsPass();
  double GetNEventsFail();
  double GetNPart1Pass();
  double GetNPart1Fail();
  double GetNPart2Pass();
  double GetNPart2Fail();
  double GetNKStarNumEntries();

  TH1* GetPart1MassFail();

private:
  AnalysisRunType fAnalysisRunType;

  TString fFileLocation;
  TString fAnalysisName;
  TString fDirectoryName;

  TObjArray *fDir;

  AnalysisType fAnalysisType;
  TString fAnalysisBaseTag;
  BFieldType fBFieldType;
  TString fBFieldTag;
  CentralityType fCentralityType;
  TString fCentralityTag;

  vector<ParticleType> fParticleTypes;
  vector<vector<ParticleType> > fDaughterParticleTypes;

  //----------
  double fNEventsPass, fNEventsFail;
  double fNPart1Pass, fNPart1Fail;
  double fNPart2Pass, fNPart2Fail;
  double fNKStarNumEntries;

  CfLite *fKStarCf;
  CfLite *fKStarCfMCTrue;

  CfLite* fModelKStarCfTrue;
  CfLite* fModelKStarCfTrueIdeal;
  CfLite* fModelKStarCfFake;
  CfLite* fModelKStarCfFakeIdeal;
  CfLite* fModelKStarCfFakeIdealSmeared;

  CfLite* fModelKStarCfTrueUnitWeights;
  CfLite* fModelKStarCfTrueIdealUnitWeights;

//  TH2* fModelKStarTrueVsRec;
  TH2* fModelKStarTrueVsRecSame;
  TH2* fModelKStarTrueVsRecRotSame;  //Rotated version of above.  x=(KRec+KTrue)/2 and y=(KRec-KTrue)/sqrt(2)
  TH2* fModelKStarTrueVsRecMixed;
  TH2* fModelKStarTrueVsRecRotMixed;  //Rotated version of above.

  vector<DaughterPairType> fDaughterPairTypes;

  vector<CfLite*> fAvgSepCfs;  //4 for V0V0, 2 for V0Track

  vector<Cf2dLite*> fSepCfs;
  vector<Cf2dLite*> fAvgSepCowSailCfs;


  Cf2dLite* fKStar2dCfKStarOut;
  CfLite *fKStar1dCfKStarOutPos, *fKStar1dCfKStarOutNeg;
  TH1* fKStar1dKStarOutPosNegRatio;

  Cf2dLite* fKStar2dCfKStarSide;
  CfLite *fKStar1dCfKStarSidePos, *fKStar1dCfKStarSideNeg;
  TH1* fKStar1dKStarSidePosNegRatio;

  Cf2dLite* fKStar2dCfKStarLong;
  CfLite *fKStar1dCfKStarLongPos, *fKStar1dCfKStarLongNeg;
  TH1* fKStar1dKStarLongPosNegRatio;

  TH1* fPart1MassFail;






#ifdef __ROOT__
  ClassDef(PartialAnalysis, 1)
#endif
};

inline AnalysisType PartialAnalysis::GetAnalysisType() {return fAnalysisType;}
inline BFieldType PartialAnalysis::GetBFieldType() {return fBFieldType;}
inline CentralityType PartialAnalysis::GetCentralityType() {return fCentralityType;}

inline CfLite* PartialAnalysis::GetKStarCf() {return fKStarCf;}
inline CfLite* PartialAnalysis::GetKStarCfMCTrue() {return fKStarCfMCTrue;}

inline CfLite* PartialAnalysis::GetModelKStarCfTrue() {return fModelKStarCfTrue;}
inline CfLite* PartialAnalysis::GetModelKStarCfTrueIdeal() {return fModelKStarCfTrueIdeal;}
inline CfLite* PartialAnalysis::GetModelKStarCfFake() {return fModelKStarCfFake;}
inline CfLite* PartialAnalysis::GetModelKStarCfFakeIdeal() {return fModelKStarCfFakeIdeal;}
inline CfLite* PartialAnalysis::GetModelKStarCfFakeIdealSmeared() {return fModelKStarCfFakeIdealSmeared;}

inline CfLite* PartialAnalysis::GetModelKStarCfTrueUnitWeights() {return fModelKStarCfTrueUnitWeights;}
inline CfLite* PartialAnalysis::GetModelKStarCfTrueIdealUnitWeights() {return fModelKStarCfTrueIdealUnitWeights;}

//inline TH2* PartialAnalysis::GetModelKStarTrueVsRec() {return fModelKStarTrueVsRec;}
inline TH2* PartialAnalysis::GetModelKStarTrueVsRecSame() {return fModelKStarTrueVsRecSame;}
inline TH2* PartialAnalysis::GetModelKStarTrueVsRecRotSame() {return fModelKStarTrueVsRecRotSame;}
inline TH2* PartialAnalysis::GetModelKStarTrueVsRecMixed() {return fModelKStarTrueVsRecSame;}
inline TH2* PartialAnalysis::GetModelKStarTrueVsRecRotMixed() {return fModelKStarTrueVsRecRotSame;}

inline vector<DaughterPairType> PartialAnalysis::GetDaughterPairTypes() {return fDaughterPairTypes;}

inline vector<CfLite*> PartialAnalysis::GetAllAvgSepCfs() {return fAvgSepCfs;}
inline vector<Cf2dLite*> PartialAnalysis::GetAllSepCfs() {return fSepCfs;}

inline Cf2dLite* PartialAnalysis::GetKStar2dCfKStarOut() {return fKStar2dCfKStarOut;}
inline Cf2dLite* PartialAnalysis::GetKStar2dCfKStarSide() {return fKStar2dCfKStarSide;}
inline Cf2dLite* PartialAnalysis::GetKStar2dCfKStarLong() {return fKStar2dCfKStarLong;}

inline vector<ParticleType> PartialAnalysis::GetParticleTypes() {return fParticleTypes;}

inline vector<vector<ParticleType> > PartialAnalysis::GetDaughterParticleTypes() {return fDaughterParticleTypes;}

inline double PartialAnalysis::GetNEventsPass() {return fNEventsPass;}
inline double PartialAnalysis::GetNEventsFail() {return fNEventsFail;}
inline double PartialAnalysis::GetNPart1Pass() {return fNPart1Pass;}
inline double PartialAnalysis::GetNPart1Fail() {return fNPart1Fail;}
inline double PartialAnalysis::GetNPart2Pass() {return fNPart2Pass;}
inline double PartialAnalysis::GetNPart2Fail() {return fNPart2Fail;}
inline double PartialAnalysis::GetNKStarNumEntries() {return fNKStarNumEntries;}

inline TH1* PartialAnalysis::GetPart1MassFail() {return fPart1MassFail;}


#endif

