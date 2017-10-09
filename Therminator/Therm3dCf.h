/* Therm3dCf.h */

#ifndef THERM3DCF_H
#define THERM3DCF_H


#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "ThermEventsCollection.h"
#include "TApplication.h"
#include "TStyle.h"
#include "TPaveText.h"
#include "TLegend.h"

#include "PIDMapping.h"
#include "ThermCommon.h"

#include "ThermPairAnalysis.h"

using namespace std;

class Therm3dCf {

public:
  Therm3dCf(AnalysisType aAnType, TString aFileLocation="/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/CorrelationFunctions_10MixedEvNum.root", int aRebin=2);
  virtual ~Therm3dCf();

  void SetPartTypes();

  TH1D* GetFull(TH3D* aHist3d, TString aPreName);
  TH1D* GetPrimaryOnly(TH3D* aHist3d, TString aPreName);
  TH1D* GetSecondaryOnly(TH3D* aHist3d, TString aPreName);
  TH1D* GetAtLeastOneSecondaryInPair(TH3D* aHist3d, TString aPreName);
  TH1D* GetWithoutSigmaSt(TH3D* aHist3d, TString aPreName);

  TH1D* GetSigmaStOnly(TH3D* aHist3d, TString aPreName);
  TH1D* GetSigmaStPOnly(TH3D* aHist3d, TString aPreName);
  TH1D* GetSigmaStMOnly(TH3D* aHist3d, TString aPreName);
  TH1D* GetSigmaSt0Only(TH3D* aHist3d, TString aPreName);

  TH1D* GetPrimaryAndShortDecays(TH3D* aHist3d, TString aPreName);

  //-----------------------------------

  TH1D* GetFullCf();
  TH1D* GetPrimaryOnlyCf();
  TH1D* GetSecondaryOnlyCf();
  TH1D* GetAtLeastOneSecondaryInPairCf();
  TH1D* GetWithoutSigmaStCf();

  TH1D* GetSigmaStOnlyCf();
  TH1D* GetSigmaStPOnlyCf();
  TH1D* GetSigmaStMOnlyCf();
  TH1D* GetSigmaSt0OnlyCf();

  TH1D* GetPrimaryAndShortDecaysCf();

  //inline----------------------------------------
  AnalysisType GetAnalysisType();

  TH1D* GetFullNum();
  TH1D* GetFullDen();

  TH1D* GetPrimaryOnlyNum();
  TH1D* GetPrimaryOnlyDen();

  TH1D* GetSecondaryOnlyNum();
  TH1D* GetSecondaryOnlyDen();

  TH1D* GetAtLeastOneSecondaryInPairNum();
  TH1D* GetAtLeastOneSecondaryInPairDen();

  TH1D* GetWithoutSigmaStNum();
  TH1D* GetWithoutSigmaStDen();

  TH1D* GetSigmaStOnlyNum();
  TH1D* GetSigmaStOnlyDen();

  TH1D* GetSigmaStPOnlyNum();
  TH1D* GetSigmaStPOnlyDen();

  TH1D* GetSigmaStMOnlyNum();
  TH1D* GetSigmaStMOnlyDen();

  TH1D* GetSigmaSt0OnlyNum();
  TH1D* GetSigmaSt0OnlyDen();

  TH1D* GetPrimaryAndShortDecaysNum();
  TH1D* GetPrimaryAndShortDecaysDen();


private:
  AnalysisType fAnalysisType;
  ParticlePDGType fPartType1, fPartType2;
  int fPartIndex1, fPartIndex2;
  double fMinNorm, fMaxNorm;
  int fRebin;

  TH3D *fNum3d, *fDen3d;



#ifdef __ROOT__
  ClassDef(Therm3dCf, 1)
#endif
};

inline AnalysisType Therm3dCf::GetAnalysisType() {return fAnalysisType;}

inline TH1D* Therm3dCf::GetFullNum() {return GetFull(fNum3d, "Num");}
inline TH1D* Therm3dCf::GetFullDen() {return GetFull(fDen3d, "Den");}

inline TH1D* Therm3dCf::GetPrimaryOnlyNum() {return GetPrimaryOnly(fNum3d, "Num");}
inline TH1D* Therm3dCf::GetPrimaryOnlyDen() {return GetPrimaryOnly(fDen3d, "Den");}

inline TH1D* Therm3dCf::GetSecondaryOnlyNum() {return GetSecondaryOnly(fNum3d, "Num");}
inline TH1D* Therm3dCf::GetSecondaryOnlyDen() {return GetSecondaryOnly(fDen3d, "Den");}

inline TH1D* Therm3dCf::GetAtLeastOneSecondaryInPairNum() {return GetAtLeastOneSecondaryInPair(fNum3d, "Num");}
inline TH1D* Therm3dCf::GetAtLeastOneSecondaryInPairDen() {return GetAtLeastOneSecondaryInPair(fDen3d, "Den");}

inline TH1D* Therm3dCf::GetWithoutSigmaStNum() {return GetWithoutSigmaSt(fNum3d, "Num");}
inline TH1D* Therm3dCf::GetWithoutSigmaStDen() {return GetWithoutSigmaSt(fDen3d, "Den");}

inline TH1D* Therm3dCf::GetSigmaStOnlyNum() {return GetSigmaStOnly(fNum3d, "Num");}
inline TH1D* Therm3dCf::GetSigmaStOnlyDen() {return GetSigmaStOnly(fDen3d, "Den");}

inline TH1D* Therm3dCf::GetSigmaStPOnlyNum() {return GetSigmaStPOnly(fNum3d, "Num");}
inline TH1D* Therm3dCf::GetSigmaStPOnlyDen() {return GetSigmaStPOnly(fDen3d, "Den");}

inline TH1D* Therm3dCf::GetSigmaStMOnlyNum() {return GetSigmaStMOnly(fNum3d, "Num");}
inline TH1D* Therm3dCf::GetSigmaStMOnlyDen() {return GetSigmaStMOnly(fDen3d, "Den");}

inline TH1D* Therm3dCf::GetSigmaSt0OnlyNum() {return GetSigmaSt0Only(fNum3d, "Num");}
inline TH1D* Therm3dCf::GetSigmaSt0OnlyDen() {return GetSigmaSt0Only(fDen3d, "Den");}

inline TH1D* Therm3dCf::GetPrimaryAndShortDecaysNum() {return GetPrimaryAndShortDecays(fNum3d, "Num");}
inline TH1D* Therm3dCf::GetPrimaryAndShortDecaysDen() {return GetPrimaryAndShortDecays(fDen3d, "Den");}


#endif
