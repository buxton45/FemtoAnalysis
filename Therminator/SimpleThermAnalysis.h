/* SimpleThermAnalysis.h */

#ifndef SIMPLETHERMANALYSIS_H
#define SIMPLETHERMANALYSIS_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cassert>

#include "TObjArray.h"
#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TSystemDirectory.h"
#include "TSystemFile.h"
#include "TCanvas.h"

#include "Types.h"
#include "PIDMapping.h"

#include "ThermPairAnalysis.h"
class ThermPairAnalysis;

#include "ThermSingleParticleAnalysis.h"
class ThermSingleParticleAnalysis;

#include "ThermFlowCollection.h"
class ThermFlowCollection;

#include "ThermEvent.h"
class ThermEvent;


using namespace std;

class SimpleThermAnalysis {

public:
  SimpleThermAnalysis();
  virtual ~SimpleThermAnalysis();

  void SetUseMixedEventsForTransforms(bool aUse);
  void SetBuildUniqueParents(bool aBuild);
  void SaveAll();

  vector<ThermEvent> ExtractEventsFromRootFile(TString aFileLocation);
  void ProcessAllInDirectory(TSystemDirectory* aEventsDirectory);
  void ProcessAll();
  void ProcessEventByEvent(vector<ThermEvent> &aEventsCollection);

  void SetBuildPairFractions(bool aBuild);
  void SetBuildTransformMatrices(bool aBuild);
  void SetBuildCorrelationFunctions(bool aBuild, bool aBuild3dHists=false);
  void SetBuildMixedEventNumerators(bool aBuild);
  void SetUnitWeightCfNums(bool aSet);
  void SetWeightCfsWithParentInteraction(bool aSet);
  void SetOnlyWeightLongDecayParents(bool aSet);
  void SetDrawRStarFromGaussian(bool aSet);
  void SetBuildSingleParticleAnalyses(bool aBuild);

  void SetMaxPrimaryDecayLength(double aMax);

  //-- inline
  void SetNEventsToMix(int aNEventsToMix);

  void SetEventsDirectory(TString aDirectory);
  void SetPairFractionsSaveName(TString aSaveName);
  void SetTransformMatricesSaveName(TString aSaveName);
  void SetCorrelationFunctionsSaveName(TString aSaveName);
  void SetSingleParticlesSaveName(TString aSaveName);
  void SetFlowAnalysisSaveName(TString aSaveName);

  void SetCheckCoECoM(bool aCheck=true);
  void SetRotateEventsByRandomAzimuthalAngles(bool aRotate=true);
  void SetPerformFlowAnalysis(bool aPerform=true);
  void SetBuildArtificialV3Signal(bool aBuild=true, int aV3InclusionProb1=25);  //NOTE: This kills v2 signal and creates v3 signal

  void SetBuildOtherPairs(bool aSet);
private:
  int fNFiles;
  int fNEvents;
  TString fEventsDirectory;
  TString fPairFractionsSaveName;
  TString fTransformMatricesSaveName;
  TString fCorrelationFunctionsSaveName;
  TString fSingleParticlesSaveName;

  vector<TString> fFileNameCollection;
  vector<ThermEvent> fEventsCollection;

  bool fMixEvents;
  bool fMixEventsForTransforms;
  unsigned int fNEventsToMix;
  vector<ThermEvent> fMixingEventsCollection;

  double fKStarMin, fKStarMax;
  int fNBinsKStar;

  double fMaxPrimaryDecayLength;

  bool fBuildUniqueParents;

  bool fBuildPairFractions;
  bool fBuildTransformMatrices;
  bool fBuildCorrelationFunctions;
  bool fBuild3dHists;
  bool fBuildMixedEventNumerators;
  bool fUnitWeightCfNums;
  bool fWeightCfsWithParentInteraction;
  bool fOnlyWeightLongDecayParents;
  bool fDrawRStarFromGaussian;
  bool fBuildSingleParticleAnalyses;

  ThermPairAnalysis *fAnalysisLamKchP, *fAnalysisALamKchM;
  ThermPairAnalysis *fAnalysisLamKchM, *fAnalysisALamKchP;
  ThermPairAnalysis *fAnalysisLamK0, *fAnalysisALamK0;

  ThermPairAnalysis *fAnalysisKchPKchP;
  ThermPairAnalysis *fAnalysisK0K0;
  ThermPairAnalysis *fAnalysisLamLam;

  ThermSingleParticleAnalysis *fSPAnalysisLam, *fSPAnalysisALam;
  ThermSingleParticleAnalysis *fSPAnalysisKchP, *fSPAnalysisKchM;
  ThermSingleParticleAnalysis *fSPAnalysisProt, *fSPAnalysisAProt;
  ThermSingleParticleAnalysis *fSPAnalysisK0;

  ThermFlowCollection *fFlowCollection;

  bool fCheckCoECoM;
  bool fRotateEventsByRandAzAngles;
  bool fPerformFlowAnalysis;
  std::pair <int, int> fArtificialV3Info;

  bool fBuildOtherPairs;

#ifdef __ROOT__
  ClassDef(SimpleThermAnalysis, 1)
#endif
};

inline void SimpleThermAnalysis::SetNEventsToMix(int aNEventsToMix) {fNEventsToMix = aNEventsToMix;}

inline void SimpleThermAnalysis::SetEventsDirectory(TString aDirectory) {fEventsDirectory = aDirectory;}
inline void SimpleThermAnalysis::SetPairFractionsSaveName(TString aSaveName) {fPairFractionsSaveName = aSaveName;}
inline void SimpleThermAnalysis::SetTransformMatricesSaveName(TString aSaveName) {fTransformMatricesSaveName = aSaveName;}
inline void SimpleThermAnalysis::SetCorrelationFunctionsSaveName(TString aSaveName) {fCorrelationFunctionsSaveName = aSaveName;}
inline void SimpleThermAnalysis::SetSingleParticlesSaveName(TString aSaveName) {fSingleParticlesSaveName = aSaveName;}
inline void SimpleThermAnalysis::SetFlowAnalysisSaveName(TString aSaveName) {fFlowCollection->SetSaveFileName(aSaveName);}

inline void SimpleThermAnalysis::SetCheckCoECoM(bool aCheck) {fCheckCoECoM = aCheck;}
inline void SimpleThermAnalysis::SetRotateEventsByRandomAzimuthalAngles(bool aRotate) {fRotateEventsByRandAzAngles = aRotate;}
inline void SimpleThermAnalysis::SetPerformFlowAnalysis(bool aPerform) {fPerformFlowAnalysis = aPerform;}
inline void SimpleThermAnalysis::SetBuildArtificialV3Signal(bool aBuild, int aV3InclusionProb1) {fArtificialV3Info=std::make_pair((int)aBuild, aV3InclusionProb1);}

inline void SimpleThermAnalysis::SetBuildOtherPairs(bool aSet) {fBuildOtherPairs = aSet;}

#endif
