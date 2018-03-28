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

#include "ThermEvent.h"
class ThermEvent;


using namespace std;

class SimpleThermAnalysis {

public:
  SimpleThermAnalysis();
  virtual ~SimpleThermAnalysis();

  void SetUseMixedEvents(bool aUse);
  void SetBuildUniqueParents(bool aBuild);
  void SaveAll();

  vector<ThermEvent> ExtractEventsFromRootFile(TString aFileLocation);
  void ProcessAllInDirectory(TSystemDirectory* aEventsDirectory);
  void ProcessAll();
  void ProcessEventByEvent(vector<ThermEvent> &aEventsCollection);

  void SetBuildPairFractions(bool aBuild);
  void SetBuildTransformMatrices(bool aBuild);
  void SetBuildCorrelationFunctions(bool aBuild);
  void SetBuildMixedEventNumerators(bool aBuild);
  void SetUnitWeightCfNums(bool aSet);
  void SetWeightCfsWithParentInteraction(bool aSet);
  void SetOnlyWeightLongDecayParents(bool aSet);
  void SetBuildSingleParticleAnalyses(bool aBuild);

  //-- inline
  void SetNEventsToMix(int aNEventsToMix);

  void SetEventsDirectory(TString aDirectory);
  void SetPairFractionsSaveName(TString aSaveName);
  void SetTransformMatricesSaveName(TString aSaveName);
  void SetCorrelationFunctionsSaveName(TString aSaveName);
  void SetSingleParticlesSaveName(TString aSaveName);

  void SetMaxPrimaryDecayLength(double aMax);

  void SetCheckCoECoM(bool aCheck=true);
  void SetRotateEventsByRandomAzimuthalAngles(bool aRotate=true);
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
  unsigned int fNEventsToMix;
  vector<ThermEvent> fMixingEventsCollection;

  double fKStarMin, fKStarMax;
  int fNBinsKStar;

  double fMaxPrimaryDecayLength;

  bool fBuildUniqueParents;

  bool fBuildPairFractions;
  bool fBuildTransformMatrices;
  bool fBuildCorrelationFunctions;
  bool fBuildMixedEventNumerators;
  bool fUnitWeightCfNums;
  bool fWeightCfsWithParentInteraction;
  bool fOnlyWeightLongDecayParents;
  bool fBuildSingleParticleAnalyses;

  ThermPairAnalysis *fAnalysisLamKchP, *fAnalysisALamKchM;
  ThermPairAnalysis *fAnalysisLamKchM, *fAnalysisALamKchP;
  ThermPairAnalysis *fAnalysisLamK0, *fAnalysisALamK0;

  ThermSingleParticleAnalysis *fSPAnalysisLam, *fSPAnalysisALam;
  ThermSingleParticleAnalysis *fSPAnalysisKchP, *fSPAnalysisKchM;
  ThermSingleParticleAnalysis *fSPAnalysisProt, *fSPAnalysisAProt;
  ThermSingleParticleAnalysis *fSPAnalysisK0;

  bool fCheckCoECoM;
  bool fRotateEventsByRandAzAngles;

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

inline void SimpleThermAnalysis::SetMaxPrimaryDecayLength(double aMax) {fMaxPrimaryDecayLength = aMax;}

inline void SimpleThermAnalysis::SetCheckCoECoM(bool aCheck) {fCheckCoECoM = aCheck;}
inline void SimpleThermAnalysis::SetRotateEventsByRandomAzimuthalAngles(bool aRotate) {fRotateEventsByRandAzAngles = aRotate;}

#endif
