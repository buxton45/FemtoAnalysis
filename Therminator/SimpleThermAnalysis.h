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
  void ProcessAll();
  void ProcessEventByEvent(vector<ThermEvent> &aEventsCollection);

  //-- inline


  void SetEventsDirectory(TString aDirectory);
  void SetPairFractionsSaveName(TString aSaveName);
  void SetTransformMatricesSaveName(TString aSaveName);
  void SetSingleParticlesSaveName(TString aSaveName);

  void SetMaxPrimaryDecayLength(double aMax);

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

  ThermPairAnalysis *fAnalysisLamKchP, *fAnalysisALamKchM;
  ThermPairAnalysis *fAnalysisLamKchM, *fAnalysisALamKchP;
  ThermPairAnalysis *fAnalysisLamK0, *fAnalysisALamK0;

  ThermSingleParticleAnalysis *fSPAnalysisLam, *fSPAnalysisALam;
  ThermSingleParticleAnalysis *fSPAnalysisKchP, *fSPAnalysisKchM;
  ThermSingleParticleAnalysis *fSPAnalysisProt, *fSPAnalysisAProt;
  ThermSingleParticleAnalysis *fSPAnalysisK0;


#ifdef __ROOT__
  ClassDef(SimpleThermAnalysis, 1)
#endif
};



inline void SimpleThermAnalysis::SetEventsDirectory(TString aDirectory) {fEventsDirectory = aDirectory;}
inline void SimpleThermAnalysis::SetPairFractionsSaveName(TString aSaveName) {fPairFractionsSaveName = aSaveName;}
inline void SimpleThermAnalysis::SetTransformMatricesSaveName(TString aSaveName) {fTransformMatricesSaveName = aSaveName;}
inline void SimpleThermAnalysis::SetSingleParticlesSaveName(TString aSaveName) {fSingleParticlesSaveName = aSaveName;}

inline void SimpleThermAnalysis::SetMaxPrimaryDecayLength(double aMax) {fMaxPrimaryDecayLength = aMax;}

#endif
