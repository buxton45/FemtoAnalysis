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

private:
  int fNFiles;
  int fNEvents;
  TString fEventsDirectory;
  TString fPairFractionsSaveName;
  TString fTransformMatricesSaveName;

  vector<TString> fFileNameCollection;
  vector<ThermEvent> fEventsCollection;

  bool fMixEvents;
  unsigned int fNEventsToMix;
  vector<ThermEvent> fMixingEventsCollection;

  double fKStarMin, fKStarMax;
  int fNBinsKStar;

  bool fBuildUniqueParents;

  ThermPairAnalysis *fAnalysisLamKchP, *fAnalysisALamKchM;
  ThermPairAnalysis *fAnalysisLamKchM, *fAnalysisALamKchP;
  ThermPairAnalysis *fAnalysisLamK0, *fAnalysisALamK0;


#ifdef __ROOT__
  ClassDef(SimpleThermAnalysis, 1)
#endif
};



inline void SimpleThermAnalysis::SetEventsDirectory(TString aDirectory) {fEventsDirectory = aDirectory;}
inline void SimpleThermAnalysis::SetPairFractionsSaveName(TString aSaveName) {fPairFractionsSaveName = aSaveName;}
inline void SimpleThermAnalysis::SetTransformMatricesSaveName(TString aSaveName) {fTransformMatricesSaveName = aSaveName;}


#endif
