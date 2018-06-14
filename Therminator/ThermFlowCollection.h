/* ThermFlowCollection.h */
//A few ThermFlowAnalysis objects collected together

#ifndef THERMFLOWCOLLECTION_H
#define THERMFLOWCOLLECTION_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cassert>
#include <complex>

#include "TGraph.h"
#include "TGraphErrors.h"
#include "TCanvas.h"

#include "ThermEvent.h"
class ThermEvent;

#include "ThermFlowAnalysis.h"
class ThermFlowAnalysis;

using namespace std;

class ThermFlowCollection {

public:
  ThermFlowCollection(int aNpTBins=60, double apTBinSize=0.1);
  virtual ~ThermFlowCollection();

  void BuildVnEPIngredients(ThermEvent &aEvent, double aHarmonic);
  void BuildVnEPIngredients(ThermEvent &aEvent);
  void DrawAllFlowHarmonics();
  void SaveAllGraphs();
  void Finalize();

  //--inline
  void SetSaveFileName(TString aName);

private:
  ThermFlowAnalysis* fAnUnIdent;
  ThermFlowAnalysis* fAnKch;
  ThermFlowAnalysis* fAnK0s;
  ThermFlowAnalysis* fAnLam;

  TString fSaveFileName;

#ifdef __ROOT__
  ClassDef(ThermFlowCollection, 1)
#endif
};

inline void ThermFlowCollection::SetSaveFileName(TString aName) {fSaveFileName = aName;}

#endif















