///////////////////////////////////////////////////////////////////////////
// PlotPartners:                                                         //
//                                                                       //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef PLOTPARTNERS_H
#define PLOTPARTNERS_H

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

#include "Analysis.h"
class Analysis;

class PlotPartners {

public:

  PlotPartners(vector<Analysis*> aAnalysisCollection, PartnerAnalysisType aPartnerAnalysisType, CentralityType aCentralityType);
  PlotPartners(TString aFileLocationBase, PartnerAnalysisType aPartnerAnalysisType, CentralityType aCentralityType, int aNAnalysis);

  virtual ~PlotPartners();




  //inline


private:


  vector<Analysis*> fAnalysisCollection;
  int fNAnalysis;









#ifdef __ROOT__
  ClassDef(PlotPartners, 1)
#endif
};



#endif

