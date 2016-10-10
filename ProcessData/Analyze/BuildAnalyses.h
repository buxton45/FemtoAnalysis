///////////////////////////////////////////////////////////////////////////
// BuildAnalyses:                                                        //
//                                                                       //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef BUILDANALYSES_H
#define BUILDANALYSES_H

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
#include "TList.h"

#include "TApplication.h"

using std::cout;
using std::endl;
using std::vector;

#include "PartialAnalysis.h"
class PartialAnalysis;

#include "Analysis.h"
class Analysis;

#include "DataAndModel.h"
class DataAndModel;

class BuildAnalyses {

public:



private:


#ifdef __ROOT__
  ClassDef(BuildAnalyses, 1)
#endif
};


#endif
