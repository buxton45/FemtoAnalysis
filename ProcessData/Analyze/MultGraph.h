/*MultGraph.h			*/

#ifndef MULTGRAPH_H
#define MULTGRAPH_H

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




class MultGraph {

public:
  MultGraph(int aNx, int aNy);
  virtual ~MultGraph();

  TAxis* SetupAxis(AxisType aAxisType, TGraphAsymmErrors* fGraph, float fXscale, float fYscale);

  TPad** CanvasPartition(TCanvas *aCanvas,const Int_t Nx = 2,const Int_t Ny = 2,
                         Float_t lMargin = 0.15, Float_t rMargin = 0.05,
                         Float_t bMargin = 0.15, Float_t tMargin = 0.05);

  float** GetScaleFactors(AxisType aAxisType, TPad **fPadArray, int Nx, int Ny);

  TPaveText* SetupTPaveText(TString fText, double fXminOffset, double fYminOffset, float fXscale, float fYscale);
  void DrawInPad(TPad** fPadArray, int Nx, int Ny, TGraphAsymmErrors* ALICEstat, TGraphAsymmErrors* ALICEsys, TPaveText* text)


protected:
  int fNx, fNy;

  float fMarginLeft, fMarginRight, fMarginBottom, fMarginTop;


  TObjArray* fGraphs;
  TPad** tPads;
  float** fXScaleFactors;
  float** fYScaleFactors;



#ifdef __ROOT__
  ClassDef(MultGraph, 1)
#endif
};




#endif
