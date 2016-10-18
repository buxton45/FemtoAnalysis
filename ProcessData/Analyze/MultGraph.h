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

#include "TROOT.h"
#include "TGraph.h"
#include "TGraphAsymmErrors.h"
#include "TLatex.h"

using std::cout;
using std::cin;
using std::endl;
using std::vector;

#include "Types.h"

typedef vector<TPad*> td1dTPadVec;
typedef vector<vector<TPad*> > td2dTPadVec;

class MultGraph {

public:
  MultGraph(TString aCanvasName, int aNx, int aNy, double aXRangeLow, double aXRangeHigh, double aYRangeLow, double aYRangeHigh, float aMarginLeft, float aMarginRight, float aMarginBottom, float aMarginTop);
  virtual ~MultGraph();

  void AddGraph(int aNx, int aNy, TH1* aGraph, TString tPadLegendName, int aMarkerStyle=20, int aMarkerColor=1, double aMarkerSize=0.75);

  void SetupAxis(AxisType aAxisType, TH1* fGraph, float fXscale, float fYscale);

  td2dTPadVec CanvasPartition(TCanvas *aCanvas,const Int_t Nx = 2,const Int_t Ny = 2,
                         Float_t lMargin = 0.15, Float_t rMargin = 0.05,
                         Float_t bMargin = 0.15, Float_t tMargin = 0.05);

  float** GetScaleFactors(AxisType aAxisType, td2dTPadVec &fPadArray, int Nx, int Ny);

  TPaveText* SetupTPaveText(TString aText, int aNx, int aNy, double aTextXmin=0.75, double aTextYmin=0.75, double aTextWidth=0.15, double aTextHeight=0.10, double aTextFont=63, double aTextSize=15);
  void DrawInPad(int aNx, int aNy);
  void DrawAll();

  void DrawXaxisTitle(TString aTitle, int aTextFont=43, int aTextSize=25, double aXLow=0.315, double aYLow=0.03);
  void DrawYaxisTitle(TString aTitle, int aTextFont=43, int aTextSize=25, double aXLow=0.05, double aYLow=0.35);

  TCanvas* GetCanvas();
  TPad* GetPad(int aNx, int aNy);
  void SetDrawUnityLine(bool aDraw);

protected:
  bool fDrawUnityLine;
  int fNx, fNy;
  double fXaxisRangeLow, fXaxisRangeHigh;
  double fYaxisRangeLow, fYaxisRangeHigh;

  float fMarginLeft, fMarginRight, fMarginBottom, fMarginTop;

  TCanvas* fCanvas;

  TObjArray* fGraphs;
  vector<TPaveText*> fGraphsPadNames;

  td2dTPadVec fPadArray;
  float** fXScaleFactors;
  float** fYScaleFactors;



#ifdef __ROOT__
  ClassDef(MultGraph, 1)
#endif
};

inline TCanvas* MultGraph::GetCanvas() {return fCanvas;}
inline TPad* MultGraph::GetPad(int aNx, int aNy) {return fPadArray[aNx][aNy];}
inline void MultGraph::SetDrawUnityLine(bool aDraw) {fDrawUnityLine = aDraw;}

#endif
