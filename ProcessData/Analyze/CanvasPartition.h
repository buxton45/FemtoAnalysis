/*CanvasPartition.h			*/

#ifndef CANVASPARTITION_H
#define CANVASPARTITION_H

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

class CanvasPartition {

public:
  CanvasPartition(TString aCanvasName, int aNx, int aNy, double aXRangeLow, double aXRangeHigh, double aYRangeLow, double aYRangeHigh, float aMarginLeft, float aMarginRight, float aMarginBottom, float aMarginTop);
  virtual ~CanvasPartition();

  //Since I am using templates, the following functions need to be defined in this header, i.e. below
  template<typename T>
  void AddGraph(int aNx, int aNy, T* aGraph, TString tPadLegendName, int aMarkerStyle=20, int aMarkerColor=1, double aMarkerSize=0.75, TString aDrawOption="psames");

  template<typename T>
  void SetupAxis(AxisType aAxisType, T* aGraph, float aXscale, float aYscale, float aLabelSize=0.15/*0.20*/, float aLabelOffSet=0.005);

  td2dTPadVec BuildPartition(TCanvas *aCanvas,const Int_t Nx = 2,const Int_t Ny = 2,
                         Float_t lMargin = 0.15, Float_t rMargin = 0.05,
                         Float_t bMargin = 0.15, Float_t tMargin = 0.05);

  float** GetScaleFactors(AxisType aAxisType, td2dTPadVec &fPadArray, int Nx, int Ny);

  TPaveText* SetupTPaveText(TString aText, int aNx, int aNy, double aTextXmin=0.75, double aTextYmin=0.75, double aTextWidth=0.15, double aTextHeight=0.10, double aTextFont=63, double aTextSize=15);
  void SetupOptStat(int aNx, int aNy, double aStatX, double aStatY, double aStatW, double aStatH);
  void AddPadPaveText(TPaveText* aText, int aNx, int aNy);

  void DrawInPad(int aNx, int aNy);
  void DrawAll();

  void DrawXaxisTitle(TString aTitle, int aTextFont=43, int aTextSize=25, double aXLow=0.315, double aYLow=0.03);
  void DrawYaxisTitle(TString aTitle, int aTextFont=43, int aTextSize=25, double aXLow=0.05, double aYLow=0.35);

  TCanvas* GetCanvas();
  TPad* GetPad(int aNx, int aNy);
  void SetDrawUnityLine(bool aDraw);
  void SetDrawOptStat(bool aDraw);

  double GetXScaleFactor(int aNx, int aNy);
  double GetYScaleFactor(int aNx, int aNy);

protected:
  bool fDrawUnityLine;
  bool fDrawOptStat;
  int fNx, fNy;
  double fXaxisRangeLow, fXaxisRangeHigh;
  double fYaxisRangeLow, fYaxisRangeHigh;

  float fMarginLeft, fMarginRight, fMarginBottom, fMarginTop;

  TCanvas* fCanvas;

  TObjArray* fGraphs;
  vector<vector<TString> > fGraphsDrawOptions;
  TObjArray* fPadPaveTexts;

  td2dTPadVec fPadArray;
  float** fXScaleFactors;
  float** fYScaleFactors;



#ifdef __ROOT__
  ClassDef(CanvasPartition, 1)
#endif
};

inline TCanvas* CanvasPartition::GetCanvas() {fCanvas->Update(); return fCanvas;}
inline TPad* CanvasPartition::GetPad(int aNx, int aNy) {return fPadArray[aNx][aNy];}
inline void CanvasPartition::SetDrawUnityLine(bool aDraw) {fDrawUnityLine = aDraw;}
inline void CanvasPartition::SetDrawOptStat(bool aDraw) {fDrawOptStat = aDraw;}

inline double CanvasPartition::GetXScaleFactor(int aNx, int aNy) {return fXScaleFactors[aNx][aNy];}
inline double CanvasPartition::GetYScaleFactor(int aNx, int aNy) {return fYScaleFactors[aNx][aNy];}

//________________________________________________________________________________________________________________
template<typename T>
inline void CanvasPartition::AddGraph(int aNx, int aNy, T* aGraph, TString tPadLegendName, int aMarkerStyle, int aMarkerColor, double aMarkerSize, TString aDrawOption)
{
  int tPosition = aNx + aNy*fNx;

  //TODO: For some reason, when SetupAxis is used with TF1* object, it causes an extra 
  //border to be drawn in the TCanvas.  For now, disable for TF1
  if(!(typeid(aGraph) == typeid(TF1*)))
  {
    SetupAxis(kXaxis,aGraph,fXScaleFactors[aNx][aNy],fYScaleFactors[aNx][aNy]);
    SetupAxis(kYaxis,aGraph,fXScaleFactors[aNx][aNy],fYScaleFactors[aNx][aNy]);
  }

  aGraph->SetMarkerStyle(aMarkerStyle);
  aGraph->SetMarkerColor(aMarkerColor);
  aGraph->SetLineColor(aMarkerColor);
  aGraph->SetMarkerSize(aMarkerSize);

  if(!tPadLegendName.IsNull()) AddPadPaveText(SetupTPaveText(tPadLegendName,aNx,aNy),aNx,aNy);
  ((TObjArray*)fGraphs->At(tPosition))->Add(aGraph);
  fGraphsDrawOptions[tPosition].push_back(aDrawOption);
}


//________________________________________________________________________________________________________________
template<typename T>
inline void CanvasPartition::SetupAxis(AxisType aAxisType, T* aGraph, float aXscale, float aYscale, float aLabelSize, float aLabelOffSet)
{
  TAxis* tReturnAxis;

  if(aAxisType == kXaxis) tReturnAxis = aGraph->GetXaxis();
  else if(aAxisType == kYaxis) tReturnAxis = aGraph->GetYaxis();
  else 
  {
    cout << "ERROR: CanvasPartition::SetupAxis: Invalid aAxisType = " << aAxisType << endl;
    assert(0);
  }

  tReturnAxis->SetTitle("");
  tReturnAxis->SetTitleSize(0.);
  tReturnAxis->SetTitleOffset(0.);

  tReturnAxis->SetLabelFont(43);
  tReturnAxis->SetLabelSize(100*aLabelSize);  //TODO not sure why this needs to be set to 100x the normal input
  tReturnAxis->SetLabelOffset(aLabelOffSet);

  if(aAxisType == kXaxis)
  {
    tReturnAxis->SetNdivisions(510);
    tReturnAxis->SetTickLength(0.04*aYscale/aXscale);
    tReturnAxis->SetRangeUser(fXaxisRangeLow,fXaxisRangeHigh);
  }
  else
  {
    tReturnAxis->SetNdivisions(505);
    tReturnAxis->SetTickLength(0.04*aXscale/aYscale);
    tReturnAxis->SetRangeUser(fYaxisRangeLow,fYaxisRangeHigh);
  }

}

#endif
