/*HistInfoPrinter.h			*/

#ifndef HISTINFOPRINTER_H
#define HISTINFOPRINTER_H

#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "TF1.h"
#include "TH1F.h"
#include "TH2F.h"
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
#include "TColor.h"
#include "TLatex.h"

using std::cout;
using std::cin;
using std::endl;
using std::vector;

class HistInfoPrinter {

public:
  HistInfoPrinter();
  virtual ~HistInfoPrinter();

  static bool AreApproxEqual(double aVal1, double aVal2, double aPrecision=0.000001);
  
  static void PrintHistInfo(TH1* aHist, FILE* aOutput, double aXAxisLow=0., double aXAxisHigh=0.);
  static void PrintHistInfoYAML(TH1* aHist, FILE* aOutput, double aXAxisLow=0., double aXAxisHigh=0.);  
  
  static void PrintHistInfowStatAndSyst(TH1* aHistStat, TH1* aHistSyst, FILE* aOutput, double aXAxisLow=0., double aXAxisHigh=0.);
  static void PrintHistInfowStatAndSystYAML(TH1* aHistStat, TH1* aHistSyst, FILE* aOutput, double aXAxisLow=0., double aXAxisHigh=0.);    


protected:



#ifdef __ROOT__
  ClassDef(HistInfoPrinter, 1)
#endif
};


#endif
