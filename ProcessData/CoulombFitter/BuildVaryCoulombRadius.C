#include "FitSharedAnalyses.h"
#include "CoulombFitter.h"
#include "TLegend.h"
#include "CanvasPartition.h"

CoulombFitter *myFitter = NULL;

//______________________________________________________________________________
void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
  myFitter->CalculateChi2PML(npar,f,par);
//  myFitter->CalculateChi2PMLwMomResCorrection(npar,f,par);
//  myFitter->CalculateChi2(npar,f,par);
//  myFitter->CalculateFakeChi2(npar,f,par);
}


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


int main(int argc, char **argv)
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  ChronoTimer tFullTimer(kMin);
  tFullTimer.Start();
//-----------------------------------------------------------------------------

  //!!!!!!!!!!!!!!!! NOTE:  must set myFitter = to whichever LednickyFitter object I want to use

//-----------------------------------------------------------------------------
//Be sure to set the following...

  TString tFileLocationBase = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cXicKch_20170406/Results_cXicKch_20170406";
  bool bSaveImage = false;

  AnalysisType tAnType;


  tAnType = kXiKchP;
  //tAnType = kAXiKchM;

  //tAnType = kXiKchM;
  //tAnType = kAXiKchP;

  int tNbinsK = 15;
  double tKMin = 0.;
  double tKMax = 0.15;



  CoulombFitter* tFitter = new CoulombFitter();
    tFitter->SetIncludeSingletAndTriplet(true);
    tFitter->SetApplyMomResCorrection(false);


  TString tFileLocationInterpHistos;
  if(tAnType==kAXiKchP || tAnType==kXiKchM) tFileLocationInterpHistos = "InterpHistsRepulsive";
  else if(tAnType==kXiKchP || tAnType==kAXiKchM) tFileLocationInterpHistos = "InterpHistsAttractive";
  tFitter->LoadInterpHistFile(tFileLocationInterpHistos);

  //-------------------------------------------

  tFitter->SetUseRandomKStarVectors(true);
  tFitter->SetUseStaticPairs(true,16384);

  //-------------------------------------------
  myFitter = tFitter;


//_______________________________________________________________________________________________________________________


  double tLambda, tRadius, tNorm;
  tLambda = 0.385577;
  tRadius = 1.;
  tNorm = 1.;

  double tXLow=0, tXHigh=0, tYLow=0, tYHigh=0;
  tXLow = 0.0;
  tXHigh = 0.15;
  if(tAnType==kXiKchP || tAnType==kAXiKchM)
  {
    tYLow = 0.94;
    tYHigh = 1.7;
  }
  else if(tAnType==kXiKchM || tAnType==kAXiKchP)
  {
    tYLow = 0.38;
    tYHigh = 1.04;
  }
  else assert(0);

  TH1* tCoulombOnlyHist = tFitter->CreateFitHistogramSampleComplete("tCoulombOnlyHist", tAnType, tNbinsK, tKMin, tKMax, tLambda, tRadius, 0., 0., 0., 0., 0., 0., tNorm);
    tCoulombOnlyHist->SetDirectory(0);

  tCoulombOnlyHist->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  tCoulombOnlyHist->GetXaxis()->SetRangeUser(tXLow,tXHigh);
  tCoulombOnlyHist->GetXaxis()->SetTitleSize(0.055);
  tCoulombOnlyHist->GetXaxis()->SetTitleOffset(0.8);

  tCoulombOnlyHist->GetYaxis()->SetTitle("#it{C}(#it{k}*)");
  tCoulombOnlyHist->GetYaxis()->SetRangeUser(tYLow,tYHigh);
  tCoulombOnlyHist->GetYaxis()->SetTitleSize(0.0575);
  tCoulombOnlyHist->GetYaxis()->SetTitleOffset(0.8);

  tCoulombOnlyHist->SetMarkerStyle(22);
  tCoulombOnlyHist->SetMarkerColor(1);
  tCoulombOnlyHist->SetLineColor(1);
  tCoulombOnlyHist->SetLineStyle(1);



  TString tCanvasName = TString("canCoulombOnly_");
  if(tAnType==kXiKchP) tCanvasName += TString("XiKchP");
  else if(tAnType==kAXiKchM) tCanvasName += TString("AXiKchM");
  else if(tAnType==kXiKchM) tCanvasName += TString("XiKchM");
  else if(tAnType==kAXiKchP) tCanvasName += TString("AXiKchP");
  else assert(0);

  TCanvas* tCan = new TCanvas(tCanvasName,tCanvasName);
  tCan->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  tCoulombOnlyHist->DrawCopy("l");
  tCan->Update();
  for(int i=1; i<=10; i++)
  {
    tRadius = tRadius + 0.5;
cout << "tRadius = " << tRadius << endl;
    tCoulombOnlyHist = tFitter->CreateFitHistogramSampleComplete("tCoulombOnlyHist", tAnType, tNbinsK, tKMin, tKMax, tLambda, tRadius, 0., 0., 0., 0., 0., 0., tNorm);
    tCoulombOnlyHist->DrawCopy("lsame");
    tCan->Update();
  }


  delete tFitter;



//-------------------------------------------------------------------------------
  tFullTimer.Stop();
  cout << "Finished program: ";
  tFullTimer.PrintInterval();

  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
  return 0;
}
