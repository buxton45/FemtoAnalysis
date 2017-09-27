#include "FitSharedAnalyses.h"
#include "CoulombFitter.h"
#include "TLegend.h"
#include "CanvasPartition.h"


#include "TColor.h"
#include <TStyle.h>
#include "TPaveText.h"
#include <TLatex.h>
#include "TGraphAsymmErrors.h"
#include "TFile.h"
/*
CoulombFitter *myFitter = NULL;

//______________________________________________________________________________
void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
  myFitter->CalculateChi2PML(npar,f,par);
//  myFitter->CalculateChi2PMLwMomResCorrection(npar,f,par);
//  myFitter->CalculateChi2(npar,f,par);
//  myFitter->CalculateFakeChi2(npar,f,par);
}
*/



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

  AnalysisType tAnType = kResASigStPKchP;

  int tNbinsKStar = 100;
  double tKStarMin = 0.;
  double tKStarMax = 1.;

  double tLambda = 1.0;

/*
  int tNbinsRStar = 200;
  double tRStarMin = 0.;
  double tRStarMax = 20.;
*/
/*
  int tNbinsRStar = 48;
  double tRStarMin = 1.;
  double tRStarMax = 13.;
*/
  int tNbinsRStar = 100;
  double tRStarMin = 0.;
  double tRStarMax = 25.;

  double tNorm = 1.;

  TH2D* t2dCoulombOnlyInterpCfs = new TH2D(TString::Format("t2dCoulombOnlyInterpCfs_%s", cAnalysisBaseTags[tAnType]),
                                           TString::Format("t2dCoulombOnlyInterpCfs_%s", cAnalysisBaseTags[tAnType]), 
                                           tNbinsKStar, tKStarMin, tKStarMax,
                                           tNbinsRStar, tRStarMin, tRStarMax);

  CoulombFitter* tFitter = new CoulombFitter(1.0);

  TString tFileLocationInterpHistos = "InterpHists";
    tFileLocationInterpHistos += TString::Format("_%s", cAnalysisBaseTags[tAnType]);
  TString tFileLocationLednickyHFile = "LednickyHFunction";
    tFileLocationLednickyHFile += TString::Format("_%s", cAnalysisBaseTags[tAnType]);
  TString tSaveName = TString::Format("2dCoulombOnlyInterpCfs_%s.root", cAnalysisBaseTags[tAnType]);

  tFitter->LoadInterpHistFile(tFileLocationInterpHistos, tFileLocationLednickyHFile);

  tFitter->SetUseRandomKStarVectors(true);
  tFitter->SetUseStaticPairs(true,50000);
  tFitter->SetIncludeSingletAndTriplet(false);
/*
  tFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(fcn);
  myFitter = tFitter;
*/
  double tRadius = -1.;
  double tRadiusBinWidth = (tRStarMax-tRStarMin)/tNbinsRStar;
  TH1* tCoulombOnlyHistSample;
cout << "tRadiusBinWidth = " << tRadiusBinWidth << endl;
  for(int iR=0; iR<tNbinsRStar; iR++)
  {
    tRadius = tRStarMin + (iR+0.5)*tRadiusBinWidth;
cout << "iR = " << iR << " and tRadius = " << tRadius << endl;
    tCoulombOnlyHistSample = tFitter->CreateFitHistogramSampleComplete(TString::Format("tCoulombOnlyHistSample_%d", iR), tAnType, 
                                                                       tNbinsKStar, tKStarMin, tKStarMax, tLambda, tRadius, 
                                                                       0., 0., 0., 0., 0., 0., tNorm);
    assert(tCoulombOnlyHistSample->GetNbinsX() == t2dCoulombOnlyInterpCfs->GetNbinsX());
    assert(tCoulombOnlyHistSample->GetXaxis()->GetBinWidth(1) == t2dCoulombOnlyInterpCfs->GetXaxis()->GetBinWidth(1));

    int tBinR = t2dCoulombOnlyInterpCfs->GetYaxis()->FindBin(tRadius);
    assert(tBinR == (iR+1));

    for(int iK=1; iK<=t2dCoulombOnlyInterpCfs->GetNbinsX(); iK++)
    {
      t2dCoulombOnlyInterpCfs->SetBinContent(iK, tBinR, tCoulombOnlyHistSample->GetBinContent(iK));
      t2dCoulombOnlyInterpCfs->SetBinError(iK, tBinR, tCoulombOnlyHistSample->GetBinError(iK));
    }
  }

  TCanvas* tTestCan = new TCanvas("tTestCan", "tTestCan");
  tTestCan->cd();
  t2dCoulombOnlyInterpCfs->Draw("colz"); 



  TFile* tSaveFile = new TFile(tSaveName, "RECREATE");
  t2dCoulombOnlyInterpCfs->Write();
  tSaveFile->Close();

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





