#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "TApplication.h"
#include "TStyle.h"
#include "TPaveText.h"
#include "TLegend.h"
#include "TF1.h"
#include "TLatex.h"
#include "TH3.h"

#include "ThermCf.h"
class ThermCf;

//________________________________________________________________________________________________________________
TH3* GetThermHist3d(TString aFileLocation, TString aHistName)
{
  TFile *tFile = TFile::Open(aFileLocation);

  TH3 *tReturnHist = (TH3*)tFile->Get(aHistName);
  TH3 *tReturnHistClone = (TH3*)tReturnHist->Clone();
  tReturnHistClone->SetDirectory(0);

  tFile->Close();
  delete tFile;

  return tReturnHistClone;
}


//________________________________________________________________________________________________________________
TH1D* ProjectOut1dHist(TH3* a3dHist, double amTLow1, double amTHigh1, double amTLow2, double amTHigh2)
{
  TString tNewName = TString::Format("%s_Project_%0.2fto%0.2f_%0.2fto%0.2f", 
                                     a3dHist->GetName(), 
                                     amTLow1, amTHigh1, 
                                     amTLow2, amTHigh2);

  TH1D* tReturnHist = a3dHist->ProjectionZ(tNewName,
                                           a3dHist->GetXaxis()->FindBin(amTLow1), a3dHist->GetXaxis()->FindBin(amTHigh1), 
                                           a3dHist->GetYaxis()->FindBin(amTLow2), a3dHist->GetYaxis()->FindBin(amTHigh2));

  return tReturnHist;
}


//________________________________________________________________________________________________________________
double FitFunctionGaussian(double *x, double *par)
{
  //4 parameters
//  return par[0]*exp(-0.5*(pow((x[0]-par[1])/par[2],2.0))) + par[3];
  return par[0]*exp(-0.5*(pow((x[0]-par[1])/par[2],2.0)))*pow(x[0],2) + par[3];  //From Adam's paper "Non-id particle femto at 200 GeV in hydro..."
}


//________________________________________________________________________________________________________________
TF1* FitHistogram(TH1* aHist, double aMinFit=0., double aMaxFit=5.)
{
  TString tFitName = TString::Format("%s_Fit", aHist->GetName());
  TF1* tReturnFunction = new TF1(tFitName, FitFunctionGaussian, aMinFit, aMaxFit, 4);

  double tMaxVal = aHist->GetMaximum();
  double tMaxPos = aHist->GetBinCenter(aHist->GetMaximumBin());


  tReturnFunction->SetParameter(0, tMaxVal);

//  tReturnFunction->SetParameter(1, tMaxPos);
  tReturnFunction->FixParameter(1, 0.);

  tReturnFunction->SetParameter(2, 5.);

  tReturnFunction->FixParameter(3, 0.);

  aHist->Fit(tFitName, "0", "", aMinFit, aMaxFit);
  return tReturnFunction;
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
//-----------------------------------------------------------------------------

  AnalysisType tAnType = kLamKchP;


  bool bSaveFigures = false;

  int tImpactParam = 3;

  TString tFileDir = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/", tImpactParam);
  TString tSingleFileName = "CorrelationFunctions_RandomEPs_NumWeight1_wOtherPairs.root";
  TString tFileLocation = TString::Format("%s%s", tFileDir.Data(), tSingleFileName.Data());

  TString tHistName3d = TString::Format("PairSource3d_mT1vmT2vRinv%s", cAnalysisBaseTags[tAnType]);

//  TString tSaveDir = "/home/jesse/Analysis/Presentations/AliFemto/20180627/Figures/";
//  TString tSaveFileBase = tSaveDir + TString::Format("%s/", cAnalysisBaseTags[tAnType]);

  TH3* tTest3d = GetThermHist3d(tFileLocation, tHistName3d);

//  TH1D* tTestProj = tTest3d->ProjectionZ();
//  TH2D* tTestProj2d = (TH2D*)tTest3d->Project3D("zx");

  double tmTLow1=1.00, tmTHigh1=2.00;
  double tmTLow2=0.00, tmTHigh2=2.00;

  TH1D* tTestProj = ProjectOut1dHist(tTest3d, tmTLow1, tmTHigh1, tmTLow2, tmTHigh2);

  TCanvas* tCan = new TCanvas("tCan", "tCan");
  tCan->cd();

  TF1* tTestFit = FitHistogram(tTestProj, 1., 12.);

  tTestProj->Draw();
  tTestFit->Draw("same");
//-------------------------------------------------------------------------------




//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
