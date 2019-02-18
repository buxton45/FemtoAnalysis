#include "FitSharedAnalyses.h"
#include "LednickyFitter.h"
#include "NumIntLednickyCf.h"
#include "SimulatedLednickyCf.h"

//________________________________________________________________________________________________________________
TH1D* BuildNumIntCf(NumIntLednickyCf* aNumIntLedCf, TString aName, vector<double> &aKStarBinCenters, double* aParams)
{
  double tNBins = aKStarBinCenters.size();
  double tKStarBinSize = aKStarBinCenters[1]-aKStarBinCenters[0];

  TH1D* tReturnCf = new TH1D(aName, aName, tNBins, 0., tNBins*tKStarBinSize);

  for(int i=0; i<tNBins; i++)
  {
    tReturnCf->SetBinContent(i+1, aNumIntLedCf->GetFitCfContent(aKStarBinCenters[i], aParams));
  }
  return tReturnCf;
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

  ChronoTimer tFullTimer(kSec);
  tFullTimer.Start();
//-----------------------------------------------------------------------------
  int tIntType = 2;
  int tNCalls = 50000;
  double tMaxIntRadius = 100.;

  NumIntLednickyCf* tNumIntLedCf = new NumIntLednickyCf(tIntType, tNCalls, tMaxIntRadius);

  double tKStarBinSize = 0.01;
  int tNBins = 50;

  double tLambda = 1.12*0.527;
  double tRadius1 = 6.33;
  double tRef0   = -0.66;
  double tImf0   = 0.58;
  double td0     = 0.77;  
  double tNorm   = 1.0;
  double tMuOut1  = 0.0;

  double tRadius2 = 4.0;
  double tMuOut2a = 4.0;
  double tMuOut2b = 2.0;
  double tMuOut2c = 1.0;

  double tParams1[7] = {tLambda, tRadius1, tRef0, tImf0, td0, tNorm, tMuOut1};

  double tParams2a[7] = {tLambda, tRadius2, tRef0, tImf0, td0, tNorm, tMuOut2a};
  double tParams2b[7] = {tLambda, tRadius2, tRef0, tImf0, td0, tNorm, tMuOut2b};
  double tParams2c[7] = {tLambda, tRadius2, tRef0, tImf0, td0, tNorm, tMuOut2c};

  vector<double> tKStarBinCenters;
  for(int i=0; i<tNBins; i++) tKStarBinCenters.push_back((i+0.5)*tKStarBinSize);

  TH1D* tCf_LedEq = new TH1D("tCf_LedEq", "tCf_LedEq", tNBins, 0., tNBins*tKStarBinSize);
  TH1D* tCf_NumInt1 = BuildNumIntCf(tNumIntLedCf, TString("tCf_NumInt1"), tKStarBinCenters, tParams1);
  TH1D* tCf_NumInt2a = BuildNumIntCf(tNumIntLedCf, TString("tCf_NumInt2a"), tKStarBinCenters, tParams2a);
  TH1D* tCf_NumInt2b = BuildNumIntCf(tNumIntLedCf, TString("tCf_NumInt2b"), tKStarBinCenters, tParams2b);
  TH1D* tCf_NumInt2c = BuildNumIntCf(tNumIntLedCf, TString("tCf_NumInt2c"), tKStarBinCenters, tParams2c);

  double x[1];
  for(int i=0; i<tNBins; i++)
  {
    x[0] = tKStarBinCenters[i];
    tCf_LedEq->SetBinContent(i+1, FitPartialAnalysis::LednickyEq(x, tParams1));
  }
  tCf_LedEq->GetYaxis()->SetRangeUser(0.86, 1.07);

  SimulatedLednickyCf* tSimLedCf = new SimulatedLednickyCf(tKStarBinSize, tNBins*tKStarBinSize, 50000);
  TH1D* tCf_SimLedCf = new TH1D("tCf_SimLedCf", "tCf_SimLedCf", tNBins, 0., tNBins*tKStarBinSize);
  TH1D* tCf_SimLedCf2a = new TH1D("tCf_SimLedCf2a", "tCf_SimLedCf2a", tNBins, 0., tNBins*tKStarBinSize);
  for(int i=0; i<tNBins; i++)
  {
//    tCf_SimLedCf->SetBinContent(i+1, tSimLedCf->GetFitCfContent(tKStarBinCenters[i], tParams1));
    tCf_SimLedCf2a->SetBinContent(i+1, tSimLedCf->GetFitCfContent(tKStarBinCenters[i], tParams2a));
  }  


  tCf_LedEq->SetLineColor(kBlack);
  tCf_NumInt1->SetLineColor(kRed);
  tCf_NumInt2a->SetLineColor(kBlue);
  tCf_NumInt2b->SetLineColor(kGreen);
  tCf_NumInt2c->SetLineColor(kViolet);
  tCf_SimLedCf->SetLineColor(kOrange);
  tCf_SimLedCf2a->SetLineColor(kCyan);

  TCanvas* tCan = new TCanvas("tCan", "tCan");
  tCan->cd();
  tCf_LedEq->Draw();
//  tCf_NumInt1->Draw("same");
  tCf_NumInt2a->Draw("same");
//  tCf_NumInt2b->Draw("same");
//  tCf_NumInt2c->Draw("same");
//  tCf_SimLedCf->Draw("same");
  tCf_SimLedCf2a->Draw("same");



//-------------------------------------------------------------------------------
  tFullTimer.Stop();
  cout << "Finished program: ";
  tFullTimer.PrintInterval();

  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
  return 0;
}
