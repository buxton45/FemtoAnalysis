#include "ChargedResidualCf.h"

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

  TString tInterpHistLocation = "/home/jesse/Analysis/FemtoAnalysis/ProcessData/CoulombFitter/InterpHistsAttractive";
  TString tHFunctionLocation = "/home/jesse/Analysis/FemtoAnalysis/ProcessData/CoulombFitter/LednickyHFunction";

  ResidualType tResidualType = kXiCKchP;

  ChargedResidualCf* tResidualCf = new ChargedResidualCf(tResidualType,tInterpHistLocation,tHFunctionLocation);

  double *tParams = new double[9];
    tParams[0] = 0.31;
    tParams[1] = 2.84;
    tParams[2] = -1.59;
    tParams[3] = -0.37;
    tParams[4] = 5.0;
    tParams[5] = -0.46;
    tParams[6] = 1.13;
    tParams[7] = -2.53;
    tParams[8] = 1.0;

  double tBinWidth = 0.01;
  double tKStarMin = 0.0;
  double tKStarMax = 0.3;
  int tNbins = (tKStarMax-tKStarMin)/tBinWidth;

  td1dVec tKStarBinCenters (tNbins,0.);
  for(int i=0; i<tNbins; i++)
  {
    tKStarBinCenters[i] = tBinWidth*(i+0.5);
cout << "tKStarBinCenters[i] = " << tKStarBinCenters[i] << endl;
  }

TH2D* tTest = new TH2D("tTest","tTest",30,0.,0.3,36,0.,3.14159);

  tResidualCf->SetIncludeSingletAndTriplet(true);
  td1dVec tCf = tResidualCf->GetCoulombResidualCorrelation(tParams,tKStarBinCenters,tTest);
  TCanvas* tCan2 = new TCanvas("tCan2","tCan2");
  tCan2->cd();
  tTest->Draw("colz");

  assert(tCf.size() == tKStarBinCenters.size());
/*
  for(unsigned int i=0; i<tCf.size(); i++)
  {
    cout << "i = " << i << endl;
    cout << "tKStarBinCenters = " << tKStarBinCenters[i] << endl;
    cout << "tCf = " << tCf[i] << endl;
    cout << endl;
  }

  TH1D* tCfHist = tResidualCf->Convert1dVecToHist(tCf,tKStarBinCenters);
  TCanvas* tCan = new TCanvas("tCan","tCan");
  tCan->cd();
  tCfHist->Draw();
*/
//-------------------------------------------------------------------------------
  tFullTimer.Stop();
  cout << "Finished program: ";
  tFullTimer.PrintInterval();

  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.


  return 0;
}
