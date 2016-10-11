#include "FitSharedAnalyses.h"
#include "LednickyFitter.h"

LednickyFitter *myFitter = NULL;

//______________________________________________________________________________
void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
  myFitter->CalculateChi2PML(npar,f,par);
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
//Be sure to set the following...

  TString FileLocation_cLamK0_Bp1 = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_AsRc_20150923/Results_cLamK0_AsRc_20150923_Bp1.root";
  TString FileLocation_cLamK0_Bp2 = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_AsRc_20150923/Results_cLamK0_AsRc_20150923_Bp2.root";
  TString FileLocation_cLamK0_Bm1 = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_AsRc_20150923/Results_cLamK0_AsRc_20150923_Bm1.root";
  TString FileLocation_cLamK0_Bm2 = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_AsRc_20150923/Results_cLamK0_AsRc_20150923_Bm2.root";
  TString FileLocation_cLamK0_Bm3 = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_AsRc_20150923/Results_cLamK0_AsRc_20150923_Bm3.root";

  FitPartialAnalysis* tFitPartialAnalysis_Bp1 = new FitPartialAnalysis(FileLocation_cLamK0_Bp1, "LamK0_0010_Bp1", kLamK0, k0010, kBp1);
  cout << sizeof(tFitPartialAnalysis_Bp1) << endl;
/*
  FitPartialAnalysis* tFitPartialAnalysis_Bp2 = new FitPartialAnalysis(FileLocation_cLamK0_Bp2, "LamK0_0010_Bp2", kLamK0, k0010, kBp2);
  FitPartialAnalysis* tFitPartialAnalysis_Bm1 = new FitPartialAnalysis(FileLocation_cLamK0_Bm1, "LamK0_0010_Bm1", kLamK0, k0010, kBm1);
  FitPartialAnalysis* tFitPartialAnalysis_Bm2 = new FitPartialAnalysis(FileLocation_cLamK0_Bm2, "LamK0_0010_Bm2", kLamK0, k0010, kBm2);
  FitPartialAnalysis* tFitPartialAnalysis_Bm3 = new FitPartialAnalysis(FileLocation_cLamK0_Bm3, "LamK0_0010_Bm3", kLamK0, k0010, kBm3);
*/
/*
  vector<FitPartialAnalysis*> tempVec;
  tempVec.push_back(tFitPartialAnalysis);

  FitPairAnalysis* tFitPairAnalysis = new FitPairAnalysis("LamK0",tempVec);
*/
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
  return 0;
}
