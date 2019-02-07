#include "StrippedSimpleFitter.h"
class StrippedSimpleFitter;

StrippedSimpleFitter *myFitter = NULL;

//______________________________________________________________________________
void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
  myFitter->StrippedSimpleFitter::CalculateFitFunction(npar,f,par);
}


//________________________________________________________________________________________________________________
TH1D* Get1dHisto(TString FileName, TString HistoName)
{
  TFile f1(FileName);
  TH1D *ReturnHisto = (TH1D*)f1.Get(HistoName);
  assert(ReturnHisto);
  TH1D *ReturnHistoClone = (TH1D*)ReturnHisto->Clone();
  ReturnHistoClone->SetDirectory(0);

  return ReturnHistoClone;
}


//______________________________________________________________________________
int main(int argc, char **argv) 
//int RunStrippedSimpleFitter() 
{
//int argc = 0; char **argv = NULL;
//  gROOT->LoadMacro("StrippedSimpleFitter.cxx+");

  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

//-----------------------------------------------------------------------------
  TH1* aNum = nullptr;
  TH1* aDen = nullptr;

  StrippedSimpleFitter::FitType tFitType = StrippedSimpleFitter::kChi2PML/*StrippedSimpleFitter::kChi2*/;
  double tMaxFitKStar = 0.3;
  double tMinNormKStar = 0.32;
  double tMaxNormKStar = 0.40;

  TString tFileLocationCfs = "/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/CorrelationFunctions_wOtherPairs.root";
  TString tNumName = "NumFullLamKchP";
  TString tDenName = "DenFullLamKchP";
  aNum = Get1dHisto(tFileLocationCfs, tNumName);
  aDen = Get1dHisto(tFileLocationCfs, tDenName);
//-------------------------------------------------------------------------------

  StrippedSimpleFitter *tSLFitter = new StrippedSimpleFitter(aNum, aDen, tMaxFitKStar, tMinNormKStar, tMaxNormKStar);

  tSLFitter->GetMinuitObject()->SetFCN(fcn);
  myFitter = tSLFitter;


  tSLFitter->SetFitType(tFitType);
  tSLFitter->DoFit();


  TString tCanName = TString("tCanCfWithFit");
  TCanvas* tCanCfWithFit = new TCanvas(tCanName, tCanName);
  tSLFitter->DrawCfWithFit((TPad*)tCanCfWithFit);




//-------------------------------------------------------------------------------
//  tFullTimer.Stop();
//  cout << "Finished program: ";
 // tFullTimer.PrintInterval();

  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
cout << "DONE" << endl;
  return 0;
}
