#include <iostream>
#include <iomanip>

#include "TApplication.h"
#include "TCanvas.h"

using std::cout;
using std::endl;
using std::vector;

#include "CorrFctnDirectYlmLite.h"


//_________________________________________________________________________________________
//*****************************************************************************************
//_________________________________________________________________________________________
int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------

  TString tFileLocation = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20181205/Results_cLamcKch_20181205_FemtoMinus.root";
  TString tDirectoryName = "LamKchP_0010";
  TString tSavedNameMod = "DirectYlmCf_LamKchP";
  TString tNewNameMod = "_0010_FemtoMinus";

  CorrFctnDirectYlmLite* tTestCfYlm = new CorrFctnDirectYlmLite(tFileLocation, tDirectoryName, tSavedNameMod, tNewNameMod, 2, 400, 0., 2.);


  TCanvas* tCan = new TCanvas("tCan", "tCan");
  tCan->cd();

  TH1D* tTestCfn = tTestCfYlm->GetCfnRealHist(1, 1); 

  tTestCfn->Draw();

//-------------------------------------------------------------------------------
  cout << "DONE" << endl;
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
