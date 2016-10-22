#include "FitGenerator.h"
class FitGenerator;

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------

  TString FileLocationBase_cLamK0 = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_AsRc_20151228_Old/Results_cLamK0_AsRc_20151228_Old";
  TString FileLocationBaseMC_cLamK0 = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_AsRc_20150923/Results_cLamK0_AsRcMC_20150923";

  TString FileLocationBase_cLamcKch = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_AsRc_20151007/Results_cLamcKch_AsRc_20151007";
  TString FileLocationBaseMC_cLamcKch = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_AsRc_20151007/Results_cLamcKch_AsRcMC_20151007";

  TString FileLocationBase_Train = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20161012/Results_cLamcKch_20161012";
  TString FileLocationBaseMC_Train = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20161012/Results_cLamcKchMC_20161012";


  FitGenerator* tLamKchP = new FitGenerator(FileLocationBase_Train,FileLocationBaseMC_Train,kLamKchP,kTrain,2,k0010);


//  TCanvas* tKStarCan = tLamKchP->DrawKStarCfs();

  //tLamKchP->SetDefaultSharedParameters();

//TODO!!!!  If I want to apply mom res correction to full fit, I need to give non-central analyses ability to grab
//           the matrix from the central analyses
  tLamKchP->DoFit(true);
  TCanvas* tKStarwFitsCan = tLamKchP->DrawKStarCfswFits();

//  TString FileLocationBase = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20161007/Results_cLamcKch_20161007";
//  TString FileLocationBaseMC = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20161007/Results_cLamcKchMC_20161007";


//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
cout << "DONE" << endl;
  return 0;
}
