#include "FitGenerator.h"
class FitGenerator;

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------

  TString DirectoryBase_Train = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20161020/";
  TString FileLocationBase_Train = DirectoryBase_Train+"Results_cLamcKch_20161020";
  TString FileLocationBaseMC_Train = DirectoryBase_Train+"Results_cLamcKchMC_20161020";


  FitGenerator* tLamKchP = new FitGenerator(FileLocationBase_Train,FileLocationBaseMC_Train,kLamKchP,kTrain,2);
  tLamKchP->SetSaveLocationBase(DirectoryBase_Train);
  bool SaveImages = true;
  bool ApplyMomResCorrection = false;

//  TCanvas* tKStarCan = tLamKchP->DrawKStarCfs();

  //tLamKchP->SetDefaultSharedParameters();

//TODO!!!!  If I want to apply mom res correction to full fit, I need to give non-central analyses ability to grab
//           the matrix from the central analyses
  tLamKchP->DoFit(ApplyMomResCorrection);
  TCanvas* tKStarwFitsCan = tLamKchP->DrawKStarCfswFits(SaveImages);


//  TString FileLocationBase = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20161007/Results_cLamcKch_20161007";
//  TString FileLocationBaseMC = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20161007/Results_cLamcKchMC_20161007";


//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
cout << "DONE" << endl;
  return 0;
}
