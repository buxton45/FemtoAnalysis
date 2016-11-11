#include "FitGenerator.h"
class FitGenerator;

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  ChronoTimer tFullTimer(kSec);
  tFullTimer.Start();
//-----------------------------------------------------------------------------
  TString tResultsDate = "20161025";

  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",tResultsDate.Data());
  TString tFileLocationBase = TString::Format("%sResults_cLamcKch_%s",tDirectoryBase.Data(),tResultsDate.Data());
  TString tFileLocationBaseMC = TString::Format("%sResults_cLamcKchMC_%s",tDirectoryBase.Data(),tResultsDate.Data());

  AnalysisType tAnType = kLamKchP;
  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;
  CentralityType tCentType = kMB;
  FitGeneratorType tGenType = kPairwConj;
  bool tShareLambdaParams = false;

  FitGenerator* tLamKchP = new FitGenerator(tFileLocationBase,tFileLocationBaseMC,tAnType,tAnRunType,tNPartialAnalysis,tCentType,tGenType,tShareLambdaParams);
  tLamKchP->SetSaveLocationBase(tDirectoryBase);
  //tLamKchP->SetFitType(kChi2);
  bool SaveImages = true;
  bool ApplyMomResCorrection = true;
  bool ApplyNonFlatBackgroundCorrection = true;

//  TCanvas* tKStarCan = tLamKchP->DrawKStarCfs();

  //tLamKchP->SetDefaultSharedParameters();

//TODO!!!!  If I want to apply mom res correction to full fit, I need to give non-central analyses ability to grab
//           the matrix from the central analyses
  tLamKchP->DoFit(ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection);
  TCanvas* tKStarwFitsCan = tLamKchP->DrawKStarCfswFits(ApplyMomResCorrection,ApplyNonFlatBackgroundCorrection,SaveImages);




//-------------------------------------------------------------------------------
  tFullTimer.Stop();
  cout << "Finished program: ";
  tFullTimer.PrintInterval();

  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
cout << "DONE" << endl;
  return 0;
}
