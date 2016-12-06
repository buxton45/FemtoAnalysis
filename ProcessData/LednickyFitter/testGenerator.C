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
  TString tResultsDate = "20161027";

  AnalysisType tAnType = kLamKchP;
  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;
  CentralityType tCentType = kMB;
  FitGeneratorType tGenType = kPairwConj;
  bool tShareLambdaParams = false;


  bool SaveImages = false;
  bool ApplyMomResCorrection = true;
  bool ApplyNonFlatBackgroundCorrection = true;

  TString tGeneralAnTypeName;
  if(tAnType==kLamK0 || tAnType==kALamK0) tGeneralAnTypeName = "cLamK0";
  else if(tAnType==kLamKchP || tAnType==kALamKchM || tAnType==kLamKchM || tAnType==kALamKchP) tGeneralAnTypeName = "cLamcKch";
  else assert(0);


  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tResultsDate.Data());
  TString tFileLocationBase = TString::Format("%sResults_%s_%s",tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());
  TString tFileLocationBaseMC = TString::Format("%sResults_%sMC_%s",tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());


  TString tSaveNameModifier = "";
  if(ApplyMomResCorrection) tSaveNameModifier += TString("_MomResCrctn");
  if(ApplyNonFlatBackgroundCorrection) tSaveNameModifier += TString("_NonFlatBgdCrctn");
  FitGenerator* tLamKchP = new FitGenerator(tFileLocationBase,tFileLocationBaseMC,tAnType,tAnRunType,tNPartialAnalysis,tCentType,tGenType,tShareLambdaParams);
  tLamKchP->SetSaveLocationBase(tDirectoryBase,tSaveNameModifier);
  //tLamKchP->SetFitType(kChi2);


//  TCanvas* tKStarCan = tLamKchP->DrawKStarCfs();

  //tLamKchP->SetDefaultSharedParameters();

//TODO!!!!  If I want to apply mom res correction to full fit, I need to give non-central analyses ability to grab
//           the matrix from the central analyses
  tLamKchP->DoFit(ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection);
  TCanvas* tKStarwFitsCan = tLamKchP->DrawKStarCfswFits(ApplyMomResCorrection,ApplyNonFlatBackgroundCorrection,SaveImages);
//  TCanvas* tKStarCfs = tLamKchP->DrawKStarCfs(SaveImages);

//  tLamKchP->FindGoodInitialValues(ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection);

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
