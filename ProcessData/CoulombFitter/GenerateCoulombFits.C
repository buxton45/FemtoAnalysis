#include "CoulombFitGenerator.h"
class CoulombFitGenerator;

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  ChronoTimer tFullTimer(kSec);
  tFullTimer.Start();
//-----------------------------------------------------------------------------
  TString tResultsDate = "20170423";

  AnalysisType tAnType = kXiKchP;
  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;
  CentralityType tCentType = k0010;  //TODO
  FitGeneratorType tGenType = kPairwConj;
  bool tShareLambdaParams = false;
  bool tAllShareSingleLambdaParam = false;

  bool SaveImages = false;
  bool ApplyMomResCorrection = false;
  bool ApplyNonFlatBackgroundCorrection = false;
  NonFlatBgdFitType tNonFlatBgdFitType = kLinear;
  bool IncludeResiduals = false;
  bool IncludeSingleAndTriplet = false;

  TString tGeneralAnTypeName = "cXicKch";

  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tResultsDate.Data());
  TString tFileLocationBase = TString::Format("%sResults_%s_%s",tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());
  TString tFileLocationBaseMC = TString::Format("%sResults_%sMC_%s",tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());


  TString tSaveNameModifier = "";
  if(ApplyMomResCorrection) tSaveNameModifier += TString("_MomResCrctn");
  if(ApplyNonFlatBackgroundCorrection) tSaveNameModifier += TString("_NonFlatBgdCrctn");
  if(tAllShareSingleLambdaParam) tSaveNameModifier += TString("_SingleLamParam");
  CoulombFitGenerator* tXiKchP = new CoulombFitGenerator(tFileLocationBase,tFileLocationBaseMC,tAnType, tCentType, tAnRunType, tNPartialAnalysis, tGenType, tShareLambdaParams, tAllShareSingleLambdaParam, TString(""), IncludeSingleAndTriplet);
//  CoulombFitGenerator* tXiKchP = new CoulombFitGenerator(tFileLocationBase, tFileLocationBaseMC, tAnType,{k0010,k1030}, tAnRunType, tNPartialAnalysis, tGenType, tShareLambdaParams, tAllShareSingleLambdaParam, TString(""), IncludeSingleAndTriplet);
//  tXiKchP->SetRadiusStartValues({3.0,4.0,5.0});
//  tXiKchP->SetRadiusLimits({{0.,10.},{0.,10.},{0.,10.}});
  tXiKchP->SetSaveLocationBase(tDirectoryBase,tSaveNameModifier);
  //tXiKchP->SetFitType(kChi2);


//  TCanvas* tKStarCan = tXiKchP->DrawKStarCfs();

  //tXiKchP->SetDefaultSharedParameters();

//TODO!!!!  If I want to apply mom res correction to full fit, I need to give non-central analyses ability to grab
//           the matrix from the central analyses
  tXiKchP->DoFit(ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, IncludeResiduals, IncludeSingleAndTriplet, tNonFlatBgdFitType);
  TCanvas* tKStarwFitsCan = tXiKchP->DrawKStarCfswFits(ApplyMomResCorrection,ApplyNonFlatBackgroundCorrection,tNonFlatBgdFitType,SaveImages);
//  TCanvas* tKStarCfs = tXiKchP->DrawKStarCfs(SaveImages);
//  TCanvas* tModelKStarCfs = tXiKchP->DrawModelKStarCfs(SaveImages);
//  tXiKchP->FindGoodInitialValues(ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection);

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
