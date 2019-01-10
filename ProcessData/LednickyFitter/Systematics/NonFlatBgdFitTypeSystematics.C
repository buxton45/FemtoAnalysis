#include "FitSystematicAnalysis.h"
class FitSystematicAnalysis;

int main(int argc, char **argv) 
{
//  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  bool bPlayCompletionBeep = true;
//-----------------------------------------------------------------------------
  AnalysisType tAnType = kLamK0;
  CentralityType tCentralityType = kMB;
  FitGeneratorType tFitGeneratorType = kPairwConj;

  bool tShareLambdaParameters = false;
  bool tAllShareSingleLambdaParam = true;

  bool ApplyMomResCorrection = true;
  bool ApplyNonFlatBackgroundCorrection = true;
  NonFlatBgdFitType tNonFlatBgdFitType = kLinear;

  IncludeResidualsType tIncludeResidualsType = kInclude3Residuals; 
  ChargedResidualsType tChargedResidualsType = kUseXiDataAndCoulombOnlyInterp;
  ResPrimMaxDecayType tResPrimMaxDecayType = k10fm;

  bool FixD0 = false;

  bool bWriteToFile = true;
  bool bSaveImages = true;

  TString tResultsDate = "20180505";

  if(tAnType==kLamK0) tAllShareSingleLambdaParam = true;

  TString tGeneralAnTypeName;
  if(tAnType==kLamK0 || tAnType==kALamK0) tGeneralAnTypeName = "cLamK0";
  else if(tAnType==kLamKchP || tAnType==kALamKchM || tAnType==kLamKchM || tAnType==kALamKchP) tGeneralAnTypeName = "cLamcKch";
  else assert(0);

  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tResultsDate.Data());
  TString tFileLocationBase = tDirectoryBase + TString::Format("Results_%s_%s",tGeneralAnTypeName.Data(),tResultsDate.Data());
  TString tFileLocationBaseMC = tDirectoryBase + TString::Format("Results_%sMC_%s",tGeneralAnTypeName.Data(),tResultsDate.Data());

  FitSystematicAnalysis* tFitSysAn = new FitSystematicAnalysis(tFileLocationBase, tFileLocationBaseMC, tAnType, tCentralityType, tFitGeneratorType, tShareLambdaParameters, tAllShareSingleLambdaParam);
  tFitSysAn->SetSaveDirectory(tDirectoryBase);
  tFitSysAn->SetApplyNonFlatBackgroundCorrection(ApplyNonFlatBackgroundCorrection);
  tFitSysAn->SetNonFlatBgdFitType(tNonFlatBgdFitType);
  tFitSysAn->SetApplyMomResCorrection(ApplyMomResCorrection);

  tFitSysAn->SetIncludeResidualCorrelationsType(tIncludeResidualsType);
  tFitSysAn->SetChargedResidualsType(tChargedResidualsType);
  tFitSysAn->SetResPrimMaxDecayType(tResPrimMaxDecayType);

  tFitSysAn->SetFixD0(FixD0);

  tFitSysAn->RunVaryNonFlatBackgroundFit(bSaveImages, bWriteToFile);

cout << "DONE" << endl;
if(bPlayCompletionBeep) system("( speaker-test -t sine -f 1000 )& pid=$! ; sleep 0.5s ; kill -9 $pid");
//-------------------------------------------------------------------------------
//  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
