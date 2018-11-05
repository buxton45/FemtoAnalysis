#include "DualieFitSystematicAnalysis.h"
class DualieFitSystematicAnalysis;

int main(int argc, char **argv) 
{
//  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  bool bPlayCompletionBeep = true;
//-----------------------------------------------------------------------------
  AnalysisType tAnType = kLamKchP;
  CentralityType tCentralityType = kMB;
  FitGeneratorType tFitGeneratorType = kPairwConj;

  bool tShareLambdaParameters = true;
  bool tAllShareSingleLambdaParam = false;

  //--Dualie sharing options
  bool tDualieShareLambda = true;
  bool tDualieShareRadii = true;

  bool ApplyMomResCorrection = true;
  bool ApplyNonFlatBackgroundCorrection = true;
  NonFlatBgdFitType tNonFlatBgdFitType = kPolynomial;

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

  DualieFitSystematicAnalysis* tDualieFitSysAn = new DualieFitSystematicAnalysis(tFileLocationBase, tFileLocationBaseMC, tAnType, tCentralityType, tFitGeneratorType, tShareLambdaParameters, tAllShareSingleLambdaParam, tDualieShareLambda, tDualieShareRadii);
  tDualieFitSysAn->SetSaveDirectory(tDirectoryBase);
  tDualieFitSysAn->SetApplyNonFlatBackgroundCorrection(ApplyNonFlatBackgroundCorrection);
  tDualieFitSysAn->SetNonFlatBgdFitType(tNonFlatBgdFitType);
  tDualieFitSysAn->SetApplyMomResCorrection(ApplyMomResCorrection);

  tDualieFitSysAn->SetIncludeResidualCorrelationsType(tIncludeResidualsType);
  tDualieFitSysAn->SetChargedResidualsType(tChargedResidualsType);
  tDualieFitSysAn->SetResPrimMaxDecayType(tResPrimMaxDecayType);

  tDualieFitSysAn->SetFixD0(FixD0);

  tDualieFitSysAn->RunVaryNonFlatBackgroundFit(bSaveImages, bWriteToFile);

cout << "DONE" << endl;
if(bPlayCompletionBeep) system("( speaker-test -t sine -f 1000 )& pid=$! ; sleep 0.5s ; kill -9 $pid");
//-------------------------------------------------------------------------------
//  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
