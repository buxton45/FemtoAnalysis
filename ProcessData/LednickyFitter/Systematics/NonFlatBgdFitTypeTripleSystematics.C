#include "TripleFitSystematicAnalysis.h"
class TripleFitSystematicAnalysis;

#include "Types_SysFileInfo.h"


//____________________________________________________________________________
//****************************************************************************
//____________________________________________________________________________
int main(int argc, char **argv) 
{
//  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  bool bPlayCompletionBeep = true;
//-----------------------------------------------------------------------------
  TString tResultsDate = "20190319";  //Parent analysis these systematics are to accompany

  CentralityType tCentralityType = kMB;
  FitGeneratorType tFitGeneratorType = kPairwConj;
  bool tShareLambdaParameters = true;
  bool tAllShareSingleLambdaParam = false;

  //--Dualie sharing options
  bool tDualieShareLambda = true;
  bool tDualieShareRadii = true;

  bool ApplyMomResCorrection = true;
  bool ApplyNonFlatBackgroundCorrection = true;
  NonFlatBgdFitType tNonFlatBgdFitType_LamKch = kPolynomial;
  NonFlatBgdFitType tNonFlatBgdFitType_LamK0  = kPolynomial;

  IncludeResidualsType tIncludeResidualsType = kInclude3Residuals; 
  ChargedResidualsType tChargedResidualsType = kUseXiDataAndCoulombOnlyInterp;
  ResPrimMaxDecayType tResPrimMaxDecayType = k10fm;

  bool FixD0 = false;

  bool bWriteToFile = true;
  bool bSaveImages = true;

  TString tGeneralAnTypeModified = "";

  TString tDirectoryBase_LamKch, tFileLocationBase_LamKch, tFileLocationBaseMC_LamKch;
  TString tDirectoryBase_LamK0, tFileLocationBase_LamK0, tFileLocationBaseMC_LamK0;

  //LamKch results come from NORMAL location
  tDirectoryBase_LamKch = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",tResultsDate.Data());
  tFileLocationBase_LamKch = TString::Format("%sResults_cLamcKch_%s",tDirectoryBase_LamKch.Data(),tResultsDate.Data());
  tFileLocationBaseMC_LamKch = TString::Format("%sResults_cLamcKchMC_%s",tDirectoryBase_LamKch.Data(),tResultsDate.Data());

  //LamK0 results come from NORMAL location
  tDirectoryBase_LamK0 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/",tResultsDate.Data());
  tFileLocationBase_LamK0 = TString::Format("%sResults_cLamK0_%s",tDirectoryBase_LamK0.Data(),tResultsDate.Data());
  tFileLocationBaseMC_LamK0 = TString::Format("%sResults_cLamK0MC_%s",tDirectoryBase_LamK0.Data(),tResultsDate.Data());

  TString tSaveDirectoryBase = tDirectoryBase_LamKch;

  TripleFitSystematicAnalysis* tTripleFitSysAn = new TripleFitSystematicAnalysis(tFileLocationBase_LamKch, tFileLocationBaseMC_LamKch,
                                                                                 tFileLocationBase_LamK0, tFileLocationBaseMC_LamK0,
                                                                                 tGeneralAnTypeModified, 
                                                                                 tCentralityType, tFitGeneratorType, 
                                                                                 tShareLambdaParameters, tAllShareSingleLambdaParam, tDualieShareLambda, tDualieShareRadii);
  tTripleFitSysAn->SetSaveDirectory(tSaveDirectoryBase);
  tTripleFitSysAn->SetApplyNonFlatBackgroundCorrection(ApplyNonFlatBackgroundCorrection);
  tTripleFitSysAn->SetNonFlatBgdFitTypes(tNonFlatBgdFitType_LamKch, tNonFlatBgdFitType_LamK0);
  tTripleFitSysAn->SetApplyMomResCorrection(ApplyMomResCorrection);

  tTripleFitSysAn->SetIncludeResidualCorrelationsType(tIncludeResidualsType);
  tTripleFitSysAn->SetChargedResidualsType(tChargedResidualsType);
  tTripleFitSysAn->SetResPrimMaxDecayType(tResPrimMaxDecayType);

  tTripleFitSysAn->SetFixD0(FixD0);

  tTripleFitSysAn->RunVaryNonFlatBackgroundFit(bSaveImages, bWriteToFile);

cout << "DONE" << endl;
if(bPlayCompletionBeep) system("( speaker-test -t sine -f 1000 )& pid=$! ; sleep 0.5s ; kill -9 $pid");
//-------------------------------------------------------------------------------
//  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
