#include "FitSystematicAnalysis.h"
class FitSystematicAnalysis;

#include "Types_SysFileInfo.h"


int main(int argc, char **argv) 
{
//  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  bool bPlayCompletionBeep = true;
//-----------------------------------------------------------------------------
  TString tParentResultsDate = "20180505";  //Parent analysis these systematics are to accompany

  AnalysisType tAnType = kLamK0;
  CentralityType tCentralityType = kMB;
  FitGeneratorType tFitGeneratorType = kPairwConj;
  bool tShareLambdaParameters = false;
  bool tAllShareSingleLambdaParam = false;

  bool ApplyMomResCorrection = true;
  bool ApplyNonFlatBackgroundCorrection = true;
  NonFlatBgdFitType tNonFlatBgdFitType = kPolynomial;

  IncludeResidualsType tIncludeResidualsType = kInclude3Residuals; 
  ChargedResidualsType tChargedResidualsType = kUseXiDataAndCoulombOnlyInterp;
  ResPrimMaxDecayType tResPrimMaxDecayType = k4fm;

  bool FixD0 = false;

  bool bWriteToFile = true;
  bool bSaveImages = true;

  if(tAnType==kLamK0) tAllShareSingleLambdaParam = true;

  SystematicsFileInfo tFileInfo = GetFileInfo_LamK(-16, tParentResultsDate);
    TString tResultsDate = tFileInfo.resultsDate;
    TString tDirNameModifierBase1 = tFileInfo.dirNameModifierBase1;
    vector<double> tModifierValues1 = tFileInfo.modifierValues1;
    TString tDirNameModifierBase2 = tFileInfo.dirNameModifierBase2;
    vector<double> tModifierValues2 = tFileInfo.modifierValues2;
    bool tAllCent = tFileInfo.allCentralities;

  TString tGeneralAnTypeName;
  if(tAnType==kLamK0 || tAnType==kALamK0) tGeneralAnTypeName = "cLamK0";
  else if(tAnType==kLamKchP || tAnType==kALamKchM || tAnType==kLamKchM || tAnType==kALamKchP) tGeneralAnTypeName = "cLamcKch";
  else assert(0);

  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Systematics_LamK_%s/Results_%s_Systematics%s", tParentResultsDate.Data(), tGeneralAnTypeName.Data(), tDirNameModifierBase1.Data());
  if(!tDirNameModifierBase2.IsNull())
  {
    tDirectoryBase.Remove(TString::kTrailing,'_');
    tDirectoryBase += tDirNameModifierBase2;
  }
  tDirectoryBase += TString::Format("%s/",tResultsDate.Data());

  TString tFileLocationBase = tDirectoryBase + TString::Format("Results_%s_Systematics%s",tGeneralAnTypeName.Data(),tDirNameModifierBase1.Data());
  TString tFileLocationBaseMC = tDirectoryBase + TString::Format("Results_%sMC_Systematics%s",tGeneralAnTypeName.Data(),tDirNameModifierBase1.Data());
  if(!tDirNameModifierBase2.IsNull())
  {
    tFileLocationBase.Remove(TString::kTrailing,'_');
    tFileLocationBaseMC.Remove(TString::kTrailing,'_');

    tFileLocationBase += tDirNameModifierBase2;
    tFileLocationBaseMC += tDirNameModifierBase2;
  }
  tFileLocationBase += tResultsDate;
  tFileLocationBaseMC += tResultsDate;

  FitSystematicAnalysis* tFitSysAn = new FitSystematicAnalysis(tFileLocationBase, tFileLocationBaseMC, tAnType, tDirNameModifierBase1, tModifierValues1, tDirNameModifierBase2, tModifierValues2, tCentralityType, tFitGeneratorType, tShareLambdaParameters, tAllShareSingleLambdaParam);
  tFitSysAn->SetSaveDirectory(tDirectoryBase);
  tFitSysAn->SetApplyNonFlatBackgroundCorrection(ApplyNonFlatBackgroundCorrection);
  tFitSysAn->SetNonFlatBgdFitType(tNonFlatBgdFitType);
  tFitSysAn->SetApplyMomResCorrection(ApplyMomResCorrection);

  tFitSysAn->SetIncludeResidualCorrelationsType(tIncludeResidualsType);
  tFitSysAn->SetChargedResidualsType(tChargedResidualsType);
  tFitSysAn->SetResPrimMaxDecayType(tResPrimMaxDecayType);

  tFitSysAn->SetFixD0(FixD0);

  tFitSysAn->RunAllFits(bSaveImages, bWriteToFile);

cout << "DONE" << endl;
if(bPlayCompletionBeep) system("( speaker-test -t sine -f 1000 )& pid=$! ; sleep 0.5s ; kill -9 $pid");
//-------------------------------------------------------------------------------
//  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
