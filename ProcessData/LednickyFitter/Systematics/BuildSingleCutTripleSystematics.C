#include "TripleFitSystematicAnalysis.h"
class TripleFitSystematicAnalysis;

#include "Types_SysFileInfo.h"

//____________________________________________________________________________
void AppendtDateAndDirNameModifierBase2(TString &aBase, bool aAddFinalSlash, TString aDate, TString aDirNameModifierBase2)
{
  if(!aDirNameModifierBase2.IsNull())
  {
    aBase.Remove(TString::kTrailing,'_');
    aBase += aDirNameModifierBase2;
  }

  if(aAddFinalSlash) aBase += TString::Format("%s/",aDate.Data());
  else               aBase += aDate.Data();
}


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
  TString tParentResultsDate = "20180505";  //Parent analysis these systematics are to accompany

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
  NonFlatBgdFitType tNonFlatBgdFitType_LamK0  = kLinear;

  IncludeResidualsType tIncludeResidualsType = kInclude3Residuals; 
  ChargedResidualsType tChargedResidualsType = kUseXiDataAndCoulombOnlyInterp;
  ResPrimMaxDecayType tResPrimMaxDecayType = k10fm;

  bool FixD0 = false;

  bool bWriteToFile = true;
  bool bSaveImages = true;

//  TString tGeneralAnTypeModified = "cLamcKch";
  TString tGeneralAnTypeModified = "cLamK0";
  int tCut = 10;
  if(tGeneralAnTypeModified.EqualTo("cLamK0")) tCut *= -1;

  SystematicsFileInfo tFileInfo = GetFileInfo_LamK(tCut, tParentResultsDate);
    TString tResultsDate = tFileInfo.resultsDate;
    TString tDirNameModifierBase1 = tFileInfo.dirNameModifierBase1;
    vector<double> tModifierValues1 = tFileInfo.modifierValues1;
    TString tDirNameModifierBase2 = tFileInfo.dirNameModifierBase2;
    vector<double> tModifierValues2 = tFileInfo.modifierValues2;
    bool tAllCent = tFileInfo.allCentralities;


  TString tDirectoryBase_LamKch, tFileLocationBase_LamKch, tFileLocationBaseMC_LamKch;
  TString tDirectoryBase_LamK0, tFileLocationBase_LamK0, tFileLocationBaseMC_LamK0;
  TString tSaveDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Systematics_LamK_%s/TripleSystematics/Results_TripleSystematics_%sVaried%s", tParentResultsDate.Data(), tGeneralAnTypeModified.Data(), tDirNameModifierBase1.Data());
  AppendtDateAndDirNameModifierBase2(tSaveDirectoryBase, true, tResultsDate, tDirNameModifierBase2);


  if(tGeneralAnTypeModified.EqualTo("cLamK0"))
  {
    //LamK0 results come from MODIFIED location
    tDirectoryBase_LamK0 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Systematics_LamK_%s/Results_cLamK0_Systematics%s", tParentResultsDate.Data(), tDirNameModifierBase1.Data());
    AppendtDateAndDirNameModifierBase2(tDirectoryBase_LamK0, true, tResultsDate, tDirNameModifierBase2);

    tFileLocationBase_LamK0 = tDirectoryBase_LamK0 + TString::Format("Results_cLamK0_Systematics%s", tDirNameModifierBase1.Data());
    tFileLocationBaseMC_LamK0 = tDirectoryBase_LamK0 + TString::Format("Results_cLamK0MC_Systematics%s", tDirNameModifierBase1.Data());
    AppendtDateAndDirNameModifierBase2(tFileLocationBase_LamK0, false, tResultsDate, tDirNameModifierBase2);
    AppendtDateAndDirNameModifierBase2(tFileLocationBaseMC_LamK0, false, tResultsDate, tDirNameModifierBase2);
    //-----------------------------------------------------------------------------------

    //LamKch results come from NORMAL location
    tDirectoryBase_LamKch = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",tParentResultsDate.Data());
    tFileLocationBase_LamKch = TString::Format("%sResults_cLamcKch_%s",tDirectoryBase_LamKch.Data(),tParentResultsDate.Data());
    tFileLocationBaseMC_LamKch = TString::Format("%sResults_cLamcKchMC_%s",tDirectoryBase_LamKch.Data(),tParentResultsDate.Data());
  }
  else if(tGeneralAnTypeModified.EqualTo("cLamcKch"))
  {
    //LamKch results come from MODIFIED location
    tDirectoryBase_LamKch = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Systematics_LamK_%s/Results_cLamcKch_Systematics%s", tParentResultsDate.Data(), tDirNameModifierBase1.Data());
    AppendtDateAndDirNameModifierBase2(tDirectoryBase_LamKch, true, tResultsDate, tDirNameModifierBase2);

    tFileLocationBase_LamKch = tDirectoryBase_LamKch + TString::Format("Results_cLamcKch_Systematics%s", tDirNameModifierBase1.Data());
    tFileLocationBaseMC_LamKch = tDirectoryBase_LamKch + TString::Format("Results_cLamcKchMC_Systematics%s", tDirNameModifierBase1.Data());
    AppendtDateAndDirNameModifierBase2(tFileLocationBase_LamKch, false, tResultsDate, tDirNameModifierBase2);
    AppendtDateAndDirNameModifierBase2(tFileLocationBaseMC_LamKch, false, tResultsDate, tDirNameModifierBase2);

    //-----------------------------------------------------------------------------------

    //LamK0 results come from NORMAL location
    tDirectoryBase_LamK0 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/",tParentResultsDate.Data());
    tFileLocationBase_LamK0 = TString::Format("%sResults_cLamK0_%s",tDirectoryBase_LamK0.Data(),tParentResultsDate.Data());
    tFileLocationBaseMC_LamK0 = TString::Format("%sResults_cLamK0MC_%s",tDirectoryBase_LamK0.Data(),tParentResultsDate.Data());
  }
  else assert(0);







  TripleFitSystematicAnalysis* tTripleFitSysAn = new TripleFitSystematicAnalysis(tFileLocationBase_LamKch, tFileLocationBaseMC_LamKch,
                                                                                 tFileLocationBase_LamK0, tFileLocationBaseMC_LamK0,
                                                                                 tGeneralAnTypeModified, 
                                                                                 tDirNameModifierBase1, tModifierValues1, 
                                                                                 tDirNameModifierBase2, tModifierValues2, 
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

  tTripleFitSysAn->RunAllFits(bSaveImages, bWriteToFile);

cout << "DONE" << endl;
if(bPlayCompletionBeep) system("( speaker-test -t sine -f 1000 )& pid=$! ; sleep 0.5s ; kill -9 $pid");
//-------------------------------------------------------------------------------
//  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
