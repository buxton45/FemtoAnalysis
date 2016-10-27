#include "SystematicAnalysis.h"
class SystematicAnalysis;



int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------

  TString tResultsDate = "20161026";
  AnalysisType tAnType = kLamKchP;
  CentralityType tCentType = k3050;

/*
  TString tDirNameModifierBase1 = "_ALLV0S_minInvMassReject_";
  vector<double> tModifierValues1 = {0.494614, 0.492614, 0.488614, 0.482614};

  TString tDirNameModifierBase2 = "_ALLV0S_maxInvMassReject_";
  vector<double> tModifierValues2 = {0.500614, 0.502614, 0.506614, 0.512614};
*/

  TString tDirNameModifierBase1 = "_ALLV0S_maxDcaV0Daughters_";
  vector<double> tModifierValues1 = {0.30,0.40,0.50};

  TString tDirNameModifierBase2 = "";
  vector<double> tModifierValues2 = {};


  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Systematics/Results_cLamcKch_Systematics%s",tDirNameModifierBase1.Data());
  if(!tDirNameModifierBase2.IsNull())
  {
    tDirectoryBase.Remove(TString::kTrailing,'_');
    tDirectoryBase += tDirNameModifierBase2;
  }
  tDirectoryBase += TString::Format("%s/",tResultsDate.Data());

  TString tFileLocationBase = tDirectoryBase + TString::Format("Results_cLamcKch_Systematics%s",tDirNameModifierBase1.Data());
  TString tFileLocationBaseMC = tDirectoryBase + TString::Format("Results_cLamcKchMC_Systematics%s",tDirNameModifierBase1.Data());
  if(!tDirNameModifierBase2.IsNull())
  {
    tFileLocationBase.Remove(TString::kTrailing,'_');
    tFileLocationBaseMC.Remove(TString::kTrailing,'_');

    tFileLocationBase += tDirNameModifierBase2;
    tFileLocationBaseMC += tDirNameModifierBase2;
  }
  tFileLocationBase += tResultsDate;
  tFileLocationBaseMC += tResultsDate;








  SystematicAnalysis* tSysAn = new SystematicAnalysis(tFileLocationBase,tAnType,tCentType,tDirNameModifierBase1,tModifierValues1,tDirNameModifierBase2,tModifierValues2);
  tSysAn->GetAllPValues();
  tSysAn->DrawAll();
  tSysAn->DrawAllDiffs();

cout << "DONE" << endl;
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
