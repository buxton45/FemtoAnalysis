#include "FitSharedAnalyses.h"
#include "LednickyFitter.h"
#include "FitValuesWriter.h"
#include "FitValuesWriterwSysErrs.h"
#include "FitValuesLatexTableHelperWriterwSysErrs.h"

#include "TObjString.h"


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


int main(int argc, char **argv)
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------
  IncludeResidualsType tIncResType = kInclude3Residuals;
  ResPrimMaxDecayType tResPrimMaxDecayType = k4fm;

  TString tMasterFileLocation_LamKch = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20180505/MasterFitResults_20180505.txt";

  TString tSystematicsFileLocation_LamKch = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20171227/Systematics/_MomResCrctn_NonFlatBgdCrctn%s%s_UsingXiDataAndCoulombOnly/FinalFitSystematics_wFitRangeSys_MomResCrctn_NonFlatBgdCrctn%s%s_UsingXiDataAndCoulombOnly_cLamcKch.txt", cIncludeResidualsTypeTags[tIncResType], cResPrimMaxDecayTypeTags[tResPrimMaxDecayType], cIncludeResidualsTypeTags[tIncResType], cResPrimMaxDecayTypeTags[tResPrimMaxDecayType]);
  TString tFitInfoTString_LamKch = TString::Format("_MomResCrctn_NonFlatBgdCrctnPolynomial%s%s_UsingXiDataAndCoulombOnly", cIncludeResidualsTypeTags[tIncResType], cResPrimMaxDecayTypeTags[tResPrimMaxDecayType]);

  TString tMasterFileLocation_LamK0 = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_20180505/MasterFitResults_20180505.txt";

  TString tSystematicsFileLocation_LamK0 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_20171227/Systematics/_MomResCrctn_NonFlatBgdCrctn%s%s_UsingXiDataAndCoulombOnly/FinalFitSystematics_wFitRangeSys_MomResCrctn_NonFlatBgdCrctn%s%s_UsingXiDataAndCoulombOnly_cLamK0.txt", cIncludeResidualsTypeTags[tIncResType], cResPrimMaxDecayTypeTags[tResPrimMaxDecayType], cIncludeResidualsTypeTags[tIncResType], cResPrimMaxDecayTypeTags[tResPrimMaxDecayType]);

  TString tHelperBaseLocation = "/home/jesse/Analysis/FemtoAnalysis/ProcessData/LednickyFitter/Systematics/TableHelper";

  AnalysisType tAnType = kLamKchP;
//  AnalysisType tAnType = kLamK0;



  TString tMasterFileLocation, tSystematicsFileLocation;
  if(tAnType==kLamKchP)
  {
    tMasterFileLocation = tMasterFileLocation_LamKch;
    tSystematicsFileLocation = tSystematicsFileLocation_LamKch;
  }
  else if(tAnType==kLamK0)
  {
    tMasterFileLocation = tMasterFileLocation_LamK0;
    tSystematicsFileLocation = tSystematicsFileLocation_LamK0;
  }
  else assert(0);

//  FitValuesWriterwSysErrs* tFitValWriterwSysErrs = new FitValuesWriterwSysErrs();
//  vector<vector<FitParameter*> > tTest = tFitValWriterwSysErrs->ReadAllParameters(tMasterFileLocation_LamKch, tSystematicsFileLocation_LamKch, tFitInfoTString_LamKch, tAnType);

  FitValuesLatexTableHelperWriterwSysErrs* tFitValLatTabHelpWriterwSysErrs = new FitValuesLatexTableHelperWriterwSysErrs();
  tFitValLatTabHelpWriterwSysErrs->WriteLatexTableHelper(tHelperBaseLocation, tMasterFileLocation, tSystematicsFileLocation, tAnType, tIncResType, tResPrimMaxDecayType);


//-------------------------------------------------------------------------------
  cout << "DONE" << endl;
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
  return 0;
}
