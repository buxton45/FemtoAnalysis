#include "FitSharedAnalyses.h"
#include "LednickyFitter.h"
#include "FitValuesWriter.h"
#include "FitValuesWriterwErrs.h"

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
  TString tMasterFileLocation = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20180505/MasterFitResults_20180505.txt";
  TString tSystematicsFileLocation = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20171227/Systematics/_MomResCrctn_NonFlatBgdCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly/FinalFitSystematics_wFitRangeSys_MomResCrctn_NonFlatBgdCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_cLamcKch.txt";
  TString tFitInfoTString = "_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly";
  AnalysisType tAnType = kLamKchP;



  FitValuesWriterwErrs* tFitValWriterwErrs = new FitValuesWriterwErrs(tMasterFileLocation, tSystematicsFileLocation, tFitInfoTString);
  vector<vector<FitParameter*> > tTest = tFitValWriterwErrs->ReadAllParameters(tAnType);
//-------------------------------------------------------------------------------
  cout << "DONE" << endl;
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
  return 0;
}
