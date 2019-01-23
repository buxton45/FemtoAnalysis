
#include "CompareFittingMethodswSysErrsForAnNote.h"





//---------------------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************************
//---------------------------------------------------------------------------------------------------------------------------------
int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  //TODO NOTE: If you want to build typical LamKchP vs LamKchM vs LamK0 plots, use CompareFittingMethodswSysErrs
  //           This macros is intended for Section 7.1.4...FitMethodComparisons

  bool bSaveFigures = false;
  TString tSaveFileType = "pdf";  //Needs to be pdf for systematics to be transparent!

  bool bDrawStatOnly = true;  //SHOULD ALWAYS BE TRUE IN THIS MACRO

  bool bDrawPredictions = false;


  TString tSaveDir = "/home/jesse/Analysis/FemtoAnalysis/AnalysisNotes/7_ResultsAndDiscussion/Figures/New/";

  //--------------------------------------------------------------------------

  vector<TString> tStatOnlyTags = {"", "_StatOnly"};

  vector<FitValWriterInfo> tFVWIVec;
  TString tCanNameMod = "";

  //--------------------------------------------------------------------------



//  tFVWIVec = tFVWIVec_Comp3An_3Res_10fm;
//  tCanNameMod = TString("_Comp3An_3Res_10fm");

//  tFVWIVec = tFVWIVec_Comp3An_WithBgdVsStav_10fm;
//  tCanNameMod = TString("_Comp3An_WithBgdVsStav_10fm");

//  tFVWIVec = tFVWIVec_Comp3An_LinrPolyStav_10fm;
//  tCanNameMod = TString("_Comp3An_LinrPolyStav_10fm");

//  tFVWIVec = tFVWIVec_CompNRes_Std;
//  tCanNameMod = TString("_CompNRes_Std");

//  tFVWIVec = tFVWIVec_CompFreevsFixedLam_Std;
//  tCanNameMod = TString("_CompFreevsFixedLam_Std");

//  tFVWIVec = tFVWIVec_CompFreevsFixedLam_SepR_Std;
//  tCanNameMod = TString("_CompFreevsFixedLam_SepR_Std");

//  tFVWIVec = tFVWIVec_CompSharedvsUniqueLam_Std;
//  tCanNameMod = TString("_CompSharedvsUniqueLam_Std");

//  tFVWIVec = tFVWIVec_CompSharedvsUniqueLam_SepR_Std;
//  tCanNameMod = TString("_CompSharedvsUniqueLam_SepR_Std");

  tFVWIVec = tFVWIVec_CompSharedvsSepRLam_Std;
  tCanNameMod = TString("_CompSharedvsSepRLam_Std");

//  tFVWIVec = tFVWIVec_CompSharedvsSepR_ShareLamConj_Std;
//  tCanNameMod = TString("_CompSharedvsSepR_ShareLamConj_Std");

  //--------------------------------------------------------------------------

  TCanvas* tCanLambdavsRadius = CompareLambdavsRadius(tFVWIVec, TString(""), TString(""), k0010, tCanNameMod, false, false, bDrawStatOnly);
  TCanvas* tCanImF0vsReF0 =         CompareImF0vsReF0(tFVWIVec, TString(""), TString(""), bDrawPredictions, tCanNameMod, false, false, bDrawStatOnly);
  TCanvas* tCanAll =                       CompareAll(tFVWIVec, TString(""), TString(""), bDrawPredictions, tCanNameMod, bDrawStatOnly);



  if(bSaveFigures)
  {
    tCanAll->SaveAs(TString::Format("%s%s%s.%s", tSaveDir.Data(), tCanAll->GetName(), tStatOnlyTags[bDrawStatOnly].Data(), tSaveFileType.Data()));
  }








//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.



  cout << "DONE" << endl;
  return 0;
}








