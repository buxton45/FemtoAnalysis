#include "CompareFittingMethodswSysErrs.cxx"


//---------------------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************************
//---------------------------------------------------------------------------------------------------------------------------------
int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  bool bSaveFigures = false;
  TString tSaveFileType = "pdf";  //Needs to be pdf for systematics to be transparent!

  bool bDrawStatOnly = false;
  bool bLamKchCombined = true;

  bool bDrawPredictions = true;

  IncludeResidualsType tIncResType = kInclude3Residuals;
  ResPrimMaxDecayType tResPrimMaxDecayType = k10fm;


  TString tFitInfoTString_LamKch = FitValuesLatexTableHelperWriter::GetFitInfoTStringFromTwoLetterID_LamKch("Ea", tIncResType, tResPrimMaxDecayType);
//  TString tFitInfoTString_LamK0 = FitValuesLatexTableHelperWriter::GetFitInfoTStringFromTwoLetterID_LamK0("Aa", tIncResType, tResPrimMaxDecayType);
  TString tFitInfoTString_LamK0 = FitValuesLatexTableHelperWriter::GetFitInfoTStringFromTwoLetterID_LamK0("Ab", tIncResType, tResPrimMaxDecayType);

  cout << "tFitInfoTString_LamKch = " << tFitInfoTString_LamKch << endl << endl;
  cout << "tFitInfoTString_LamK0 = " << tFitInfoTString_LamK0 << endl << endl; 


  TString tSystematicsFileLocation_LamKch = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/%s/Systematics/FinalFitSystematics_wFitRangeSys%s_cLamcKch.txt", tResultsDate.Data(), tFitInfoTString_LamKch.Data(), tFitInfoTString_LamKch.Data());
  TString tSystematicsFileLocation_LamK0 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/%s/Systematics/FinalFitSystematics_wFitRangeSys%s_cLamK0.txt", tResultsDate.Data(), tFitInfoTString_LamK0.Data(), tFitInfoTString_LamK0.Data());

  TString tSaveDirBase_LamKch = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/%s/Comparisons/", tResultsDate.Data(), tFitInfoTString_LamKch.Data());
  TString tSaveDirBase_LamK0 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/%s/Comparisons/", tResultsDate.Data(), tFitInfoTString_LamK0.Data());
  //--------------------------------------------------------------------------

  vector<TString> tStatOnlyTags = {"", "_StatOnly"};
  vector<TString> tResidualsTags = {"No Residuals (Suppress Markers)", "10 Residuals (Suppress Markers)", "3 Residuals (Suppress Markers)"};

  vector<FitValWriterInfo> tFVWIVec;
  TString tCanNameMod = "";

  //--------------------------------------------------------------------------

  TString tFitInfoTString_LamKch_3Res_PolyBgd_10fm = FitValuesWriter::BuildFitInfoTString(true, true, kPolynomial, 
                                                                                         tIncResType, tResPrimMaxDecayType, 
                                                                                         kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                         false, false, false, false, false, 
                                                                                         true, false, false, true, 
                                                                                         true, true);

  TString tFitInfoTString_LamK0_3Res_LinrBgd_10fm = FitValuesWriter::BuildFitInfoTString(true, true, kLinear, 
                                                                                    tIncResType, tResPrimMaxDecayType, 
                                                                                    kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                    false, false, false, false, false, 
                                                                                    false, true, false, false, 
                                                                                    false, false);


  vector<FitValWriterInfo> tFVWIVec_Comp3An_3Res_10fm = {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_3Res_PolyBgd_10fm, 
                                                                         tResidualsTags[tIncResType], tColorLamKchP, 20, tMarkerSize, bLamKchCombined), 
                                                        FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_3Res_PolyBgd_10fm, 
                                                                         "#LambdaK^{#pm}: Share R and #lambda, Poly. Bgd (Suppress Markers)", tColorLamKchM, 20, tMarkerSize, bLamKchCombined), 
                                                        FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_3Res_LinrBgd_10fm, 
                                                                         "#LambdaK^{0}_{S}: Single #lambda, Linr. Bgd (Suppress Markers)", tColorLamK0, 20, tMarkerSize, false)};
//  tFVWIVec = tFVWIVec_Comp3An_3Res_10fm;
//  tCanNameMod = TString("_Comp3An_3Res_10fm");
//  tCanNameMod = TString("_Comp3An");


  //--------------------------------------------------------------------------
//TRIPLE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  vector<NonFlatBgdFitType> tNonFlatBgdFitTypes{kLinear, kLinear,
                                                kPolynomial, kPolynomial, kPolynomial, kPolynomial};

  TString tFitInfoTString_3Res_10fm_Triple_LamK0LinearLamKchPoly = FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypes, 
                                                                                         kInclude3Residuals, k10fm, 
                                                                                         kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                         false, false, false, false, false, 
                                                                                         true, false, false, true, 
                                                                                         true, true);


  vector<FitValWriterInfo> tFVWIVec_Comp3An_Triple = {
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_3Res_10fm_Triple_LamK0LinearLamKchPoly, 
                                                                                 "#LambdaK^{#pm} and #LambdaK^{0}_{S} fit together", tColorLamKchP, 20, tMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_3Res_10fm_Triple_LamK0LinearLamKchPoly, 
                                                                                 "#LambdaK^{#pm} and #LambdaK^{0}_{S} fit together", tColorLamKchM, 20, tMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_3Res_10fm_Triple_LamK0LinearLamKchPoly, 
                                                                                 "#LambdaK^{#pm} and #LambdaK^{0}_{S} fit together", tColorLamK0, 20, tMarkerSize, false, true)};

  tFVWIVec = tFVWIVec_Comp3An_Triple;
  tCanNameMod = TString("_Comp3An_Triple");

  TCanvas* tCanAll2Panel = CompareAll2Panel(tFVWIVec, tSystematicsFileLocation_LamKch, tSystematicsFileLocation_LamK0, bDrawPredictions, tCanNameMod, bDrawStatOnly);
//  tCanAll2Panel->SaveAs("Comp3An_Triple.pdf");
  //--------------------------------------------------------------------------

//  tFVWIVec = tFVWIVec_CompNumRes_ShareR_PolyBgd;
//  tCanNameMod = TString("_CompNumRes_ShareR_PolyBgd");

//--------------------------------------------
//  tFVWIVec = tFVWIVec_Comp3An_3Res;
//  tCanNameMod = TString("_Comp3An_3Res");

//  tFVWIVec = tFVWIVec_Comp3An_10Res;
//  tCanNameMod = TString("_Comp3An_10Res");

//  tFVWIVec = tFVWIVec_Comp3An_NoRes;
//  tCanNameMod = TString("_Comp3An_NoRes");
//--------------------------------------------

//  tFVWIVec = tFVWIVec_CompNumRes;
//  tCanNameMod = TString("_CompNumRes");

//  tFVWIVec = tFVWIVec_CompBgdTreatment;
//  tCanNameMod = TString("_CompBgdTreatment");

//  tFVWIVec = tFVWIVec_CompFreevsFixedlam_ShareR;
//  tCanNameMod = TString("_CompFreevsFixedlam_ShareR");

//  tFVWIVec = tFVWIVec_CompFreevsFixedlam_SepR;
//  tCanNameMod = TString("_CompFreevsFixedlam_SepR");

//  tFVWIVec = tFVWIVec_CompFreevsFixedlam_SepR_Seplam;
//  tCanNameMod = TString("_CompFreevsFixedlam_SepR_Seplam");

//  tFVWIVec = tFVWIVec_CompSharesvsSepR;
//  tCanNameMod = TString("_CompSharedvsSepR");

//  tFVWIVec = tFVWIVec_CompSharelam_SepR;
//  tCanNameMod = TString("_CompSharelam_SepR");

//  tFVWIVec = tFVWIVec_CompSharelam_SharedR;
//  tCanNameMod = TString("_CompSharelam_SharedR");



  TCanvas* tCanLambdavsRadius = CompareLambdavsRadius(tFVWIVec, tSystematicsFileLocation_LamKch, tSystematicsFileLocation_LamK0, k0010, tCanNameMod, false, false, bDrawStatOnly);
  TCanvas* tCanImF0vsReF0 = CompareImF0vsReF0(tFVWIVec, tSystematicsFileLocation_LamKch, tSystematicsFileLocation_LamK0, bDrawPredictions, tCanNameMod, false, false, bDrawStatOnly);
  TCanvas* tCanAll = CompareAll(tFVWIVec, tSystematicsFileLocation_LamKch, tSystematicsFileLocation_LamK0, bDrawPredictions, tCanNameMod, bDrawStatOnly);



  if(bSaveFigures)
  {
    TString tSaveDirMod = TString::Format("LAMKCH%s_vs_LAMK0%s/", tFitInfoTString_LamKch.Data(), tFitInfoTString_LamK0.Data());

    TString tSaveDir_LamKch = TString::Format("%s%s", tSaveDirBase_LamKch.Data(), tSaveDirMod.Data());
    TString tSaveDir_LamK0 = TString::Format("%s%s", tSaveDirBase_LamK0.Data(), tSaveDirMod.Data());
      gSystem->mkdir(tSaveDir_LamKch, kTRUE);
      gSystem->mkdir(tSaveDir_LamK0, kTRUE);

    tCanAll->SaveAs(TString::Format("%s%s%s.%s", tSaveDir_LamKch.Data(), tCanAll->GetName(), tStatOnlyTags[bDrawStatOnly].Data(), tSaveFileType.Data()));
    tCanAll->SaveAs(TString::Format("%s%s%s.%s", tSaveDir_LamK0.Data(), tCanAll->GetName(), tStatOnlyTags[bDrawStatOnly].Data(), tSaveFileType.Data()));
  }








//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.



  cout << "DONE" << endl;
  return 0;
}








