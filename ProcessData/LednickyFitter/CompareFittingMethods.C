#include "CompareFittingMethods.cxx"

//---------------------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************************
//---------------------------------------------------------------------------------------------------------------------------------
int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  bool bDrawPredictions = false;

  bool bSaveFigures = false;
  TString tSaveFileType = "pdf";
  TString tSaveDir = "/home/jesse/Analysis/FemtoAnalysis/AnalysisNotes/7_ResultsAndDiscussion/Figures/New/";

  vector<FitValWriterInfo> tFVWIVec;
  TString tCanNameMod = "";


//---------------------------------------------------------------------



  TString tFitInfoTString_LamKch_3Res_PolyBgd_10fm = FitValuesWriter::BuildFitInfoTString(true, true, kPolynomial, 
                                                                                         kInclude3Residuals, k10fm, 
                                                                                         kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                         false, false, false, false, false, 
                                                                                         true, false, false, true, 
                                                                                         true, true);

  TString tFitInfoTString_LamKch_3Res_LinrBgd_10fm = FitValuesWriter::BuildFitInfoTString(true, true, kLinear, 
                                                                                         kInclude3Residuals, k10fm, 
                                                                                         kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                         false, false, false, false, false, 
                                                                                         true, false, false, true, 
                                                                                         true, true);

  TString tFitInfoTString_LamKch_3Res_StavCf_10fm = FitValuesWriter::BuildFitInfoTString(true, false, kPolynomial, 
                                                                                         kInclude3Residuals, k10fm, 
                                                                                         kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                         true, false, false, false, false, 
                                                                                         true, false, false, true, 
                                                                                         true, true);

  //----------

  TString tFitInfoTString_LamK0_3Res_LinrBgd_10fm = FitValuesWriter::BuildFitInfoTString(true, true, kLinear, 
                                                                                    kInclude3Residuals, k10fm, 
                                                                                    kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                    false, false, false, false, false, 
                                                                                    false, true, false, false, 
                                                                                    false, false);

  TString tFitInfoTString_LamK0_3Res_PolyBgd_10fm = FitValuesWriter::BuildFitInfoTString(true, true, kPolynomial, 
                                                                                    kInclude3Residuals, k10fm, 
                                                                                    kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                    false, false, false, false, false, 
                                                                                    false, true, false, false, 
                                                                                    false, false);

  TString tFitInfoTString_LamK0_3Res_StavCf_10fm = FitValuesWriter::BuildFitInfoTString(true, false, kLinear, 
                                                                                    kInclude3Residuals, k10fm, 
                                                                                    kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                    true, false, false, false, false, 
                                                                                    false, true, false, false, 
                                                                                    false, false);
//---------------------------------------------------------------------
//Triple fitter!

  vector<NonFlatBgdFitType> tNonFlatBgdFitTypes{kLinear, kLinear,
                                                kPolynomial, kPolynomial, kPolynomial, kPolynomial};

  TString tFitInfoTString_3Res_10fm_Triple_LamK0LinearLamKchPoly = FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypes, 
                                                                                         kInclude3Residuals, k10fm, 
                                                                                         kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                         false, false, false, false, false, 
                                                                                         true, false, false, true, 
                                                                                         true, true);

//---------------------------------------------------------------------


  vector<FitValWriterInfo> tFVWIVec_Comp3An_3Res_10fm = {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_3Res_PolyBgd_10fm, 
                                                                         "3 Residuals (Suppress Markers)", tColorLamKchP, 20, tMarkerSize, true), 
                                                        FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_3Res_PolyBgd_10fm, 
                                                                         "#LambdaK^{#pm}: Share R and #lambda, Poly. Bgd (Suppress Markers)", tColorLamKchM, 20, tMarkerSize, true), 
                                                        FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_3Res_LinrBgd_10fm, 
                                                                         "#LambdaK^{0}_{S}: Single #lambda, Linr. Bgd (Suppress Markers)", tColorLamK0, 20, tMarkerSize, false)};
//  tFVWIVec = tFVWIVec_Comp3An_3Res_10fm;
//  tCanNameMod = TString("_Comp3An_3Res_10fm");
//---------------------------------------------------------------------


  vector<FitValWriterInfo> tFVWIVec_Comp3An_WithBgdVsStav_10fm = {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_3Res_PolyBgd_10fm, 
                                                                                   "3 Residuals (Suppress Markers)", tColorLamKchP, 20, tMarkerSize, true), 
                                                                  FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_3Res_PolyBgd_10fm, 
                                                                                   "#LambdaK^{#pm}: Share R and #lambda, Poly. Bgd", tColorLamKchM, 20, tMarkerSize, true), 
                                                                  FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_3Res_LinrBgd_10fm, 
                                                                                   "#LambdaK^{0}_{S}: Single #lambda, Linr. Bgd (Suppress Markers)", tColorLamK0, 20, tMarkerSize, false),
                                                                  FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_3Res_StavCf_10fm, 
                                                                                   "3 Residuals (Suppress Markers)", tColorLamKchP, 34, tMarkerSize, true),
                                                                  FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_3Res_StavCf_10fm, 
                                                                                   "Stav. Cf", tColorLamKchM, 34, tMarkerSize, true), 
                                                                  FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_3Res_StavCf_10fm, 
                                                                                   "#LambdaK^{0}_{S}: Single #lambda, Linr. Bgd (Suppress Markers)", tColorLamK0, 34, tMarkerSize, false)};
//  tFVWIVec = tFVWIVec_Comp3An_WithBgdVsStav_10fm;
//  tCanNameMod = TString("_Comp3An_WithBgdVsStav_10fm");
//---------------------------------------------------------------------


  vector<FitValWriterInfo> tFVWIVec_Comp3An_LinrPolyStav_10fm = {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_3Res_PolyBgd_10fm, 
                                                                                  "3 Residuals (Suppress Markers)", tColorLamKchP, 20, tMarkerSize, true), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_3Res_PolyBgd_10fm, 
                                                                                  "Poly. Bgd", tColorLamKchM, 20, tMarkerSize, true), 
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_3Res_PolyBgd_10fm, 
                                                                                  "Poly. Bgd", tColorLamK0, 20, tMarkerSize, false),

                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_3Res_LinrBgd_10fm, 
                                                                                  "Linr. Bgd", tColorLamKchP, 21, tMarkerSize, true), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_3Res_LinrBgd_10fm, 
                                                                                  "Linr. Bgd", tColorLamKchM, 21, tMarkerSize, true), 
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_3Res_LinrBgd_10fm, 
                                                                                  "Linr. Bgd", tColorLamK0, 21, tMarkerSize, false),

                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_3Res_StavCf_10fm, 
                                                                                  "Stav. Cf", tColorLamKchP, 34, tMarkerSize, true),
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_3Res_StavCf_10fm, 
                                                                                  "Stav. Cf", tColorLamKchM, 34, tMarkerSize, true), 
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_3Res_StavCf_10fm, 
                                                                                  "Stav. Cf", tColorLamK0, 34, tMarkerSize, false)};
//  tFVWIVec = tFVWIVec_Comp3An_LinrPolyStav_10fm;
//  tCanNameMod = TString("_Comp3An_LinrPolyStav_10fm");
//---------------------------------------------------------------------


  vector<FitValWriterInfo> tFVWIVec_Comp3An_TripleVsDualie = {
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_3Res_PolyBgd_10fm, 
                                                                                 "#LambdaK^{#pm} and #LambdaK^{0}_{S} fit separate", tColorLamKchP, 20, tMarkerSize, true), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_3Res_PolyBgd_10fm, 
                                                                                 "#LambdaK^{#pm} and #LambdaK^{0}_{S} fit separate", tColorLamKchM, 20, tMarkerSize, true), 
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_3Res_LinrBgd_10fm, 
                                                                                 "#LambdaK^{#pm} and #LambdaK^{0}_{S} fit separate", tColorLamK0, 20, tMarkerSize, false),

                                                                 //Outline the points
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_3Res_PolyBgd_10fm, 
                                                                                 "#LambdaK^{#pm} and #LambdaK^{0}_{S} fit separate", kViolet, 24, tMarkerSize, true), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_3Res_PolyBgd_10fm, 
                                                                                 "#LambdaK^{#pm} and #LambdaK^{0}_{S} fit separate", kViolet, 24, tMarkerSize, true),

                                                                 //-------------------------------------------------

                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_3Res_10fm_Triple_LamK0LinearLamKchPoly, 
                                                                                 "#LambdaK^{#pm} and #LambdaK^{0}_{S} fit together", tColorLamKchP, 29, tMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_3Res_10fm_Triple_LamK0LinearLamKchPoly, 
                                                                                 "#LambdaK^{#pm} and #LambdaK^{0}_{S} fit together", tColorLamKchM, 29, tMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_3Res_10fm_Triple_LamK0LinearLamKchPoly, 
                                                                                 "#LambdaK^{#pm} and #LambdaK^{0}_{S} fit together", tColorLamK0, 29, tMarkerSize, false, true),

                                                                 //Outline the points
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_3Res_10fm_Triple_LamK0LinearLamKchPoly, 
                                                                                 "#LambdaK^{#pm} and #LambdaK^{0}_{S} fit together", kGreen, 30, tMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_3Res_10fm_Triple_LamK0LinearLamKchPoly, 
                                                                                 "#LambdaK^{#pm} and #LambdaK^{0}_{S} fit together", kGreen, 30, tMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_3Res_10fm_Triple_LamK0LinearLamKchPoly, 
                                                                                 "#LambdaK^{#pm} and #LambdaK^{0}_{S} fit together", kGreen, 30, tMarkerSize, false, true)};

  tFVWIVec = tFVWIVec_Comp3An_TripleVsDualie;
  tCanNameMod = TString("_Comp3An_TripleVsDualie");

//---------------------------------------------------------------------

//  tFVWIVec = tFVWIVec_SepR_Seplam;
//  tCanNameMod = TString("_SepR_Seplam");

//  tFVWIVec = tFVWIVec_SepR_Sharelam;
//  tCanNameMod = TString("_SepR_Sharelam");

//  tFVWIVec = tFVWIVec_ShareR_Sharelam;
//  tCanNameMod = TString("_ShareR_Sharelam");

//  tFVWIVec = tFVWIVec_ShareR_SharelamConj;
//  tCanNameMod = TString("_ShareR_SharelamConj");

//  tFVWIVec = tFVWIVec_SharevsSepR;
//  tCanNameMod = TString("_SharevsSepR");

//---------------------------------------------------------------------

//  tFVWIVec = tFVWIVec_FreevsFixlam_SepR;
//  tCanNameMod = TString("_FreevsFixlam_SepR");

//  tFVWIVec = tFVWIVec_FreevsFixlam_SepR_NoStav;
//  tCanNameMod = TString("_FreevsFixlam_SepR_NoStav");

//  tFVWIVec = tFVWIVec_FreevsFixlam_SepR_PolyBgd;
//  tCanNameMod = TString("_FreevsFixlam_SepR_PolyBgd");

//  tFVWIVec = tFVWIVec_FreevsFixlam_SepR_LinrBgd;
//  tCanNameMod = TString("_FreevsFixlam_SepR_LinrBgd");

//  tFVWIVec = tFVWIVec_FreevsFixlam_SepR_StavCf_NoBgd;
//  tCanNameMod = TString("_FreevsFixlam_SepR_StavCf_NoBgd");

//---------------------------------------------------------------------

//  tFVWIVec = tFVWIVec_FreevsFixlam_ShareR;
//  tCanNameMod = TString("_FreevsFixlam_ShareR");

//  tFVWIVec = tFVWIVec_FreevsFixlam_ShareR_NoStav;
//  tCanNameMod = TString("_FreevsFixlam_ShareR_NoStav");

//  tFVWIVec = tFVWIVec_FreevsFixlam_ShareR_PolyBgd;
//  tCanNameMod = TString("_FreevsFixlam_ShareR_PolyBgd");

//  tFVWIVec = tFVWIVec_FreevsFixlam_ShareR_LinrBgd;
//  tCanNameMod = TString("_FreevsFixlam_ShareR_LinrBgd");

//  tFVWIVec = tFVWIVec_FreevsFixlam_ShareR_StavCf_NoBgd;
//  tCanNameMod = TString("_FreevsFixlam_ShareR_StavCf_NoBgd");

  TCanvas* tCanLambdavsRadius = CompareLambdavsRadius(tFVWIVec, k0010, tCanNameMod);
  TCanvas* tCanImF0vsReF0 = CompareImF0vsReF0(tFVWIVec, bDrawPredictions, tCanNameMod);
  TCanvas* tCanAll = CompareAll(tFVWIVec, bDrawPredictions, tCanNameMod);



  if(bSaveFigures)
  {
    tCanAll->SaveAs(TString::Format("%s%s.%s", tSaveDir.Data(), tCanAll->GetName(), tSaveFileType.Data()));
  }








//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.



  cout << "DONE" << endl;
  return 0;
}








