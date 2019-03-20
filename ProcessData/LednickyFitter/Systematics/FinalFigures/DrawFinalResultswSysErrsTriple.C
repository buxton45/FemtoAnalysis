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

  vector<TString> tStatOnlyTags = {"", "_StatOnly"};
  vector<TString> tVerticalTags = {"", "_Vertical"};

  bool bDrawStatOnly = false;
  bool bDrawPredictions = true;
  bool bDrawVertical = false;

  IncludeResidualsType tIncResType = kInclude3Residuals;
  ResPrimMaxDecayType tResPrimMaxDecayType = k10fm;





  //--------------------------------------------------------------------------
//TRIPLE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  vector<NonFlatBgdFitType> tNonFlatBgdFitTypes{kLinear, kLinear,
                                                kPolynomial, kPolynomial, kPolynomial, kPolynomial};

  TString tFitInfoTString = 
                                                                 FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypes, 
                                                                                                      tIncResType, tResPrimMaxDecayType, 
                                                                                                      kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                      false, false, false, false, false, 
                                                                                                      true, false, false, true, 
                                                                                                      true, true);


  vector<FitValWriterInfo> tFVWIVec_Comp3An_Triple = {
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString, 
                                                                                 " (Suppress Markers)", tColorLamKchP, 20, tMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString, 
                                                                                 " (Suppress Markers)", tColorLamKchM, 20, tMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString, 
                                                                                 " (Suppress Markers)", tColorLamK0, 20, tMarkerSize, false, true)};

  vector<FitValWriterInfo> tFVWIVec = tFVWIVec_Comp3An_Triple;
  TString tCanNameMod = TString("_Comp3An_Triple");


  //--------------------------------------------------------------------------
  TString tSystematicsFileLocation = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/%s/Systematics/FinalFitSystematics_wFitRangeSys%s.txt", tResultsDate.Data(), tFitInfoTString.Data(), tFitInfoTString.Data());
  TString tSaveDirBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/%s/Comparisons/", tResultsDate.Data(), tFitInfoTString.Data());


  TCanvas* tCanAll2Panel = CompareAll2Panel(tFVWIVec, tSystematicsFileLocation, tSystematicsFileLocation, bDrawPredictions, tCanNameMod, bDrawStatOnly, bDrawVertical);
  if(bSaveFigures) tCanAll2Panel->SaveAs(TString::Format("%sFinalResults_Comp3An%s%s.%s", tSaveDirBase.Data(), tStatOnlyTags[bDrawStatOnly].Data(), tVerticalTags[bDrawVertical].Data(), tSaveFileType.Data()));



//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.



  cout << "DONE" << endl;
  return 0;
}








