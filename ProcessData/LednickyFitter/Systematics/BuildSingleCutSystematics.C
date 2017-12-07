#include "FitSystematicAnalysis.h"
class FitSystematicAnalysis;

SystematicsFileInfo GetFileInfo(int aNumber)
{
  SystematicsFileInfo gInfoLamKch1;
    gInfoLamKch1.resultsDate = "20161103";
    gInfoLamKch1.dirNameModifierBase1 = "_ALam_minNegDaughterToPrimVertex_";
    gInfoLamKch1.modifierValues1 = vector<double> {0.05,0.10,0.20};
    gInfoLamKch1.dirNameModifierBase2 = "";
    gInfoLamKch1.modifierValues2 = vector<double> {};
    gInfoLamKch1.allCentralities = true;

  SystematicsFileInfo gInfoLamKch2;
    gInfoLamKch2.resultsDate = "20161103";
    gInfoLamKch2.dirNameModifierBase1 = "_ALam_minPosDaughterToPrimVertex_";
    gInfoLamKch2.modifierValues1 = vector<double> {0.20, 0.30, 0.40};
    gInfoLamKch2.dirNameModifierBase2 = "";
    gInfoLamKch2.modifierValues2 = vector<double> {};
    gInfoLamKch2.allCentralities = true;

  SystematicsFileInfo gInfoLamKch3;
    gInfoLamKch3.resultsDate = "20161109";
    gInfoLamKch3.dirNameModifierBase1 = "_ALLTRACKS_maxImpactXY_";
    gInfoLamKch3.modifierValues1 = vector<double> {1.92,2.4,2.88};
    gInfoLamKch3.dirNameModifierBase2 = "";
    gInfoLamKch3.modifierValues2 = vector<double> {};
    gInfoLamKch3.allCentralities = true;

  SystematicsFileInfo gInfoLamKch4;
    gInfoLamKch4.resultsDate = "20161109";
    gInfoLamKch4.dirNameModifierBase1 = "_ALLTRACKS_maxImpactZ_";
    gInfoLamKch4.modifierValues1 = vector<double> {2.4,3.0,3.6};
    gInfoLamKch4.dirNameModifierBase2 = "";
    gInfoLamKch4.modifierValues2 = vector<double> {};
    gInfoLamKch4.allCentralities = true;

  SystematicsFileInfo gInfoLamKch5;
    gInfoLamKch5.resultsDate = "20161026";
    gInfoLamKch5.dirNameModifierBase1 = "_ALLV0S_maxDcaV0Daughters_";
    gInfoLamKch5.modifierValues1 = vector<double> {0.30,0.40,0.50};
    gInfoLamKch5.dirNameModifierBase2 = "";
    gInfoLamKch5.modifierValues2 = vector<double> {};
    gInfoLamKch5.allCentralities = true;

  SystematicsFileInfo gInfoLamKch6;
    gInfoLamKch6.resultsDate = "20161025";
    gInfoLamKch6.dirNameModifierBase1 = "_ALLV0S_minInvMassReject_";
    gInfoLamKch6.modifierValues1 = vector<double> {0.494614, 0.492614, 0.488614, 0.482614};
    gInfoLamKch6.dirNameModifierBase2 = "_ALLV0S_maxInvMassReject_";
    gInfoLamKch6.modifierValues2 = vector<double> {0.500614, 0.502614, 0.506614, 0.512614};
    gInfoLamKch6.allCentralities = false;

  SystematicsFileInfo gInfoLamKch7;
    gInfoLamKch7.resultsDate = "20161026";
    gInfoLamKch7.dirNameModifierBase1 = "_CLAM_maxDcaV0_";
    gInfoLamKch7.modifierValues1 = vector<double> {0.40, 0.50, 0.60};
    gInfoLamKch7.dirNameModifierBase2 = "";
    gInfoLamKch7.modifierValues2 = vector<double> {};
    gInfoLamKch7.allCentralities = true;

  SystematicsFileInfo gInfoLamKch8;
    gInfoLamKch8.resultsDate = "20161031";
    gInfoLamKch8.dirNameModifierBase1 = "_CLAM_minCosPointingAngle_";
    gInfoLamKch8.modifierValues1 = vector<double> {0.9992, 0.9993, 0.9994};
    gInfoLamKch8.dirNameModifierBase2 = "";
    gInfoLamKch8.modifierValues2 = vector<double> {};
    gInfoLamKch8.allCentralities = true;

  SystematicsFileInfo gInfoLamKch9;
    gInfoLamKch9.resultsDate = "20161103";
    gInfoLamKch9.dirNameModifierBase1 = "_Lam_minNegDaughterToPrimVertex_";
    gInfoLamKch9.modifierValues1 = vector<double> {0.2, 0.3, 0.4};
    gInfoLamKch9.dirNameModifierBase2 = "";
    gInfoLamKch9.modifierValues2 = vector<double> {};
    gInfoLamKch9.allCentralities = true;

  SystematicsFileInfo gInfoLamKch10;
    gInfoLamKch10.resultsDate = "20161103";
    gInfoLamKch10.dirNameModifierBase1 = "_Lam_minPosDaughterToPrimVertex_";
    gInfoLamKch10.modifierValues1 = vector<double> {0.05, 0.1, 0.2};
    gInfoLamKch10.dirNameModifierBase2 = "";
    gInfoLamKch10.modifierValues2 = vector<double> {};
    gInfoLamKch10.allCentralities = true;

  SystematicsFileInfo gInfoLamKch11;
    gInfoLamKch11.resultsDate = "20161106";
    gInfoLamKch11.dirNameModifierBase1 = "_minAvgSepTrackPos_";
    gInfoLamKch11.modifierValues1 = vector<double> {7.0, 8.0, 9.0};
    gInfoLamKch11.dirNameModifierBase2 = "";
    gInfoLamKch11.modifierValues2 = vector<double> {};
    gInfoLamKch11.allCentralities = true;

  SystematicsFileInfo gInfoLamKch12;
    gInfoLamKch12.resultsDate = "20161108";
    gInfoLamKch12.dirNameModifierBase1 = "_minAvgSepTrackPos_";
    gInfoLamKch12.modifierValues1 = vector<double> {7.5, 8.0, 8.5};
    gInfoLamKch12.dirNameModifierBase2 = "";
    gInfoLamKch12.modifierValues2 = vector<double> {};
    gInfoLamKch12.allCentralities = true;

  //---------------------------------------------------------------------- 


  SystematicsFileInfo gInfoLamK01;
    gInfoLamK01.resultsDate = "20161103";
    gInfoLamK01.dirNameModifierBase1 = "_ALam_minNegDaughterToPrimVertex_";
    gInfoLamK01.modifierValues1 = vector<double> {0.05,0.10,0.20};
    gInfoLamK01.dirNameModifierBase2 = "";
    gInfoLamK01.modifierValues2 = vector<double> {};
    gInfoLamK01.allCentralities = true;

  SystematicsFileInfo gInfoLamK02;
    gInfoLamK02.resultsDate = "20161103";
    gInfoLamK02.dirNameModifierBase1 = "_ALam_minPosDaughterToPrimVertex_";
    gInfoLamK02.modifierValues1 = vector<double> {0.20,0.30,0.40};
    gInfoLamK02.dirNameModifierBase2 = "";
    gInfoLamK02.modifierValues2 = vector<double> {};
    gInfoLamK02.allCentralities = true;

  SystematicsFileInfo gInfoLamK03;
    gInfoLamK03.resultsDate = "20161026";
    gInfoLamK03.dirNameModifierBase1 = "_CLAM_maxDcaV0_";
    gInfoLamK03.modifierValues1 = vector<double> {0.40,0.50,0.60};
    gInfoLamK03.dirNameModifierBase2 = "";
    gInfoLamK03.modifierValues2 = vector<double> {};
    gInfoLamK03.allCentralities = true;

  SystematicsFileInfo gInfoLamK04;
    gInfoLamK04.resultsDate = "20161026";
    gInfoLamK04.dirNameModifierBase1 = "_CLAM_maxDcaV0Daughters_";
    gInfoLamK04.modifierValues1 = vector<double> {0.30,0.40,0.50};
    gInfoLamK04.dirNameModifierBase2 = "";
    gInfoLamK04.modifierValues2 = vector<double> {};
    gInfoLamK04.allCentralities = true;

  SystematicsFileInfo gInfoLamK05;
    gInfoLamK05.resultsDate = "20161031";
    gInfoLamK05.dirNameModifierBase1 = "_CLAM_minCosPointingAngle_";
    gInfoLamK05.modifierValues1 = vector<double> {0.9992, 0.9993, 0.9994};
    gInfoLamK05.dirNameModifierBase2 = "";
    gInfoLamK05.modifierValues2 = vector<double> {};
    gInfoLamK05.allCentralities = true;

  SystematicsFileInfo gInfoLamK06;
    gInfoLamK06.resultsDate = "20161026";
    gInfoLamK06.dirNameModifierBase1 = "_K0s_maxDcaV0_";
    gInfoLamK06.modifierValues1 = vector<double> {0.20,0.30,0.40};
    gInfoLamK06.dirNameModifierBase2 = "";
    gInfoLamK06.modifierValues2 = vector<double> {};
    gInfoLamK06.allCentralities = true;

  SystematicsFileInfo gInfoLamK07;
    gInfoLamK07.resultsDate = "20161026";
    gInfoLamK07.dirNameModifierBase1 = "_K0s_maxDcaV0Daughters_";
    gInfoLamK07.modifierValues1 = vector<double> {0.20,0.30,0.40};
    gInfoLamK07.dirNameModifierBase2 = "";
    gInfoLamK07.modifierValues2 = vector<double> {};
    gInfoLamK07.allCentralities = true;

  SystematicsFileInfo gInfoLamK08;
    gInfoLamK08.resultsDate = "20161102";
    gInfoLamK08.dirNameModifierBase1 = "_K0s_minCosPointingAngle_";
    gInfoLamK08.modifierValues1 = vector<double> {0.9992, 0.9993, 0.9994};
    gInfoLamK08.dirNameModifierBase2 = "";
    gInfoLamK08.modifierValues2 = vector<double> {};
    gInfoLamK08.allCentralities = true;

  SystematicsFileInfo gInfoLamK09;
    gInfoLamK09.resultsDate = "20161025";
    gInfoLamK09.dirNameModifierBase1 = "_K0s_minInvMassReject_";
    gInfoLamK09.modifierValues1 = vector<double> {1.112683, 1.110683, 1.106683, 1.100683};
    gInfoLamK09.dirNameModifierBase2 = "_K0s_maxInvMassReject_";
    gInfoLamK09.modifierValues2 = vector<double> {1.118683, 1.120683, 1.124683, 1.130683};
    gInfoLamK09.allCentralities = false;

  SystematicsFileInfo gInfoLamK010;
    gInfoLamK010.resultsDate = "20161103";
    gInfoLamK010.dirNameModifierBase1 = "_K0s_minNegDaughterToPrimVertex_";
    gInfoLamK010.modifierValues1 = vector<double> {0.2, 0.3, 0.4};
    gInfoLamK010.dirNameModifierBase2 = "";
    gInfoLamK010.modifierValues2 = vector<double> {};
    gInfoLamK010.allCentralities = true;

  SystematicsFileInfo gInfoLamK011;
    gInfoLamK011.resultsDate = "20161103";
    gInfoLamK011.dirNameModifierBase1 = "_K0s_minPosDaughterToPrimVertex_";
    gInfoLamK011.modifierValues1 = vector<double> {0.2, 0.3, 0.4};
    gInfoLamK011.dirNameModifierBase2 = "";
    gInfoLamK011.modifierValues2 = vector<double> {};
    gInfoLamK011.allCentralities = true;

  SystematicsFileInfo gInfoLamK012;
    gInfoLamK012.resultsDate = "20161103";
    gInfoLamK012.dirNameModifierBase1 = "_Lam_minNegDaughterToPrimVertex_";
    gInfoLamK012.modifierValues1 = vector<double> {0.2, 0.3, 0.4};
    gInfoLamK012.dirNameModifierBase2 = "";
    gInfoLamK012.modifierValues2 = vector<double> {};
    gInfoLamK012.allCentralities = true;

  SystematicsFileInfo gInfoLamK013;
    gInfoLamK013.resultsDate = "20161103";
    gInfoLamK013.dirNameModifierBase1 = "_Lam_minPosDaughterToPrimVertex_";
    gInfoLamK013.modifierValues1 = vector<double> {0.05, 0.1, 0.2};
    gInfoLamK013.dirNameModifierBase2 = "";
    gInfoLamK013.modifierValues2 = vector<double> {};
    gInfoLamK013.allCentralities = true;

  SystematicsFileInfo gInfoLamK014;
    gInfoLamK014.resultsDate = "20161106";
    gInfoLamK014.dirNameModifierBase1 = "_minAvgSepNegNeg_";
    gInfoLamK014.modifierValues1 = vector<double> {5.0, 6.0, 7.0};
    gInfoLamK014.dirNameModifierBase2 = "";
    gInfoLamK014.modifierValues2 = vector<double> {};
    gInfoLamK014.allCentralities = true;

  SystematicsFileInfo gInfoLamK015;
    gInfoLamK015.resultsDate = "20161108";
    gInfoLamK015.dirNameModifierBase1 = "_minAvgSepNegNeg_";
    gInfoLamK015.modifierValues1 = vector<double> {5.5, 6.0, 6.5};
    gInfoLamK015.dirNameModifierBase2 = "";
    gInfoLamK015.modifierValues2 = vector<double> {};
    gInfoLamK015.allCentralities = true;

  SystematicsFileInfo gInfoLamK016;
    gInfoLamK016.resultsDate = "20161106";
    gInfoLamK016.dirNameModifierBase1 = "_minAvgSepPosPos_";
    gInfoLamK016.modifierValues1 = vector<double> {5.0, 6.0, 7.0};
    gInfoLamK016.dirNameModifierBase2 = "";
    gInfoLamK016.modifierValues2 = vector<double> {};
    gInfoLamK016.allCentralities = true;

  SystematicsFileInfo gInfoLamK017;
    gInfoLamK017.resultsDate = "20161108";
    gInfoLamK017.dirNameModifierBase1 = "_minAvgSepPosPos_";
    gInfoLamK017.modifierValues1 = vector<double> {5.5, 6.0, 6.5};
    gInfoLamK017.dirNameModifierBase2 = "";
    gInfoLamK017.modifierValues2 = vector<double> {};
    gInfoLamK017.allCentralities = true;


  //----------------------------------------------------------------------

  if(aNumber==1) return gInfoLamKch1;
  else if(aNumber==2) return gInfoLamKch2;
  else if(aNumber==3) return gInfoLamKch3;
  else if(aNumber==4) return gInfoLamKch4;
  else if(aNumber==5) return gInfoLamKch5;
  else if(aNumber==6) return gInfoLamKch6;
  else if(aNumber==7) return gInfoLamKch7;
  else if(aNumber==8) return gInfoLamKch8;
  else if(aNumber==9) return gInfoLamKch9;
  else if(aNumber==10) return gInfoLamKch10;
  else if(aNumber==11) return gInfoLamKch11;
  else if(aNumber==12) return gInfoLamKch12;

  else if(aNumber==-1) return gInfoLamK01;
  else if(aNumber==-2) return gInfoLamK02;
  else if(aNumber==-3) return gInfoLamK03;
  else if(aNumber==-4) return gInfoLamK04;
  else if(aNumber==-5) return gInfoLamK05;
  else if(aNumber==-6) return gInfoLamK06;
  else if(aNumber==-7) return gInfoLamK07;
  else if(aNumber==-8) return gInfoLamK08;
  else if(aNumber==-9) return gInfoLamK09;
  else if(aNumber==-10) return gInfoLamK010;
  else if(aNumber==-11) return gInfoLamK011;
  else if(aNumber==-12) return gInfoLamK012;
  else if(aNumber==-13) return gInfoLamK013;
  else if(aNumber==-14) return gInfoLamK014;
  else if(aNumber==-15) return gInfoLamK015;
  else if(aNumber==-16) return gInfoLamK016;
  else if(aNumber==-17) return gInfoLamK017;

  else
  {
    cout << "ERROR: SystematicsFileInfo GetFileInfo" << endl;
    assert(0);
    return gInfoLamKch1;
  }
}

int main(int argc, char **argv) 
{
//  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  bool bPlayCompletionBeep = true;
//-----------------------------------------------------------------------------
  AnalysisType tAnType = kLamKchP;
  CentralityType tCentralityType = kMB;
  FitGeneratorType tFitGeneratorType = kPairwConj;
  bool tShareLambdaParameters = false;
  bool tAllShareSingleLambdaParam = false;

  bool ApplyMomResCorrection = true;
  bool ApplyNonFlatBackgroundCorrection = true;
  NonFlatBgdFitType tNonFlatBgdFitType = kLinear;

  IncludeResidualsType tIncludeResidualsType = kInclude10Residuals; 
  ChargedResidualsType tChargedResidualsType = kUseXiDataAndCoulombOnlyInterp;
  ResPrimMaxDecayType tResPrimMaxDecayType = k5fm;

  bool bWriteToFile = true;
  bool bSaveImages = true;

  SystematicsFileInfo tFileInfo = GetFileInfo(1);
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

  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Systematics/Results_%s_Systematics%s",tGeneralAnTypeName.Data(),tDirNameModifierBase1.Data());
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

  tFitSysAn->RunAllFits(bSaveImages, bWriteToFile);

cout << "DONE" << endl;
if(bPlayCompletionBeep) system("( speaker-test -t sine -f 1000 )& pid=$! ; sleep 0.5s ; kill -9 $pid");
//-------------------------------------------------------------------------------
//  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
