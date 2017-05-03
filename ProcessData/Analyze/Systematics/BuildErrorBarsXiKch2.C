#include "SystematicAnalysis.h"
class SystematicAnalysis;

//Use data from LamKch analyses if not yet available for XiKch

SystematicsFileInfo GetFileInfo(int aNumber)
{
  SystematicsFileInfo gInfoXiKch1;
    gInfoXiKch1.resultsDate = "20170501";
    gInfoXiKch1.dirNameModifierBase1 = "_ALLTRACKS_maxImpactXY_";
    gInfoXiKch1.modifierValues1 = vector<double> {1.92,2.4,2.88};
    gInfoXiKch1.dirNameModifierBase2 = "";
    gInfoXiKch1.modifierValues2 = vector<double> {};
    gInfoXiKch1.allCentralities = true;

  SystematicsFileInfo gInfoXiKch2;
    gInfoXiKch2.resultsDate = "20170501";
    gInfoXiKch2.dirNameModifierBase1 = "_ALLTRACKS_maxImpactZ_";
    gInfoXiKch2.modifierValues1 = vector<double> {2.4,3.0,3.6};
    gInfoXiKch2.dirNameModifierBase2 = "";
    gInfoXiKch2.modifierValues2 = vector<double> {};
    gInfoXiKch2.allCentralities = true;


  SystematicsFileInfo gInfoXiKch3;
    gInfoXiKch3.resultsDate = "2017xxxx";  //Handled by LamKch
    gInfoXiKch3.dirNameModifierBase1 = "_ALLXIS_maxDcaV0Daughters_";
    gInfoXiKch3.modifierValues1 = vector<double> {0.30,0.40,0.50};
    gInfoXiKch3.dirNameModifierBase2 = "";
    gInfoXiKch3.modifierValues2 = vector<double> {};
    gInfoXiKch3.allCentralities = true;

  SystematicsFileInfo gInfoXiKch4;
    gInfoXiKch4.resultsDate = "20170429";
    gInfoXiKch4.dirNameModifierBase1 = "_ALLXIS_maxDcaXi_";
    gInfoXiKch4.modifierValues1 = vector<double> {0.20,0.30,0.40};
    gInfoXiKch4.dirNameModifierBase2 = "";
    gInfoXiKch4.modifierValues2 = vector<double> {};
    gInfoXiKch4.allCentralities = true;

  SystematicsFileInfo gInfoXiKch5;
    gInfoXiKch5.resultsDate = "20170429";
    gInfoXiKch5.dirNameModifierBase1 = "_ALLXIS_maxDcaXiDaughters_";
    gInfoXiKch5.modifierValues1 = vector<double> {0.20,0.30,0.40};
    gInfoXiKch5.dirNameModifierBase2 = "";
    gInfoXiKch5.modifierValues2 = vector<double> {};
    gInfoXiKch5.allCentralities = true;

  SystematicsFileInfo gInfoXiKch6;
    gInfoXiKch6.resultsDate = "2017xxxx";  //Handled by LamKch
    gInfoXiKch6.dirNameModifierBase1 = "_ALLXIS_minCosPointingAngleV0_";
    gInfoXiKch6.modifierValues1 = vector<double> {0.9992, 0.9993, 0.9994};
    gInfoXiKch6.dirNameModifierBase2 = "";
    gInfoXiKch6.modifierValues2 = vector<double> {};
    gInfoXiKch6.allCentralities = true;

  SystematicsFileInfo gInfoXiKch7;
    gInfoXiKch7.resultsDate = "20170429";
    gInfoXiKch7.dirNameModifierBase1 = "_ALLXIS_minCosPointingAngleV0toXi_";
    gInfoXiKch7.modifierValues1 = vector<double> {0.9992, 0.9993, 0.9994};
    gInfoXiKch7.dirNameModifierBase2 = "";
    gInfoXiKch7.modifierValues2 = vector<double> {};
    gInfoXiKch7.allCentralities = true;

  SystematicsFileInfo gInfoXiKch8;
    gInfoXiKch8.resultsDate = "20170429";
    gInfoXiKch8.dirNameModifierBase1 = "_ALLXIS_minCosPointingAngleXi_";
    gInfoXiKch8.modifierValues1 = vector<double> {0.9991, 0.9992, 0.9993};
    gInfoXiKch8.dirNameModifierBase2 = "";
    gInfoXiKch8.modifierValues2 = vector<double> {};
    gInfoXiKch8.allCentralities = true;

  SystematicsFileInfo gInfoXiKch9;
    gInfoXiKch9.resultsDate = "20170429";
    gInfoXiKch9.dirNameModifierBase1 = "_ALLXIS_minDcaV0_";
    gInfoXiKch9.modifierValues1 = vector<double> {0.05, 0.10, 0.20};
    gInfoXiKch9.dirNameModifierBase2 = "";
    gInfoXiKch9.modifierValues2 = vector<double> {};
    gInfoXiKch9.allCentralities = true;
/*
  SystematicsFileInfo gInfoXiKch9;
    gInfoXiKch9.resultsDate = "20170502";
    gInfoXiKch9.dirNameModifierBase1 = "_ALLXIS_minDcaV0_";
    gInfoXiKch9.modifierValues1 = vector<double> {0.10, 0.20, 0.30};
    gInfoXiKch9.dirNameModifierBase2 = "";
    gInfoXiKch9.modifierValues2 = vector<double> {};
    gInfoXiKch9.allCentralities = true;
*/
  SystematicsFileInfo gInfoXiKch10;
    gInfoXiKch10.resultsDate = "20170501";
    gInfoXiKch10.dirNameModifierBase1 = "_ALLXIS_minDcaXiBac_";
    gInfoXiKch10.modifierValues1 = vector<double> {0.02, 0.03, 0.04}; //TODO
    gInfoXiKch10.dirNameModifierBase2 = "";
    gInfoXiKch10.modifierValues2 = vector<double> {};
    gInfoXiKch10.allCentralities = true;
/*
  SystematicsFileInfo gInfoXiKch10;
    gInfoXiKch10.resultsDate = "20170502";
    gInfoXiKch10.dirNameModifierBase1 = "_ALLXIS_minDcaXiBac_";
    gInfoXiKch10.modifierValues1 = vector<double> {0.05, 0.1, 0.2}; //TODO
    gInfoXiKch10.dirNameModifierBase2 = "";
    gInfoXiKch10.modifierValues2 = vector<double> {};
    gInfoXiKch10.allCentralities = true;
*/
  SystematicsFileInfo gInfoXiKch11;
    gInfoXiKch11.resultsDate = "2017xxxx";  //Handled by LamKch
    gInfoXiKch11.dirNameModifierBase1 = "_AXi_minV0NegDaughterToPrimVertex_";
    gInfoXiKch11.modifierValues1 = vector<double> {0.05,0.10,0.20};
    gInfoXiKch11.dirNameModifierBase2 = "";
    gInfoXiKch11.modifierValues2 = vector<double> {};
    gInfoXiKch11.allCentralities = true;

  SystematicsFileInfo gInfoXiKch12;
    gInfoXiKch12.resultsDate = "20170429";
    gInfoXiKch12.dirNameModifierBase1 = "_AXi_minV0PosDaughterToPrimVertex_";
    gInfoXiKch12.modifierValues1 = vector<double> {0.20, 0.30, 0.40};
    gInfoXiKch12.dirNameModifierBase2 = "";
    gInfoXiKch12.modifierValues2 = vector<double> {};
    gInfoXiKch12.allCentralities = true;

  SystematicsFileInfo gInfoXiKch13;
    gInfoXiKch13.resultsDate = "2017xxxx";  //TODO
    gInfoXiKch13.dirNameModifierBase1 = "_minAvgSepTrackBacPion_";
    gInfoXiKch13.modifierValues1 = vector<double> {7.0, 8.0, 9.0};
    gInfoXiKch13.dirNameModifierBase2 = "";
    gInfoXiKch13.modifierValues2 = vector<double> {};
    gInfoXiKch13.allCentralities = true;

  SystematicsFileInfo gInfoXiKch14;
    gInfoXiKch14.resultsDate = "2017xxxx";  //Handled by LamKch
    gInfoXiKch14.dirNameModifierBase1 = "_minAvgSepTrackPos_";
    gInfoXiKch14.modifierValues1 = vector<double> {7.0, 8.0, 9.0};
    gInfoXiKch14.dirNameModifierBase2 = "";
    gInfoXiKch14.modifierValues2 = vector<double> {};
    gInfoXiKch14.allCentralities = true;


  SystematicsFileInfo gInfoXiKch15;
    gInfoXiKch15.resultsDate = "20170429";
    gInfoXiKch15.dirNameModifierBase1 = "_Xi_minV0NegDaughterToPrimVertex_";
    gInfoXiKch15.modifierValues1 = vector<double> {0.2, 0.3, 0.4};
    gInfoXiKch15.dirNameModifierBase2 = "";
    gInfoXiKch15.modifierValues2 = vector<double> {};
    gInfoXiKch15.allCentralities = true;

  SystematicsFileInfo gInfoXiKch16;
    gInfoXiKch16.resultsDate = "20170429";
    gInfoXiKch16.dirNameModifierBase1 = "_Xi_minV0PosDaughterToPrimVertex_";
    gInfoXiKch16.modifierValues1 = vector<double> {0.05, 0.1, 0.2};
    gInfoXiKch16.dirNameModifierBase2 = "";
    gInfoXiKch16.modifierValues2 = vector<double> {};
    gInfoXiKch16.allCentralities = true;

/*
  SystematicsFileInfo gInfoXiKch6;
    gInfoXiKch6.resultsDate = "2017xxxx";
    gInfoXiKch6.dirNameModifierBase1 = "_ALLV0S_minInvMassReject_";
    gInfoXiKch6.modifierValues1 = vector<double> {0.494614, 0.492614, 0.488614, 0.482614};
    gInfoXiKch6.dirNameModifierBase2 = "_ALLV0S_maxInvMassReject_";
    gInfoXiKch6.modifierValues2 = vector<double> {0.500614, 0.502614, 0.506614, 0.512614};
    gInfoXiKch6.allCentralities = false;
*/

  //---------------------------------------------------------------------------------------------
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


  //---------------------------------------------------------------------------------------------

  if(aNumber==1) return gInfoXiKch1;
  else if(aNumber==2) return gInfoXiKch2;
  else if(aNumber==3) return gInfoXiKch3;
  else if(aNumber==4) return gInfoXiKch4;
  else if(aNumber==5) return gInfoXiKch5;
  else if(aNumber==6) return gInfoXiKch6;
  else if(aNumber==7) return gInfoXiKch7;
  else if(aNumber==8) return gInfoXiKch8;
  else if(aNumber==9) return gInfoXiKch9;
  else if(aNumber==10) return gInfoXiKch10;
  else if(aNumber==11) return gInfoXiKch11;
  else if(aNumber==12) return gInfoXiKch12;
  else if(aNumber==13) return gInfoXiKch13;
  else if(aNumber==14) return gInfoXiKch14;
  else if(aNumber==15) return gInfoXiKch15;
  else if(aNumber==16) return gInfoXiKch16;

  else if(aNumber==-1) return gInfoLamKch1;
  else if(aNumber==-2) return gInfoLamKch2;
  else if(aNumber==-3) return gInfoLamKch3;
  else if(aNumber==-4) return gInfoLamKch4;
  else if(aNumber==-5) return gInfoLamKch5;
  else if(aNumber==-6) return gInfoLamKch6;
  else if(aNumber==-7) return gInfoLamKch7;
  else if(aNumber==-8) return gInfoLamKch8;
  else if(aNumber==-9) return gInfoLamKch9;
  else if(aNumber==-10) return gInfoLamKch10;
  else if(aNumber==-11) return gInfoLamKch11;
  else if(aNumber==-12) return gInfoLamKch12;

  else
  {
    cout << "ERROR: SystematicsFileInfo GetFileInfo" << endl;
    assert(0);
    return gInfoXiKch1;
  }
}


//--------------------------------------------------------------------------------
void AddToCfValuesVector(td2dVec &aVecToAdd, td2dVec &aTotalVec)
{
  assert(aVecToAdd.size() == aTotalVec.size());
  for(unsigned int iBin=0; iBin<aTotalVec.size(); iBin++)
  {
    for(unsigned int iVal=0; iVal<aVecToAdd[iBin].size(); iVal++)
    {
      aTotalVec[iBin].push_back(aVecToAdd[iBin][iVal]);
    }
  }
}


//--------------------------------------------------------------------------------
void BuildErrorsAndAverageVectors(td2dVec &aTotalVec, td1dVec &aErrorsVec, td1dVec &aAvgsVec)
{
  for(unsigned int iBin=0; iBin<aTotalVec.size(); iBin++)
  {
    double tAverage = 0;
    unsigned int tNVals = aTotalVec[iBin].size();
    for(unsigned int iVal=0; iVal<tNVals; iVal++) tAverage += aTotalVec[iBin][iVal];
    tAverage /= tNVals;
    aAvgsVec.push_back(tAverage);

    double tVarSq = 0;
    for(unsigned int iVal=0; iVal<tNVals; iVal++) tVarSq += pow((aTotalVec[iBin][iVal]-tAverage),2);
    tVarSq /= tNVals;
    double tVar = sqrt(tVarSq);
    aErrorsVec.push_back(tVar);
  }
}

//--------------------------------------------------------------------------------
td1dVec CombineXiAndLam(td1dVec &aXiVec, td1dVec &aLamVec)
{
  td1dVec tReturnVec(0);
  assert(aXiVec.size() == aLamVec.size());
  tReturnVec.resize(aXiVec.size());

  double tVarSq;
  for(unsigned int i=0; i<aXiVec.size(); i++)
  {
    tVarSq = aXiVec[i]*aXiVec[i] + aLamVec[i]*aLamVec[i];
    tReturnVec[i] = sqrt(tVarSq);
  }
  return tReturnVec;
}


int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  bool bPlayCompletionBeep = true;


//--------------------------------XiKch---------------------------------------------
  AnalysisType tAnTypeXiKch = kXiKchP;
  CentralityType tCentType = k0010;

  TString tResultsDate_Save = "20170501";  //TODO //TODO //TODO CHOOSE CORRECT minDcaXiBac AND minDcaV0!!!!!!!!!!!!!!

  bool tSaveFile = true;

  TString tGeneralAnTypeNameXiKch;
  if(tAnTypeXiKch==kXiKchP || tAnTypeXiKch==kAXiKchM || tAnTypeXiKch==kXiKchM || tAnTypeXiKch==kAXiKchP) tGeneralAnTypeNameXiKch = "cXicKch";
  else assert(0);

  TString tDirectoryBase_Save = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeNameXiKch.Data(),tResultsDate_Save.Data());
  TString tFileLocationBase_Save = tDirectoryBase_Save + TString::Format("Results_%s_%s",tGeneralAnTypeNameXiKch.Data(),tResultsDate_Save.Data());

  td2dVec tAllCfValuesXiKch(0);

  vector<double> tCutsXiKch {1,2,4,5,7,8,9,10,12,15,16};
  for(unsigned int iCut=0; iCut<tCutsXiKch.size(); iCut++)
  {
    int tCut = tCutsXiKch[iCut];
    cout << "tCut = " << tCut << endl;

    SystematicsFileInfo tFileInfo = GetFileInfo(tCut);
      TString tResultsDate = tFileInfo.resultsDate;
      TString tDirNameModifierBase1 = tFileInfo.dirNameModifierBase1;
      vector<double> tModifierValues1 = tFileInfo.modifierValues1;
      TString tDirNameModifierBase2 = tFileInfo.dirNameModifierBase2;
      vector<double> tModifierValues2 = tFileInfo.modifierValues2;

    TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Systematics/Results_%s_Systematics%s",tGeneralAnTypeNameXiKch.Data(),tDirNameModifierBase1.Data());
    if(!tDirNameModifierBase2.IsNull())
    {
      tDirectoryBase.Remove(TString::kTrailing,'_');
      tDirectoryBase += tDirNameModifierBase2;
    }
    tDirectoryBase += TString::Format("%s/",tResultsDate.Data());


    TString tFileLocationBase = tDirectoryBase + TString::Format("Results_%s_Systematics%s",tGeneralAnTypeNameXiKch.Data(),tDirNameModifierBase1.Data());
    TString tFileLocationBaseMC = tDirectoryBase + TString::Format("Results_%sMC_Systematics%s",tGeneralAnTypeNameXiKch.Data(),tDirNameModifierBase1.Data());
    if(!tDirNameModifierBase2.IsNull())
    {
      tFileLocationBase.Remove(TString::kTrailing,'_');
      tFileLocationBaseMC.Remove(TString::kTrailing,'_');

      tFileLocationBase += tDirNameModifierBase2;
      tFileLocationBaseMC += tDirNameModifierBase2;
    }
    tFileLocationBase += tResultsDate;
    tFileLocationBaseMC += tResultsDate;

    SystematicAnalysis* tSysAn = new SystematicAnalysis(tFileLocationBase, static_cast<AnalysisType>(tAnTypeXiKch), static_cast<CentralityType>(tCentType), tDirNameModifierBase1, tModifierValues1, tDirNameModifierBase2, tModifierValues2);

    td2dVec tCfValues = tSysAn->GetAllCfValues();
    if(iCut==0) tAllCfValuesXiKch = tCfValues;
    else AddToCfValuesVector(tCfValues,tAllCfValuesXiKch);
  }

  td1dVec tAvgsVecXiKch(0);
  td1dVec tErrorsVecXiKch(0);
  BuildErrorsAndAverageVectors(tAllCfValuesXiKch,tErrorsVecXiKch,tAvgsVecXiKch);
/*
  TH1F* tCfAvgWithErrors = new TH1F("tCfAvgWithErrors","tCfAvgWithErrors",tAvgsVecXiKch.size(),0.,1.);
  for(unsigned int i=0; i<tAvgsVecXiKch.size(); i++)
  {
    tCfAvgWithErrors->SetBinContent(i+1,tAvgsVecXiKch[i]);
    tCfAvgWithErrors->SetBinError(i+1,tErrorsVecXiKch[i]);
  }
  tCfAvgWithErrors->SetMarkerStyle(20);
  tCfAvgWithErrors->SetMarkerSize(1.);
  tCfAvgWithErrors->SetMarkerColor(2);
  tCfAvgWithErrors->SetLineColor(2);

  TCanvas* tCan = new TCanvas("tCan","tCan");
    tCan->cd();

  tCfAvgWithErrors->Draw();
*/

//--------------------------------LamKch---------------------------------------------
  AnalysisType tAnTypeLamKch;
  TString tGeneralAnTypeNameLamKch = "cLamcKch";
  if(tAnTypeXiKch==kXiKchP) tAnTypeLamKch = kLamKchP;
  else if(tAnTypeXiKch==kAXiKchM) tAnTypeLamKch = kALamKchM;
  else if(tAnTypeXiKch==kXiKchM) tAnTypeLamKch = kLamKchM;
  else if(tAnTypeXiKch==kAXiKchP) tAnTypeLamKch = kALamKchP;
  else assert(0);

  td2dVec tAllCfValuesLamKch(0);

  vector<double> tCutsLamKch {5, 8, 1, 9, 11};
  for(unsigned int iCut=0; iCut<tCutsLamKch.size(); iCut++)
  {
    int tCut = -1*tCutsLamKch[iCut];
    cout << "tCut = " << tCut << endl;

    SystematicsFileInfo tFileInfo = GetFileInfo(tCut);
      TString tResultsDate = tFileInfo.resultsDate;
      TString tDirNameModifierBase1 = tFileInfo.dirNameModifierBase1;
      vector<double> tModifierValues1 = tFileInfo.modifierValues1;
      TString tDirNameModifierBase2 = tFileInfo.dirNameModifierBase2;
      vector<double> tModifierValues2 = tFileInfo.modifierValues2;

    TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Systematics/Results_%s_Systematics%s",tGeneralAnTypeNameLamKch.Data(),tDirNameModifierBase1.Data());
    if(!tDirNameModifierBase2.IsNull())
    {
      tDirectoryBase.Remove(TString::kTrailing,'_');
      tDirectoryBase += tDirNameModifierBase2;
    }
    tDirectoryBase += TString::Format("%s/",tResultsDate.Data());


    TString tFileLocationBase = tDirectoryBase + TString::Format("Results_%s_Systematics%s",tGeneralAnTypeNameLamKch.Data(),tDirNameModifierBase1.Data());
    TString tFileLocationBaseMC = tDirectoryBase + TString::Format("Results_%sMC_Systematics%s",tGeneralAnTypeNameLamKch.Data(),tDirNameModifierBase1.Data());
    if(!tDirNameModifierBase2.IsNull())
    {
      tFileLocationBase.Remove(TString::kTrailing,'_');
      tFileLocationBaseMC.Remove(TString::kTrailing,'_');

      tFileLocationBase += tDirNameModifierBase2;
      tFileLocationBaseMC += tDirNameModifierBase2;
    }
    tFileLocationBase += tResultsDate;
    tFileLocationBaseMC += tResultsDate;

    SystematicAnalysis* tSysAn = new SystematicAnalysis(tFileLocationBase, static_cast<AnalysisType>(tAnTypeLamKch), static_cast<CentralityType>(tCentType), tDirNameModifierBase1, tModifierValues1, tDirNameModifierBase2, tModifierValues2);

    td2dVec tCfValues = tSysAn->GetAllCfValues();
    if(iCut==0) tAllCfValuesLamKch = tCfValues;
    else AddToCfValuesVector(tCfValues,tAllCfValuesLamKch);
  }

  td1dVec tAvgsVecLamKch(0);
  td1dVec tErrorsVecLamKch(0);
  BuildErrorsAndAverageVectors(tAllCfValuesLamKch,tErrorsVecLamKch,tAvgsVecLamKch);


//-------------------------------------------------------------------------------
  td1dVec tErrorsVec = CombineXiAndLam(tErrorsVecXiKch, tErrorsVecLamKch);

//-------------------------------------------------------------------------------

  Analysis* tSaveAnalysis = new Analysis(tFileLocationBase_Save,tAnTypeXiKch,tCentType);
    tSaveAnalysis->BuildKStarHeavyCf(0.32,0.4,2);
  TH1* tCfwErrors = tSaveAnalysis->GetKStarHeavyCf()->GetHeavyCfClone();
  TString tCfwErrorsTitle = TString(cAnalysisRootTags[tAnTypeXiKch])+TString(cCentralityTags[tCentType])+TString("_wSysErrors");
  TString tCfwErrorsName = TString(cAnalysisBaseTags[tAnTypeXiKch])+TString(cCentralityTags[tCentType])+TString("_wSysErrors");
    tCfwErrors->SetTitle(tCfwErrorsTitle);
    tCfwErrors->SetName(tCfwErrorsName);

  assert(tCfwErrors->GetNbinsX() == (int)tErrorsVec.size());
  for(int i=0; i<tCfwErrors->GetNbinsX(); i++)
  {
    tCfwErrors->SetBinError(i+1,tErrorsVec[i]);
  }


  TCanvas* tCan2 = new TCanvas("tCan2","tCan2");
    tCan2->cd();
  tCfwErrors->Draw();

  if(tSaveFile)
  {
    TString tFileName = tDirectoryBase_Save + TString::Format("SystematicResults_%s%s_%s.root",cAnalysisBaseTags[tAnTypeXiKch],cCentralityTags[tCentType],tResultsDate_Save.Data());
    TFile *tSaveFile = new TFile(tFileName, "RECREATE");
    tCfwErrors->Write();
    tSaveFile->Close();
  }


cout << "DONE" << endl;
if(bPlayCompletionBeep) system("( speaker-test -t sine -f 1000 )& pid=$! ; sleep 0.5s ; kill -9 $pid");
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}


