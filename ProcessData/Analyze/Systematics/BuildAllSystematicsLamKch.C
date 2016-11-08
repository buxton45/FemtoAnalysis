#include "SystematicAnalysis.h"
class SystematicAnalysis;

SystematicsFileInfo GetFileInfo(int aNumber)
{
  SystematicsFileInfo gInfo1;
    gInfo1.resultsDate = "20161103";
    gInfo1.dirNameModifierBase1 = "_ALam_minNegDaughterToPrimVertex_";
    gInfo1.modifierValues1 = vector<double> {0.05,0.10,0.20};
    gInfo1.dirNameModifierBase2 = "";
    gInfo1.modifierValues2 = vector<double> {};
    gInfo1.allCentralities = true;

  SystematicsFileInfo gInfo2;
    gInfo2.resultsDate = "20161103";
    gInfo2.dirNameModifierBase1 = "_ALam_minPosDaughterToPrimVertex_";
    gInfo2.modifierValues1 = vector<double> {0.20, 0.30, 0.40};
    gInfo2.dirNameModifierBase2 = "";
    gInfo2.modifierValues2 = vector<double> {};
    gInfo2.allCentralities = true;

  SystematicsFileInfo gInfo3;
    gInfo3.resultsDate = "20161026";
    gInfo3.dirNameModifierBase1 = "_ALLV0S_maxDcaV0Daughters_";
    gInfo3.modifierValues1 = vector<double> {0.30,0.40,0.50};
    gInfo3.dirNameModifierBase2 = "";
    gInfo3.modifierValues2 = vector<double> {};
    gInfo3.allCentralities = true;

  SystematicsFileInfo gInfo4;
    gInfo4.resultsDate = "20161025";
    gInfo4.dirNameModifierBase1 = "_ALLV0S_minInvMassReject_";
    gInfo4.modifierValues1 = vector<double> {0.494614, 0.492614, 0.488614, 0.482614};
    gInfo4.dirNameModifierBase2 = "_ALLV0S_maxInvMassReject_";
    gInfo4.modifierValues2 = vector<double> {0.500614, 0.502614, 0.506614, 0.512614};
    gInfo4.allCentralities = false;

  SystematicsFileInfo gInfo5;
    gInfo5.resultsDate = "20161026";
    gInfo5.dirNameModifierBase1 = "_CLAM_maxDcaV0_";
    gInfo5.modifierValues1 = vector<double> {0.40, 0.50, 0.60};
    gInfo5.dirNameModifierBase2 = "";
    gInfo5.modifierValues2 = vector<double> {};
    gInfo5.allCentralities = true;

  SystematicsFileInfo gInfo6;
    gInfo6.resultsDate = "20161031";
    gInfo6.dirNameModifierBase1 = "_CLAM_minCosPointingAngle_";
    gInfo6.modifierValues1 = vector<double> {0.9992, 0.9993, 0.9994};
    gInfo6.dirNameModifierBase2 = "";
    gInfo6.modifierValues2 = vector<double> {};
    gInfo6.allCentralities = true;

  SystematicsFileInfo gInfo7;
    gInfo7.resultsDate = "20161103";
    gInfo7.dirNameModifierBase1 = "_Lam_minNegDaughterToPrimVertex_";
    gInfo7.modifierValues1 = vector<double> {0.2, 0.3, 0.4};
    gInfo7.dirNameModifierBase2 = "";
    gInfo7.modifierValues2 = vector<double> {};
    gInfo7.allCentralities = true;

  SystematicsFileInfo gInfo8;
    gInfo8.resultsDate = "20161103";
    gInfo8.dirNameModifierBase1 = "_Lam_minPosDaughterToPrimVertex_";
    gInfo8.modifierValues1 = vector<double> {0.05, 0.1, 0.2};
    gInfo8.dirNameModifierBase2 = "";
    gInfo8.modifierValues2 = vector<double> {};
    gInfo8.allCentralities = true;

  if(aNumber==1) return gInfo1;
  else if(aNumber==2) return gInfo2;
  else if(aNumber==3) return gInfo3;
  else if(aNumber==4) return gInfo4;
  else if(aNumber==5) return gInfo5;
  else if(aNumber==6) return gInfo6;
  else if(aNumber==7) return gInfo7;
  else if(aNumber==8) return gInfo8;
  else
  {
    cout << "ERROR: SystematicsFileInfo GetFileInfo" << endl;
    assert(0);
    return gInfo1;
  }
}


int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------
  TString tGeneralAnTypeName = "cLamcKch";
  bool bWriteToFile = true;

  SystematicsFileInfo tFileInfo = GetFileInfo(2);
    TString tResultsDate = tFileInfo.resultsDate;
    TString tDirNameModifierBase1 = tFileInfo.dirNameModifierBase1;
    vector<double> tModifierValues1 = tFileInfo.modifierValues1;
    TString tDirNameModifierBase2 = tFileInfo.dirNameModifierBase2;
    vector<double> tModifierValues2 = tFileInfo.modifierValues2;
    bool tAllCent = tFileInfo.allCentralities;

  CentralityType tMaxCentType = kMB;
  if(!tAllCent) tMaxCentType = k1030;


  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Systematics/Results_%s_Systematics%s",tGeneralAnTypeName.Data(),tDirNameModifierBase1.Data());
  if(!tDirNameModifierBase2.IsNull())
  {
    tDirectoryBase.Remove(TString::kTrailing,'_');
    tDirectoryBase += tDirNameModifierBase2;
  }
  tDirectoryBase += TString::Format("%s/",tResultsDate.Data());

  TString tOutputFileName = tDirectoryBase + TString("AllPValues.txt");
  std::ofstream tOutputFile;
  if(bWriteToFile) tOutputFile.open(tOutputFileName);

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

  for(int iAnType=kLamKchP; iAnType<kXiKchP; iAnType++)
  {
    for(int iCent=k0010; iCent<tMaxCentType; iCent++)
    {
      SystematicAnalysis* tSysAn = new SystematicAnalysis(tFileLocationBase, static_cast<AnalysisType>(iAnType), static_cast<CentralityType>(iCent), tDirNameModifierBase1, tModifierValues1, tDirNameModifierBase2, tModifierValues2);
      if(bWriteToFile) tSysAn->GetAllPValues(tOutputFile);
      else tSysAn->GetAllPValues();
    //tSysAn->DrawAll();
    //tSysAn->DrawAllDiffs();
    }
  }

cout << "DONE" << endl;
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
