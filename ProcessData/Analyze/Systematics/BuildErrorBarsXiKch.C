#include "SystematicAnalysis.h"
class SystematicAnalysis;

#include "Types_SysFileInfo.h"

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



int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  bool bPlayCompletionBeep = true;
//-----------------------------------------------------------------------------

  TString tParentResultsDate = "20161027";  //Parent analysis these systematics are to accompany


  AnalysisType tAnType = kXiKchP;
  CentralityType tCentType = k0010;

  TString tResultsDate_Save = "20170501";  //TODO //TODO //TODO CHOOSE CORRECT minDcaXiBac and minDcaV0!!!!!!!!!!!!!!

  bool tSaveFile = true;

  TString tGeneralAnTypeName;
  if(tAnType==kXiKchP || tAnType==kAXiKchM || tAnType==kXiKchM || tAnType==kAXiKchP) tGeneralAnTypeName = "cXicKch";
  else assert(0);

  TString tDirectoryBase_Save = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tResultsDate_Save.Data());
  TString tFileLocationBase_Save = tDirectoryBase_Save + TString::Format("Results_%s_%s",tGeneralAnTypeName.Data(),tResultsDate_Save.Data());

  td2dVec tAllCfValues(0);

  int tMaxCut = 16;

  for(int iCut=1; iCut<=tMaxCut; iCut++)
  {
    int tCut = iCut;
    if(iCut==6) continue;
    cout << "tCut = " << tCut << endl;

    SystematicsFileInfo tFileInfo = GetFileInfo_XiKch(tCut, tParentResultsDate);
      TString tResultsDate = tFileInfo.resultsDate;
      TString tDirNameModifierBase1 = tFileInfo.dirNameModifierBase1;
      vector<double> tModifierValues1 = tFileInfo.modifierValues1;
      TString tDirNameModifierBase2 = tFileInfo.dirNameModifierBase2;
      vector<double> tModifierValues2 = tFileInfo.modifierValues2;

    TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Systematics_XiK/Results_%s_Systematics%s",tGeneralAnTypeName.Data(),tDirNameModifierBase1.Data());
//    TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Systematics_XiK_%s/Results_%s_Systematics%s", tParentResultsDate.Data(), tGeneralAnTypeName.Data(), tDirNameModifierBase1.Data());
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

    SystematicAnalysis* tSysAn = new SystematicAnalysis(tFileLocationBase, static_cast<AnalysisType>(tAnType), static_cast<CentralityType>(tCentType), tDirNameModifierBase1, tModifierValues1, tDirNameModifierBase2, tModifierValues2);

    td2dVec tCfValues = tSysAn->GetAllCfValues();
    if(iCut==1) tAllCfValues = tCfValues;
    else AddToCfValuesVector(tCfValues,tAllCfValues);
  }

  td1dVec tAvgsVec(0);
  td1dVec tErrorsVec(0);
  BuildErrorsAndAverageVectors(tAllCfValues,tErrorsVec,tAvgsVec);

  TH1F* tCfAvgWithErrors = new TH1F("tCfAvgWithErrors","tCfAvgWithErrors",tAvgsVec.size(),0.,1.);
  for(unsigned int i=0; i<tAvgsVec.size(); i++)
  {
    tCfAvgWithErrors->SetBinContent(i+1,tAvgsVec[i]);
    tCfAvgWithErrors->SetBinError(i+1,tErrorsVec[i]);
  }
  tCfAvgWithErrors->SetMarkerStyle(20);
  tCfAvgWithErrors->SetMarkerSize(1.);
  tCfAvgWithErrors->SetMarkerColor(2);
  tCfAvgWithErrors->SetLineColor(2);

  TCanvas* tCan = new TCanvas("tCan","tCan");
    tCan->cd();

  tCfAvgWithErrors->Draw();

//-------------------------------------------------------------------------------
  Analysis* tSaveAnalysis = new Analysis(tFileLocationBase_Save,tAnType,tCentType);
    tSaveAnalysis->BuildKStarHeavyCf(0.32,0.4,2);
  TH1* tCfwErrors = tSaveAnalysis->GetKStarHeavyCf()->GetHeavyCfClone();
  TString tCfwErrorsTitle = TString(cAnalysisRootTags[tAnType])+TString(cCentralityTags[tCentType])+TString("_wSysErrors");
  TString tCfwErrorsName = TString(cAnalysisBaseTags[tAnType])+TString(cCentralityTags[tCentType])+TString("_wSysErrors");
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
    TString tFileName = tDirectoryBase_Save + TString::Format("SystematicResults_%s%s_%s.root",cAnalysisBaseTags[tAnType],cCentralityTags[tCentType],tResultsDate_Save.Data());
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


