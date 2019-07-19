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


//_________________________________________________________________________________________
//*****************************************************************************************
//_________________________________________________________________________________________
int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  bool bPlayCompletionBeep = true;
//-----------------------------------------------------------------------------
  TString tParentResultsDate = "20181205";  //Parent analysis these systematics are to accompany

  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;

  AnalysisType tAnType = kLamKchP;
  CentralityType tCentType = k0010;

  int aRebin=2;

  bool tSaveFile = true;

  TString tGeneralAnTypeName;
  if(tAnType==kLamK0 || tAnType==kALamK0) tGeneralAnTypeName = "cLamK0";
  else if(tAnType==kLamKchP || tAnType==kALamKchM || tAnType==kLamKchM || tAnType==kALamKchP) tGeneralAnTypeName = "cLamcKch";
  else assert(0);

  TString tDirectoryBase_Save = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tParentResultsDate.Data());
  TString tFileLocationBase_Save = tDirectoryBase_Save + TString::Format("Results_%s_%s",tGeneralAnTypeName.Data(),tParentResultsDate.Data());

  td2dVec tAllCfValues(0);
  td2dVec tAllCfValues_ReC00(0);
  td2dVec tAllCfValues_ReC11(0);

  int tMaxCut;
  if(tAnType==kLamK0 || tAnType==kALamK0) tMaxCut = 17;
  else if(tAnType==kLamKchP || tAnType==kALamKchM || tAnType==kLamKchM || tAnType==kALamKchP) tMaxCut = 12;
  else assert(0);

  td1dVec tKStarBinningInfo;
  td1dVec tKStarBinningInfo_ReC00;
  td1dVec tKStarBinningInfo_ReC11;
  double tNbinsKStar=-1., tKStarMin=-1., tKStarMax=-1.;
  for(int iCut=1; iCut<=tMaxCut; iCut++)
  {
    if(tGeneralAnTypeName=="cLamcKch" && (iCut==6 || iCut==12)) continue;
    if(tGeneralAnTypeName=="cLamK0" && (iCut==9 || iCut==15 || iCut==17)) continue;

    int tCut = iCut;
    if(tAnType==kLamK0 || tAnType==kALamK0) tCut *= -1;
    cout << "tCut = " << tCut << endl;

    SystematicsFileInfo tFileInfo = GetFileInfo_LamK(tCut, tParentResultsDate);
      TString tResultsDate = tFileInfo.resultsDate;
      TString tDirNameModifierBase1 = tFileInfo.dirNameModifierBase1;
      vector<double> tModifierValues1 = tFileInfo.modifierValues1;
      TString tDirNameModifierBase2 = tFileInfo.dirNameModifierBase2;
      vector<double> tModifierValues2 = tFileInfo.modifierValues2;

    TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Systematics_LamK_%s/Results_%s_Systematics%s", tParentResultsDate.Data(), tGeneralAnTypeName.Data(), tDirNameModifierBase1.Data());
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
    //----------------------------------------------------
    tKStarBinningInfo = tSysAn->GetKStarBinningInfo();
    if(iCut==1)
    {
      tNbinsKStar=tKStarBinningInfo[0];
      tKStarMin  =tKStarBinningInfo[1];
      tKStarMax  =tKStarBinningInfo[2];
    }
    else
    {
      assert(tNbinsKStar==tKStarBinningInfo[0]);
      assert(tKStarMin  ==tKStarBinningInfo[1]);
      assert(tKStarMax  ==tKStarBinningInfo[2]);
    }
    tKStarBinningInfo_ReC00 = tSysAn->GetYlmKStarBinningInfo(kYlmReal, 0, 0, aRebin);
    tKStarBinningInfo_ReC11 = tSysAn->GetYlmKStarBinningInfo(kYlmReal, 1, 1, aRebin);
    for(int i=0; i<tKStarBinningInfo.size(); i++)
    {
      assert(tKStarBinningInfo[i]==tKStarBinningInfo_ReC00[i]);
      assert(tKStarBinningInfo[i]==tKStarBinningInfo_ReC11[i]);
    }
    //----------------------------------------------------
    td2dVec tCfValues = tSysAn->GetAllCfValues();
    if(iCut==1) tAllCfValues = tCfValues;
    else AddToCfValuesVector(tCfValues,tAllCfValues);
    //----------------------------------------------------
    td2dVec tCfValues_ReC00 = tSysAn->GetAllYlmCfValues(kYlmReal, 0, 0, aRebin);
    if(iCut==1) tAllCfValues_ReC00 = tCfValues_ReC00;
    else AddToCfValuesVector(tCfValues_ReC00,tAllCfValues_ReC00);
    //----------------------------------------------------
    td2dVec tCfValues_ReC11 = tSysAn->GetAllYlmCfValues(kYlmReal, 1, 1, aRebin);
    if(iCut==1) tAllCfValues_ReC11 = tCfValues_ReC11;
    else AddToCfValuesVector(tCfValues_ReC11,tAllCfValues_ReC11);
  }

  //------------------------------------------------------
  td1dVec tAvgsVec(0);
  td1dVec tErrorsVec(0);
  BuildErrorsAndAverageVectors(tAllCfValues,tErrorsVec,tAvgsVec);
  //------------------------------------------------------
  td1dVec tAvgsVec_ReC00(0);
  td1dVec tErrorsVec_ReC00(0);
  BuildErrorsAndAverageVectors(tAllCfValues_ReC00,tErrorsVec_ReC00,tAvgsVec_ReC00);
  //------------------------------------------------------
  td1dVec tAvgsVec_ReC11(0);
  td1dVec tErrorsVec_ReC11(0);
  BuildErrorsAndAverageVectors(tAllCfValues_ReC11,tErrorsVec_ReC11,tAvgsVec_ReC11);

//-------------------------------------------------------------------------------
  Analysis* tSaveAnalysis = new Analysis(tFileLocationBase_Save, tAnType, tCentType, tAnRunType, tNPartialAnalysis, TString(""), false);
    tSaveAnalysis->BuildKStarHeavyCf(0.32,0.4,2);
  TH1* tCfwErrors = tSaveAnalysis->GetKStarHeavyCf()->GetHeavyCfClone();

  assert(tCfwErrors->GetNbinsX()                 ==tNbinsKStar);
  assert(tCfwErrors->GetBinLowEdge(1)            ==tKStarMin);
  assert(tCfwErrors->GetBinLowEdge(tNbinsKStar+1)==tKStarMax);

  tCfwErrors->SetTitle(TString::Format("%s%s_wSysErrors", cAnalysisRootTags[tAnType], cCentralityTags[tCentType]));
  tCfwErrors->SetName( TString::Format("%s%s_wSysErrors", cAnalysisRootTags[tAnType], cCentralityTags[tCentType]));

  assert(tCfwErrors->GetNbinsX() == (int)tErrorsVec.size());
  for(int i=0; i<tCfwErrors->GetNbinsX(); i++)
  {
    tCfwErrors->SetBinError(i+1,tErrorsVec[i]);
  }

  TCanvas* tCan = new TCanvas("tCan","tCan");
    tCan->cd();
  tCfwErrors->Draw();

//-------------------------------------------------------------------------------
  TH1D* tCfwErrors_ReC00 = (TH1D*)tSaveAnalysis->GetYlmCfnHist(kYlmReal, 0, 0, aRebin)->Clone();

  assert(tCfwErrors_ReC00->GetNbinsX()                 ==tNbinsKStar);
  assert(tCfwErrors_ReC00->GetBinLowEdge(1)            ==tKStarMin);
  assert(tCfwErrors_ReC00->GetBinLowEdge(tNbinsKStar+1)==tKStarMax);

  tCfwErrors_ReC00->SetTitle(TString::Format("%s%s_ReC00_wSysErrors", cAnalysisRootTags[tAnType], cCentralityTags[tCentType]));
  tCfwErrors_ReC00->SetName( TString::Format("%s%s_ReC00_wSysErrors", cAnalysisRootTags[tAnType], cCentralityTags[tCentType]));

  assert(tCfwErrors_ReC00->GetNbinsX() == (int)tErrorsVec_ReC00.size());
  for(int i=0; i<tCfwErrors_ReC00->GetNbinsX(); i++)
  {
    tCfwErrors_ReC00->SetBinError(i+1,tErrorsVec_ReC00[i]);
  }

  TCanvas* tCan_ReC00 = new TCanvas("tCan_ReC00","tCan_ReC00");
    tCan_ReC00->cd();
  tCfwErrors_ReC00->Draw();

//-------------------------------------------------------------------------------
  TH1D* tCfwErrors_ReC11 = (TH1D*)tSaveAnalysis->GetYlmCfnHist(kYlmReal, 1, 1, aRebin)->Clone();

  assert(tCfwErrors_ReC11->GetNbinsX()                 ==tNbinsKStar);
  assert(tCfwErrors_ReC11->GetBinLowEdge(1)            ==tKStarMin);
  assert(tCfwErrors_ReC11->GetBinLowEdge(tNbinsKStar+1)==tKStarMax);

  tCfwErrors_ReC11->SetTitle(TString::Format("%s%s_ReC11_wSysErrors", cAnalysisRootTags[tAnType], cCentralityTags[tCentType]));
  tCfwErrors_ReC11->SetName( TString::Format("%s%s_ReC11_wSysErrors", cAnalysisRootTags[tAnType], cCentralityTags[tCentType]));

  assert(tCfwErrors_ReC11->GetNbinsX() == (int)tErrorsVec_ReC11.size());
  for(int i=0; i<tCfwErrors_ReC11->GetNbinsX(); i++)
  {
    tCfwErrors_ReC11->SetBinError(i+1,tErrorsVec_ReC11[i]);
  }

  TCanvas* tCan_ReC11 = new TCanvas("tCan_ReC11","tCan_ReC11");
    tCan_ReC11->cd();
  tCfwErrors_ReC11->Draw();

//-------------------------------------------------------------------------------

  if(tSaveFile)
  {
    TString tFileName = tDirectoryBase_Save + TString::Format("SystematicResults_%s%s_%s.root",cAnalysisBaseTags[tAnType],cCentralityTags[tCentType],tParentResultsDate.Data());
    TFile *tSaveFile = new TFile(tFileName, "RECREATE");
    tCfwErrors->Write();
    tCfwErrors_ReC00->Write();
    tCfwErrors_ReC11->Write();
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


