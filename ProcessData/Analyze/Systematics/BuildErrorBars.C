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
  TString tParentResultsDate = "20190319";  //Parent analysis these systematics are to accompany
  bool aBuildStavCfs = true;

  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;

  AnalysisType tAnType = kLamKchP;
  CentralityType tCentType = k0010;

  bool tSaveFile = true;

  TString tGeneralAnTypeName;
  if(tAnType==kLamK0 || tAnType==kALamK0) tGeneralAnTypeName = "cLamK0";
  else if(tAnType==kLamKchP || tAnType==kALamKchM || tAnType==kLamKchM || tAnType==kALamKchP) tGeneralAnTypeName = "cLamcKch";
  else assert(0);

  TString tDirectoryBase_Save = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tParentResultsDate.Data());
  TString tFileLocationBase_Save = tDirectoryBase_Save + TString::Format("Results_%s_%s",tGeneralAnTypeName.Data(),tParentResultsDate.Data());

  td2dVec tAllCfValues(0);
  td2dVec tAllStavCfValues(0);

  int tMaxCut;
  if(tAnType==kLamK0 || tAnType==kALamK0) tMaxCut = 17;
  else if(tAnType==kLamKchP || tAnType==kALamKchM || tAnType==kLamKchM || tAnType==kALamKchP) tMaxCut = 12;
  else assert(0);

  td1dVec tKStarBinningInfo;
  td1dVec tStavBinningInfo;
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
    if(aBuildStavCfs) 
    {
      tSysAn->BuildStavHeavyCfs();
      tStavBinningInfo = tSysAn->GetStavBinningInfo();
      for(int i=0; i<tKStarBinningInfo.size(); i++)
      {
        assert(tKStarBinningInfo[i]==tStavBinningInfo[i]);
      }
    }
    //----------------------------------------------------
    td2dVec tCfValues = tSysAn->GetAllCfValues();
    if(iCut==1) tAllCfValues = tCfValues;
    else AddToCfValuesVector(tCfValues,tAllCfValues);
    //----------------------------------------------------
    td2dVec tStavCfValues(0);
    if(aBuildStavCfs)
    {
      tStavCfValues = tSysAn->GetAllStavCfValues();
      if(iCut==1) tAllStavCfValues = tStavCfValues;
      else AddToCfValuesVector(tStavCfValues,tAllStavCfValues);
    }
    //----------------------------------------------------
  }

  //------------------------------------------------------
  td1dVec tAvgsVec(0);
  td1dVec tErrorsVec(0);
  BuildErrorsAndAverageVectors(tAllCfValues,tErrorsVec,tAvgsVec);
  //------------------------------------------------------
  td1dVec tAvgsVecStav(0);
  td1dVec tErrorsVecStav(0);
  if(aBuildStavCfs) BuildErrorsAndAverageVectors(tAllStavCfValues,tErrorsVecStav,tAvgsVecStav);
/*
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
*/
//-------------------------------------------------------------------------------
  Analysis* tSaveAnalysis = new Analysis(tFileLocationBase_Save, tAnType, tCentType, tAnRunType, tNPartialAnalysis, TString(""), false);
    tSaveAnalysis->BuildKStarHeavyCf(0.32,0.4,2);
  TH1* tCfwErrorsFull = tSaveAnalysis->GetKStarHeavyCf()->GetHeavyCfClone();
  TString tCfwErrorsTitle = TString(cAnalysisRootTags[tAnType])+TString(cCentralityTags[tCentType])+TString("_wSysErrors");
  TString tCfwErrorsName = TString(cAnalysisBaseTags[tAnType])+TString(cCentralityTags[tCentType])+TString("_wSysErrors");

/*
  assert(tCfwErrorsFull->GetNbinsX() >= (int)tErrorsVec.size());
  TH1D* tCfwErrors;
  if(tCfwErrorsFull->GetNbinsX() > (int)tErrorsVec.size())
  {
    int tNbins = (int)tErrorsVec.size();
    double tMin = tCfwErrorsFull->GetXaxis()->GetBinLowEdge(1);
    double tMax = tCfwErrorsFull->GetXaxis()->GetBinUpEdge(tNbins);
    tCfwErrors = new TH1D(tCfwErrorsName, tCfwErrorsTitle, tNbins, tMin, tMax);
    tCfwErrors->Sumw2();
    for(int i=1; i<=tNbins; i++)
    {
      tCfwErrors->SetBinContent(i, tCfwErrorsFull->GetBinContent(i));
      tCfwErrors->SetBinError(i, tCfwErrorsFull->GetBinError(i));
    }
  }
  else tCfwErrors = (TH1D*)tCfwErrorsFull->Clone();
*/
  assert(tCfwErrorsFull->GetNbinsX()                 ==tNbinsKStar);
  assert(tCfwErrorsFull->GetBinLowEdge(1)            ==tKStarMin);
  assert(tCfwErrorsFull->GetBinLowEdge(tNbinsKStar+1)==tKStarMax);

  TH1D* tCfwErrors;
  tCfwErrors = (TH1D*)tCfwErrorsFull->Clone();
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

//-------------------------------------------------------------------------------
  TH1* tStavCfwErrors;
  if(aBuildStavCfs)
  {
    tSaveAnalysis->BuildStavHeavyCf(0.32,0.4,2);
    tStavCfwErrors = tSaveAnalysis->GetStavHeavyCf()->GetHeavyCfClone();
    TString tStavCfwErrorsTitle = TString("StavCf_") + TString(cAnalysisRootTags[tAnType])+TString(cCentralityTags[tCentType])+TString("_wSysErrors");
    TString tStavCfwErrorsName =  TString("StavCf_") + TString(cAnalysisBaseTags[tAnType])+TString(cCentralityTags[tCentType])+TString("_wSysErrors");

    assert(tStavCfwErrors->GetNbinsX()                 ==tNbinsKStar);
    assert(tStavCfwErrors->GetBinLowEdge(1)            ==tKStarMin);
    assert(tStavCfwErrors->GetBinLowEdge(tNbinsKStar+1)==tKStarMax);
  
    tStavCfwErrors->SetTitle(tStavCfwErrorsTitle);
    tStavCfwErrors->SetName(tStavCfwErrorsName);

    assert(tStavCfwErrors->GetNbinsX() == (int)tErrorsVecStav.size());
    for(int i=0; i<tStavCfwErrors->GetNbinsX(); i++)
    {
      tStavCfwErrors->SetBinError(i+1,tErrorsVecStav[i]);
    }


    TCanvas* tCan3 = new TCanvas("tCan3","tCan3");
      tCan3->cd();
    tStavCfwErrors->Draw();
  }

//-------------------------------------------------------------------------------

  if(tSaveFile)
  {
    TString tFileName = tDirectoryBase_Save + TString::Format("SystematicResults_%s%s_%s.root",cAnalysisBaseTags[tAnType],cCentralityTags[tCentType],tParentResultsDate.Data());
    TFile *tSaveFile = new TFile(tFileName, "RECREATE");
    tCfwErrors->Write();
    if(aBuildStavCfs) tStavCfwErrors->Write();
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


