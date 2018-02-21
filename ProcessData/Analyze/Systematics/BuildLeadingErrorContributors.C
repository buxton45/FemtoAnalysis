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
void BuildErrorsAndAverageVectors(td2dVec &aTotalVec, td1dVec &aErrorsVec, td1dVec &aAvgsVec, TH1D* aMaxContrHist)
{
  for(unsigned int iBin=0; iBin<aTotalVec.size(); iBin++)
  {
    double tAverage = 0;
    unsigned int tNVals = aTotalVec[iBin].size();
    for(unsigned int iVal=0; iVal<tNVals; iVal++) tAverage += aTotalVec[iBin][iVal];
    tAverage /= tNVals;
    aAvgsVec.push_back(tAverage);

    double tVarSq = 0;
    double tMaxContribution = 0.0;
    int tValMaxContribution = -1;
    for(unsigned int iVal=0; iVal<tNVals; iVal++)
    {
      if(pow((aTotalVec[iBin][iVal]-tAverage),2) > tMaxContribution)
      {
        tMaxContribution = pow((aTotalVec[iBin][iVal]-tAverage),2);
        tValMaxContribution = floor(iVal/3.);  //3 variations of each cut, i.e iVal=0,1,2 all do to varying cut0
      }
      tVarSq += pow((aTotalVec[iBin][iVal]-tAverage),2);
    }
    tVarSq /= tNVals;
    double tVar = sqrt(tVarSq);
    aErrorsVec.push_back(tVar);

    cout << "iBin = " << iBin << endl;
    cout << "\ttValMaxContribution = " << tValMaxContribution << endl;
    aMaxContrHist->Fill(tValMaxContribution);
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
  TString tResultsDate_Save = "20161027";
  TH1D* tMaxContrHist = new TH1D("tMaxContrHist","tMaxContrHist",20,0,20);

  for(int iAn=(int)kLamKchP; iAn<(int)kXiKchP; iAn++)
  {
    AnalysisType tAnType = static_cast<AnalysisType>(iAn);
    for(int iCent=(int)k0010; iCent<(int)kMB; iCent++)
    {
      CentralityType tCentType = static_cast<CentralityType>(iCent);

      TString tGeneralAnTypeName;
      if(tAnType==kLamK0 || tAnType==kALamK0) tGeneralAnTypeName = "cLamK0";
      else if(tAnType==kLamKchP || tAnType==kALamKchM || tAnType==kLamKchM || tAnType==kALamKchP) tGeneralAnTypeName = "cLamcKch";
      else assert(0);

      TString tDirectoryBase_Save = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tResultsDate_Save.Data());
      TString tFileLocationBase_Save = tDirectoryBase_Save + TString::Format("Results_%s_%s",tGeneralAnTypeName.Data(),tResultsDate_Save.Data());

      td2dVec tAllCfValues(0);

      int tMaxCut;
      if(tAnType==kLamK0 || tAnType==kALamK0) tMaxCut = 17;
      else if(tAnType==kLamKchP || tAnType==kALamKchM || tAnType==kLamKchM || tAnType==kALamKchP) tMaxCut = 12;
      else assert(0);


      for(int iCut=1; iCut<=tMaxCut; iCut++)
      {
        if(tGeneralAnTypeName=="cLamcKch" && (iCut==6 || iCut==12)) continue;
        if(tGeneralAnTypeName=="cLamK0" && (iCut==9 || iCut==15 || iCut==17)) continue;

        int tCut = iCut;
        if(tAnType==kLamK0 || tAnType==kALamK0) tCut *= -1;
        cout << "tCut = " << tCut << endl;

        SystematicsFileInfo tFileInfo = GetFileInfo_LamK(tCut);
          TString tResultsDate = tFileInfo.resultsDate;
          TString tDirNameModifierBase1 = tFileInfo.dirNameModifierBase1;
          vector<double> tModifierValues1 = tFileInfo.modifierValues1;
          TString tDirNameModifierBase2 = tFileInfo.dirNameModifierBase2;
          vector<double> tModifierValues2 = tFileInfo.modifierValues2;

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

        SystematicAnalysis* tSysAn = new SystematicAnalysis(tFileLocationBase, static_cast<AnalysisType>(tAnType), static_cast<CentralityType>(tCentType), tDirNameModifierBase1, tModifierValues1, tDirNameModifierBase2, tModifierValues2);

        td2dVec tCfValues = tSysAn->GetAllCfValues();
        if(iCut==1) tAllCfValues = tCfValues;
        else AddToCfValuesVector(tCfValues,tAllCfValues);

        delete tSysAn;
      }

      td1dVec tAvgsVec(0);
      td1dVec tErrorsVec(0);
      BuildErrorsAndAverageVectors(tAllCfValues,tErrorsVec,tAvgsVec,tMaxContrHist);
    }
  }

  TCanvas* tCan = new TCanvas("tCan","tCan");
  tCan->cd();
  tMaxContrHist->Draw();

cout << "DONE" << endl;
if(bPlayCompletionBeep) system("( speaker-test -t sine -f 1000 )& pid=$! ; sleep 0.5s ; kill -9 $pid");
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}


