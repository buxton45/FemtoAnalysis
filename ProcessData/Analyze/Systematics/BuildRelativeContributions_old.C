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
void BuildRelativeConts(td2dVec &aCfsVec)
{
  td2dVec tPctDiffsVec(aCfsVec.size());
  for(unsigned int iBin=0; iBin<aCfsVec.size(); iBin++)
  {
    assert(aCfsVec[iBin].size()==3);  //Middle value, ie index 1, is value used for analysis
    for(unsigned int iCut=0; iCut<aCfsVec[iBin].size(); iCut++)
    {
      if(iCut==1)
      {
        tPctDiffsVec[iBin].push_back(0.);
        continue;
      }
      double tPctDiff = fabs((aCfsVec[iBin][iCut]-aCfsVec[iBin][1])/aCfsVec[iBin][1]);
      tPctDiffsVec[iBin].push_back(tPctDiff);
    }
    assert(tPctDiffsVec[iBin].size() == aCfsVec[iBin].size());
  }
  assert(tPctDiffsVec.size() == aCfsVec.size());
    
  //----------------------------------------------------------------------
  //for(unsigned int iBin=0; iBin<aCfsVec.size(); iBin++)
  for(unsigned int iBin=0; iBin<30; iBin++)
  {  
    cout << "iBin = " << iBin << endl;
    for(unsigned int iCut=0; iCut<tPctDiffsVec[iBin].size(); iCut++)
    {
      cout << "iCut = " << iCut << " : tPctDiffsVec[iBin][iCut] = " << 100*tPctDiffsVec[iBin][iCut] << endl; 
    }
    cout << endl << endl;
  }
    
   cout << "--------------------------------------------------" << endl << endl << endl;
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
  
  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;
//-----------------------------------------------------------------------------

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

      int tMaxCut;
      if(tAnType==kLamK0 || tAnType==kALamK0) tMaxCut = 17;
      else if(tAnType==kLamKchP || tAnType==kALamKchM || tAnType==kLamKchM || tAnType==kALamKchP) tMaxCut = 12;
      else assert(0);

      td1dVec tKStarBinningInfo;
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
        //----------------------------------------------------
        cout << "tDirNameModifierBase1 = " << tDirNameModifierBase1 << endl;
        td2dVec tCfValues = tSysAn->GetAllCfValues();
        BuildRelativeConts(tCfValues);        

        delete tSysAn;
      }
    }
  }


cout << "DONE" << endl;
if(bPlayCompletionBeep) system("( speaker-test -t sine -f 1000 )& pid=$! ; sleep 0.5s ; kill -9 $pid");
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}


