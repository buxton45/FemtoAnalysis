#include "FitSystematicAnalysis.h"
class FitSystematicAnalysis;

#include "Types_SysFileInfo.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>

#include "TObjString.h"

//----------------------------------------------------------------------
AnalysisType GetAnalysisType(TString aLine)
{
  int tBeg = aLine.First("AnalysisType");
  TString tSmallLine = aLine.Remove(0,tBeg);

  int tEnd = tSmallLine.First('|');
  tSmallLine = tSmallLine.Remove(tEnd,tSmallLine.Length());

  TObjArray* tContents = tSmallLine.Tokenize("=");
  TString tAnName = ((TObjString*)tContents->At(1))->String().Strip(TString::kBoth, ' ');

  AnalysisType tAnType;
  if(tAnName.EqualTo("LamKchP")) tAnType = kLamKchP;
  else if(tAnName.EqualTo("ALamKchM")) tAnType = kALamKchM;

  else if(tAnName.EqualTo("LamKchM")) tAnType = kLamKchM;
  else if(tAnName.EqualTo("ALamKchP")) tAnType = kALamKchP;

  else if(tAnName.EqualTo("LamK0")) tAnType = kLamK0;
  else if(tAnName.EqualTo("ALamK0")) tAnType = kALamK0;

  else assert(0);

  return tAnType;
}

//----------------------------------------------------------------------
CentralityType GetCentralityType(TString aLine)
{
  int tBeg = aLine.First("Centrality Type");
  TString tSmallLine = aLine.Remove(0,tBeg);

  int tEnd = tSmallLine.First('|');
  tSmallLine = tSmallLine.Remove(tEnd,tSmallLine.Length());

  TObjArray* tContents = tSmallLine.Tokenize("=");
  TString tCentName = ((TObjString*)tContents->At(1))->String().Strip(TString::kBoth, ' ');

  CentralityType tCentType;
  if(tCentName.EqualTo("0-10%")) tCentType = k0010;
  else if(tCentName.EqualTo("10-30%")) tCentType = k1030;
  else if(tCentName.EqualTo("30-50%")) tCentType = k3050;
  else assert(0);

  return tCentType;
}

//----------------------------------------------------------------------
td1dVec ReadParameterDifferences(TString aLine)
{
  td1dVec tReturnVec(0);

  TObjArray* tCuts = aLine.Tokenize('|');
  assert(tCuts->GetEntries()==4);

  TString tCut1 = ((TObjString*)tCuts->At(0))->String().Strip(TString::kBoth, ' ');
  TString tCut2 = ((TObjString*)tCuts->At(1))->String().Strip(TString::kBoth, ' ');
  TString tCut3 = ((TObjString*)tCuts->At(2))->String().Strip(TString::kBoth, ' ');

  int tBeg1 = tCut1.First('(');
  tCut1.Remove(0,tBeg1+1);
  int tEnd1 = tCut1.First(")");
  tCut1.Remove(tEnd1,tCut1.Length()-tEnd1);
  tCut1.Strip(TString::kBoth, ' ');

  int tBeg2 = tCut2.First('(');
  tCut2.Remove(0,tBeg2+1);
  int tEnd2 = tCut2.First(")");
  tCut2.Remove(tEnd2,tCut2.Length()-tEnd2);
  tCut2.Strip(TString::kBoth, ' ');

  int tBeg3 = tCut3.First('(');
  tCut3.Remove(0,tBeg3+1);
  int tEnd3 = tCut3.First(")");
  tCut3.Remove(tEnd3,tCut3.Length()-tEnd3);
  tCut3.Strip(TString::kBoth, ' ');

  tReturnVec.push_back(tCut1.Atof());
  tReturnVec.push_back(tCut2.Atof());
  tReturnVec.push_back(tCut3.Atof());

  return tReturnVec;
}

//----------------------------------------------------------------------
td1dVec ReadParameterValue(TString aLine)
{
  td1dVec tReturnVec(0);

  TObjArray* tCuts = aLine.Tokenize('|');
  assert(tCuts->GetEntries()==4);

  TString tCut1 = ((TObjString*)tCuts->At(0))->String().Strip(TString::kBoth, ' ');
  TString tCut2 = ((TObjString*)tCuts->At(1))->String().Strip(TString::kBoth, ' ');
  TString tCut3 = ((TObjString*)tCuts->At(2))->String().Strip(TString::kBoth, ' ');

  int tBeg1 = tCut1.First(':');
  tCut1.Remove(0,tBeg1+1);
  int tEnd1 = tCut1.First("+");
  tCut1.Remove(tEnd1,tCut1.Length()-tEnd1);
  tCut1.Strip(TString::kBoth, ' ');

  int tBeg2 = tCut2.First(':');
  tCut2.Remove(0,tBeg2+1);
  int tEnd2 = tCut2.First("+");
  tCut2.Remove(tEnd2,tCut2.Length()-tEnd2);
  tCut2.Strip(TString::kBoth, ' ');

  int tBeg3 = tCut3.First(':');
  tCut3.Remove(0,tBeg3+1);
  int tEnd3 = tCut3.First("+");
  tCut3.Remove(tEnd3,tCut3.Length()-tEnd3);
  tCut3.Strip(TString::kBoth, ' ');

  tReturnVec.push_back(tCut1.Atof());
  tReturnVec.push_back(tCut2.Atof());
  tReturnVec.push_back(tCut3.Atof());

  return tReturnVec;
}


//----------------------------------------------------------------------
void ReadFile(TString aFileLocation, td4dVec &aAll, double aTolerance=100)
{
  ifstream tFileIn(aFileLocation);

  AnalysisType tCurrentAnType;
  CentralityType tCurrentCentralityType;
  ParameterType tCurrentParameterType;

  std::string tStdString;
  TString tLine;
  while(getline(tFileIn, tStdString))
  {
    tLine = TString(tStdString);

    if(tLine.Contains("AnalysisType")) tCurrentAnType = GetAnalysisType(tLine);
    if(tLine.Contains("CentralityType")) tCurrentCentralityType = GetCentralityType(tLine);
    if(tLine.Contains("Lambda") || tLine.Contains("Radius") || tLine.Contains("Ref0") || tLine.Contains("Imf0") || tLine.Contains("d0"))
    {
      td1dVec tDiffVec = ReadParameterDifferences(tLine);
      td1dVec tValuesVec = ReadParameterValue(tLine);
      if(tLine.Contains("Lambda")) tCurrentParameterType = kLambda;
      else if(tLine.Contains("Radius")) tCurrentParameterType = kRadius;
      else if(tLine.Contains("Ref0")) tCurrentParameterType = kRef0;
      else if(tLine.Contains("Imf0")) tCurrentParameterType = kImf0;
      else if(tLine.Contains("d0")) tCurrentParameterType = kd0;
      else assert(0);

      for(unsigned int i=0; i<tDiffVec.size(); i++)
      {
        if(tValuesVec[i] != 0. && tDiffVec[1]!=0.)
        {
          cout << "tDiffVec[1]!=0. !!!!!!!!!!!!!!!!!!!!" << endl;
          cout << "Analysis   = " << cAnalysisBaseTags[tCurrentAnType] << endl;
          cout << "Centrality = " << cCentralityTags[tCurrentCentralityType] << endl;
          cout << "Parameter  = " << cParameterNames[tCurrentParameterType] << endl;
        }
        if(tValuesVec[i] != 0.) assert(tDiffVec[1]==0.);  //If tValuesVec[i]==0., then tDiffVec[i] = nan

        if(abs(tDiffVec[i]) > aTolerance)
        {
          cout << std::setw(10) << cAnalysisBaseTags[tCurrentAnType] << " | ";
          cout << std::setw(10) << cPrettyCentralityTags[tCurrentCentralityType] << " | ";
          cout << std::setw(10) << cParameterNames[tCurrentParameterType] << " | ";
          cout << std::setw(5)  << "i = " << i << " | ";
          cout << std::setw(20) << TString::Format("Diff = %0.3f", tDiffVec[i]) << " | ";
          cout << std::setw(20) << TString::Format("Val = %0.3f", tValuesVec[i]) << std::setw(5) << " ";
          cout << std::setw(10) << TString::Format(" (FitVal = %0.3f)", tValuesVec[1]) << endl;

        }
      }

    }
  }
  tFileIn.close();
}







//----------------------------------------------------------------------
void ReadAllCutSys(TString aSystematicsDirectory, td4dVec &aAllCutSysToFill, AnalysisType aAnType, CentralityType aCentType, 
                   bool aApplyMomResCrctn, bool aApplyNonFlatBgdCrctn, 
                   IncludeResidualsType aIncResType, ResPrimMaxDecayType aMaxDecayType, ChargedResidualsType aChargedResType, 
                   bool aFixD0, bool aRunOldQMNaming)
{
  TString tGeneralAnTypeName;

  assert(aAnType==kLamK0 || aAnType==kLamKchP || aAnType==kLamKchM);
  if(aAnType==kLamK0) tGeneralAnTypeName = "cLamK0";
  else if(aAnType==kLamKchP || aAnType==kLamKchM) tGeneralAnTypeName = "cLamcKch";
  else assert(0);

  //----------------------------------------------
  cout << endl;
  cout << "________________________________";
  cout << " Reading systematics for " << cAnalysisBaseTags[aAnType] << " and " << cAnalysisBaseTags[aAnType+1];
  cout << "________________________________" << endl;
  //----------------------------------------------

  int tNCuts;
  if(aAnType==kLamK0) tNCuts = 17;
  else if(aAnType==kLamKchP || aAnType==kLamKchM) tNCuts = 12;
  else assert(0);

  int tCut=-999;
  for(int iCut=1; iCut<=tNCuts; iCut++)
  {
    if(aAnType==kLamK0 && (iCut==9 || iCut==15 || iCut==17)) continue;
    if((aAnType==kLamKchP || aAnType==kLamKchM) && (iCut==6 || iCut==12)) continue;

    if(aAnType==kLamK0) tCut = -1*iCut;
    else tCut = iCut;
    cout << "tCut = " << tCut << endl;

    SystematicsFileInfo tFileInfo = GetFileInfo_LamK(tCut);
      TString tResultsDate = tFileInfo.resultsDate;
      TString tDirNameModifierBase1 = tFileInfo.dirNameModifierBase1;
      vector<double> tModifierValues1 = tFileInfo.modifierValues1;
      TString tDirNameModifierBase2 = tFileInfo.dirNameModifierBase2;
      vector<double> tModifierValues2 = tFileInfo.modifierValues2;

    TString tDirectoryBase = TString::Format("%sResults_%s_Systematics%s", aSystematicsDirectory.Data(), tGeneralAnTypeName.Data(), tDirNameModifierBase1.Data());
    if(!tDirNameModifierBase2.IsNull())
    {
      tDirectoryBase.Remove(TString::kTrailing,'_');
      tDirectoryBase += tDirNameModifierBase2;
    }
    tDirectoryBase += TString::Format("%s/",tResultsDate.Data());

    TString tFileLocationBase;
    if(aRunOldQMNaming)
    {
      tFileLocationBase = tDirectoryBase + TString::Format("CfFitValues_%s_MomResCrctn_NonFlatBgdCrctn.txt", cAnalysisBaseTags[aAnType]);
    }
    else
    {
      tFileLocationBase = tDirectoryBase;
      LednickyFitter::AppendFitInfo(tFileLocationBase, aApplyMomResCrctn, aApplyNonFlatBgdCrctn, 
                                           aIncResType, aMaxDecayType, aChargedResType, aFixD0);
      tFileLocationBase += TString::Format("/CfFitValues_%s%s", cAnalysisBaseTags[aAnType], cCentralityTags[aCentType]);
      LednickyFitter::AppendFitInfo(tFileLocationBase, aApplyMomResCrctn, aApplyNonFlatBgdCrctn, 
                                           aIncResType, aMaxDecayType, aChargedResType, aFixD0);
      tFileLocationBase += TString(".txt");
    }

    cout << "Read file " << tFileLocationBase << endl;
    ReadFile(tFileLocationBase,aAllCutSysToFill);
    cout << endl;
  }

}


//----------------------------------------------------------------------
void ReadFitRangeAndNonFlatBgdSys(TString aResultsDirectory, td4dVec &aAllFitSysToFill, AnalysisType aAnType, CentralityType aCentType, 
                                  bool aApplyMomResCrctn, bool aApplyNonFlatBgdCrctn, 
                                  IncludeResidualsType aIncResType, ResPrimMaxDecayType aMaxDecayType, ChargedResidualsType aChargedResType, 
                                  bool aFixD0, bool aRunOldQMNaming)
{
  //----------------------------------------------
  cout << endl;
  cout << "________________________________";
  cout << " Reading FitRangeSys and NonFlatBgd for " << cAnalysisBaseTags[aAnType] << " and " << cAnalysisBaseTags[aAnType+1];
  cout << "________________________________" << endl;
  //----------------------------------------------

  TString tFileLocationFitRangeSys;
  TString tFileLocationNonFlatBgdSys;
  if(aRunOldQMNaming)
  {
    tFileLocationFitRangeSys = TString::Format("%sCfFitValues_VaryMaxFitKStar_%s_MomResCrctn_NonFlatBgdCrctn.txt", aResultsDirectory.Data(), cAnalysisBaseTags[aAnType]);
    tFileLocationNonFlatBgdSys = TString::Format("%sCfFitValues_VaryNonFlatBgdFitType_%s_MomResCrctn_NonFlatBgdCrctn.txt", aResultsDirectory.Data(), cAnalysisBaseTags[aAnType]);
  }
  else
  {
    TString tFileLocationBase_FitRangeAndNonFlat = TString::Format("%sSystematics/", aResultsDirectory.Data());
    LednickyFitter::AppendFitInfo(tFileLocationBase_FitRangeAndNonFlat, aApplyMomResCrctn, aApplyNonFlatBgdCrctn, aIncResType, aMaxDecayType, aChargedResType, aFixD0);
    tFileLocationBase_FitRangeAndNonFlat += TString("/");

    tFileLocationFitRangeSys = TString::Format("%sCfFitValues_VaryMaxFitKStar_%s%s", tFileLocationBase_FitRangeAndNonFlat.Data(), 
                                               cAnalysisBaseTags[aAnType], cCentralityTags[aCentType]);
    LednickyFitter::AppendFitInfo(tFileLocationFitRangeSys, aApplyMomResCrctn, aApplyNonFlatBgdCrctn, aIncResType, aMaxDecayType, aChargedResType, aFixD0);
    tFileLocationFitRangeSys += TString(".txt");


    tFileLocationNonFlatBgdSys = TString::Format("%sCfFitValues_VaryNonFlatBgdFitType_%s%s", tFileLocationBase_FitRangeAndNonFlat.Data(), 
                                                 cAnalysisBaseTags[aAnType], cCentralityTags[aCentType]);
    LednickyFitter::AppendFitInfo(tFileLocationNonFlatBgdSys, aApplyMomResCrctn, aApplyNonFlatBgdCrctn, aIncResType, aMaxDecayType, aChargedResType, aFixD0);
    tFileLocationNonFlatBgdSys += TString(".txt");
  }

  cout << "Read file " << tFileLocationFitRangeSys << endl;
  ReadFile(tFileLocationFitRangeSys, aAllFitSysToFill);

  cout << "Read file " << tFileLocationNonFlatBgdSys << endl;
  ReadFile(tFileLocationNonFlatBgdSys, aAllFitSysToFill);

}


//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//********************************************************************************************************************************************************************************
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------

  bool bRunOldQMNaming = false;

  AnalysisType tAnType = kLamKchP;
  CentralityType tCentralityType = kMB;  //Probably should always be kMB

  bool ApplyMomResCorrection = true;
  bool ApplyNonFlatBackgroundCorrection = true;
  NonFlatBgdFitType tNonFlatBgdFitType = kLinear;

  IncludeResidualsType tIncludeResidualsType = kInclude3Residuals; 
  ResPrimMaxDecayType tResPrimMaxDecayType = k4fm;
  ChargedResidualsType tChargedResidualsType = kUseXiDataAndCoulombOnlyInterp;

  bool tFixD0 = false;

  if(bRunOldQMNaming) tIncludeResidualsType = kIncludeNoResiduals; 

  //-------------------------------

  TString tSystematicsDirectory = "/home/jesse/Analysis/FemtoAnalysis/Results/Systematics/";

  TString tResultsDirectory_cLamK0 = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_20171227/";
  TString tResultsDirectory_cLamcKch = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20171227/";

  //-------------------------------

  //Stealing methods from ReadSystematicFitResults.C, which require these 4d vectors
  //Otherwise, they are useless
  td4dVec tAllCutSys(0);
  td4dVec tAllFitSys(0);

  bool bIncludeFitRangeSys = true;
  bool bWriteToFile = false;

  //---------------------------------------------------------------------------------------
  ReadAllCutSys(tSystematicsDirectory, tAllCutSys, tAnType, tCentralityType, 
                ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, 
                tIncludeResidualsType, tResPrimMaxDecayType, tChargedResidualsType, 
                tFixD0, bRunOldQMNaming);

  ReadFitRangeAndNonFlatBgdSys(tResultsDirectory_cLamcKch, tAllFitSys, tAnType, tCentralityType,
                               ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, 
                               tIncludeResidualsType, tResPrimMaxDecayType, tChargedResidualsType, 
                               tFixD0, bRunOldQMNaming);
  //-----------------------------------------------------------------------------------------




cout << "DONE" << endl;
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
