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
void FillCutsVector(td4dVec &aAll, td1dVec &aValues, ParameterType aParamType, AnalysisType aAnType, CentralityType aCentType)
{
  for(unsigned int iVal=0; iVal<aValues.size(); iVal++)
  {
    aAll[aAnType][aCentType][aParamType].push_back(aValues[iVal]);
  }
}

//----------------------------------------------------------------------
void ReadFile(TString aFileLocation, td4dVec &aAll)
{
  ifstream tFileIn(aFileLocation);
  if(!tFileIn.is_open()) cout << "FAILURE - FILE NOT OPEN: " << aFileLocation << endl;
  assert(tFileIn.is_open());

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
      td1dVec tValuesVec = ReadParameterValue(tLine);
      if(tLine.Contains("Lambda")) tCurrentParameterType = kLambda;
      else if(tLine.Contains("Radius")) tCurrentParameterType = kRadius;
      else if(tLine.Contains("Ref0")) tCurrentParameterType = kRef0;
      else if(tLine.Contains("Imf0")) tCurrentParameterType = kImf0;
      else if(tLine.Contains("d0")) tCurrentParameterType = kd0;
      else assert(0);

      FillCutsVector(aAll, tValuesVec, tCurrentParameterType, tCurrentAnType, tCurrentCentralityType);
    }
  }
  tFileIn.close();
}


//----------------------------------------------------------------------
td4dVec ReduceCutsVector(td4dVec &aAll)
{
  td4dVec tReturnVec(0);
  tReturnVec.resize(aAll.size(), td3dVec(aAll[0].size(), td2dVec(aAll[0][0].size(), td1dVec(2,0))));

  for(unsigned int iAnType=0; iAnType<aAll.size(); iAnType++)
  {
    for(unsigned int iCentType=0; iCentType<aAll[iAnType].size(); iCentType++)
    {
      for(unsigned int iParamType=0; iParamType<aAll[iAnType][iCentType].size(); iParamType++)
      {
        double tSum=0.;
        int tCounter1 = 0;
        for(unsigned int iVal=0; iVal<aAll[iAnType][iCentType][iParamType].size(); iVal++)
        {
          tSum += aAll[iAnType][iCentType][iParamType][iVal];
          tCounter1 ++;
        }
        tSum /= tCounter1;

        double tVarSq = 0.;
        for(unsigned int iVal=0; iVal<aAll[iAnType][iCentType][iParamType].size(); iVal++)
        {
          tVarSq += pow((aAll[iAnType][iCentType][iParamType][iVal]-tSum),2);
        }
        tVarSq /= tCounter1;
        double tVar = sqrt(tVarSq);

        tReturnVec[iAnType][iCentType][iParamType][0] = tSum;
        tReturnVec[iAnType][iCentType][iParamType][1] = tVar;
      }
    }
  }

  return tReturnVec;
}


//----------------------------------------------------------------------
td4dVec CombineCutSyswFitSys(td4dVec &aCutSys, td4dVec &aFitSys)
{
  //TODO implement a better comparison
  assert(aCutSys.size() == aFitSys.size());
  for(unsigned int i=0; i<aCutSys.size(); i++) assert(aCutSys[i].size()==aFitSys[i].size());
  for(unsigned int i=0; i<aCutSys[0].size(); i++) assert(aCutSys[0][i].size()==aFitSys[0][i].size());
  for(unsigned int i=0; i<aCutSys[0][0].size(); i++) assert(aCutSys[0][0][i].size()==aFitSys[0][0][i].size());

  td4dVec tReturnVec(0);
  tReturnVec.resize(aCutSys.size(), td3dVec(aCutSys[0].size(), td2dVec(aCutSys[0][0].size(), td1dVec(2,0))));

  for(unsigned int iAnType=0; iAnType<aCutSys.size(); iAnType++)
  {
    for(unsigned int iCentType=0; iCentType<aCutSys[iAnType].size(); iCentType++)
    {
      for(unsigned int iParamType=0; iParamType<aCutSys[iAnType][iCentType].size(); iParamType++)
      {
        double tSum = 0.5*(aCutSys[iAnType][iCentType][iParamType][0] + aFitSys[iAnType][iCentType][iParamType][0]);
        double tError = pow(aCutSys[iAnType][iCentType][iParamType][1],2) + pow(aFitSys[iAnType][iCentType][iParamType][1],2);
        tError = sqrt(tError);

        tReturnVec[iAnType][iCentType][iParamType][0] = tSum;
        tReturnVec[iAnType][iCentType][iParamType][1] = tError;
      }
    }
  }
  return tReturnVec;
}

/*
//----------------------------------------------------------------------
void PrintFinalVec(td4dVec &aFinal)
{
  for(unsigned int iAnType=0; iAnType<aFinal.size(); iAnType++)
  {
    for(unsigned int iCentType=0; iCentType<aFinal[iAnType].size(); iCentType++)
    {
      cout << "------------------------------------------------------------" << endl;
      cout << "AnalysisType = " << cAnalysisBaseTags[iAnType] << endl;
      cout << "CentralityType = " << cPrettyCentralityTags[iCentType] << endl;
      for(unsigned int iParamType=0; iParamType<aFinal[iAnType][iCentType].size(); iParamType++)
      {
        cout << "ParamType = " << cParameterNames[iParamType] << endl;
        cout << "\tAverage = " << aFinal[iAnType][iCentType][iParamType][0] << endl;
        cout << "\tError   = " << aFinal[iAnType][iCentType][iParamType][1] << endl;
	cout << "\t\t Percent Error = " << 100*aFinal[iAnType][iCentType][iParamType][1]/fabs(aFinal[iAnType][iCentType][iParamType][0]) << "%" << endl;
        cout << endl;
      }
      cout << "------------------------------------------------------------" << endl;
    }
  }
}
*/
//----------------------------------------------------------------------

void PrintFinalVec(td4dVec &aFinal, ostream &aOut=std::cout)
{
  for(unsigned int iAnType=0; iAnType<aFinal.size(); iAnType++)
  {
    unsigned int tNCent = aFinal[iAnType].size();
    for(unsigned int i=1; i<tNCent; i++) assert(aFinal[iAnType][i-1].size() == aFinal[iAnType][i].size());
    unsigned int tNParams = aFinal[iAnType][0].size();
    for(unsigned int i=0; i<tNCent; i++) aOut << std::setw(50) << "--------------------------------------------- | ";
    aOut << endl;
    for(unsigned int i=0; i<tNCent; i++) aOut << TString::Format("AnalysisType = %s", cAnalysisBaseTags[iAnType]) << std::setw(50-TString::Format("AnalysisType = %s", cAnalysisBaseTags[iAnType]).Sizeof()+1) << " | ";
    aOut << endl;
    for(unsigned int i=0; i<tNCent; i++) aOut << TString::Format("CentralityType = %s", cPrettyCentralityTags[i]) << std::setw(50-TString::Format("CentralityType = %s", cPrettyCentralityTags[i]).Sizeof()+1) << " | ";
    aOut << endl;
    for(unsigned int iPar=0; iPar<tNParams; iPar++)
    {
      for(unsigned int iCent=0; iCent<tNCent; iCent++) aOut << TString::Format("\tParamType = %s", cParameterNames[iPar]) << std::setw(50-TString::Format("\tParamType = %s", cParameterNames[iPar]).Sizeof()+1-7+iCent) << " | ";
      aOut << endl;
      for(unsigned int iCent=0; iCent<tNCent; iCent++) aOut << TString::Format("\t\tAverage = %f", aFinal[iAnType][iCent][iPar][0]) << std::setw(50-TString::Format("\t\tAverage = %f", aFinal[iAnType][iCent][iPar][0]).Sizeof()+1-14+iCent) << " | ";
      aOut << endl;
      for(unsigned int iCent=0; iCent<tNCent; iCent++) aOut << TString::Format("\t\tError   = %f", aFinal[iAnType][iCent][iPar][1]) << std::setw(50-TString::Format("\t\tError   = %f", aFinal[iAnType][iCent][iPar][1]).Sizeof()+1-14+iCent) << " | ";
      aOut << endl;
      for(unsigned int iCent=0; iCent<tNCent; iCent++) aOut << TString::Format("\t\t\t%%Error = %f%%",100*aFinal[iAnType][iCent][iPar][1]/fabs(aFinal[iAnType][iCent][iPar][0])) << std::setw(50-TString::Format("\t\t\t%%Error = %f%%",100*aFinal[iAnType][iCent][iPar][1]/fabs(aFinal[iAnType][iCent][iPar][0])).Sizeof()+1-21+iCent) << " | ";
      aOut << endl;
      aOut << std::setw(50) << " | " << std::setw(50) << " | " << std::setw(50) << " | " << endl;
    }
  }
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

    ReadFile(tFileLocationBase,aAllCutSysToFill);
    cout << "Read file " << tFileLocationBase << endl << endl;
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

  ReadFile(tFileLocationFitRangeSys, aAllFitSysToFill);
  ReadFile(tFileLocationNonFlatBgdSys, aAllFitSysToFill);

  cout << "Read file " << tFileLocationFitRangeSys << endl;
  cout << "Read file " << tFileLocationNonFlatBgdSys << endl;
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

  CentralityType tCentralityType = kMB;  //Probably should always be kMB

  bool ApplyMomResCorrection = true;
  bool ApplyNonFlatBackgroundCorrection = true;
  NonFlatBgdFitType tNonFlatBgdFitType = kLinear;

  IncludeResidualsType tIncludeResidualsType = kInclude3Residuals; 
  ResPrimMaxDecayType tResPrimMaxDecayType = k4fm;
  ChargedResidualsType tChargedResidualsType = kUseXiDataAndCoulombOnlyInterp;

  bool tFixD0 = false;

  bool bIncludeFitRangeSys = true;
  bool bWriteToFile = false;

  if(bRunOldQMNaming) tIncludeResidualsType = kIncludeNoResiduals; 

  //-------------------------------

  TString tSystematicsDirectory = "/home/jesse/Analysis/FemtoAnalysis/Results/Systematics/";

  TString tResultsDirectory_cLamK0 = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_20161027/";
  TString tResultsDirectory_cLamcKch = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20161027/";

  TString tSaveDirectory_cLamK0 = tResultsDirectory_cLamK0;
  TString tSaveDirectory_cLamcKch = tResultsDirectory_cLamcKch;

  if(!bRunOldQMNaming)
  {
    tSaveDirectory_cLamK0 += TString("Systematics/");
    LednickyFitter::AppendFitInfo(tSaveDirectory_cLamK0, ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, 
                                         tIncludeResidualsType, tResPrimMaxDecayType, tChargedResidualsType, tFixD0);
    tSaveDirectory_cLamK0 += TString("/");
    //-----
    tSaveDirectory_cLamcKch += TString("Systematics/");
    LednickyFitter::AppendFitInfo(tSaveDirectory_cLamcKch, ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, 
                                         tIncludeResidualsType, tResPrimMaxDecayType, tChargedResidualsType, tFixD0);
    tSaveDirectory_cLamcKch += TString("/");
  }

  //-------------------------------
  int tNAnalysisTypes = 6; //kLamK0, kALamK0, kLamKchP, kALamKchP, kLamKchM, kALamKchM
  int tNCentralityTypes = 3; //k0010, k1030, k3050
  int tNParameterTypes = 5; //kLambda, kRadius, kRef0, kImf0, kd0

  td4dVec tAllCutSys(0);
    tAllCutSys.resize(tNAnalysisTypes, td3dVec(tNCentralityTypes, td2dVec(tNParameterTypes, td1dVec(0))));

  //---------------------------------------------------------------------------------------

  ReadAllCutSys(tSystematicsDirectory, tAllCutSys, kLamKchP, tCentralityType, 
                ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, 
                tIncludeResidualsType, tResPrimMaxDecayType, tChargedResidualsType, 
                tFixD0, bRunOldQMNaming);

  ReadAllCutSys(tSystematicsDirectory, tAllCutSys, kLamKchM, tCentralityType, 
                ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, 
                tIncludeResidualsType, tResPrimMaxDecayType, tChargedResidualsType, 
                tFixD0, bRunOldQMNaming);

  ReadAllCutSys(tSystematicsDirectory, tAllCutSys, kLamK0, tCentralityType, 
                ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, 
                tIncludeResidualsType, tResPrimMaxDecayType, tChargedResidualsType, 
                tFixD0, bRunOldQMNaming);

  //---------------------------------------------------------------------------------------
  td4dVec tFinalCutSysVec = ReduceCutsVector(tAllCutSys);
  //-----------------------------------------------------------------------------------------
  td4dVec tFinalVec;



  if(bIncludeFitRangeSys)
  {
    td4dVec tAllFitSys(0);
      tAllFitSys.resize(tNAnalysisTypes, td3dVec(tNCentralityTypes, td2dVec(tNParameterTypes, td1dVec(0))));

    ReadFitRangeAndNonFlatBgdSys(tResultsDirectory_cLamcKch, tAllFitSys, kLamKchP, tCentralityType,
                                 ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, 
                                 tIncludeResidualsType, tResPrimMaxDecayType, tChargedResidualsType, 
                                 tFixD0, bRunOldQMNaming);

    ReadFitRangeAndNonFlatBgdSys(tResultsDirectory_cLamcKch, tAllFitSys, kLamKchM, tCentralityType,
                                 ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, 
                                 tIncludeResidualsType, tResPrimMaxDecayType, tChargedResidualsType, 
                                 tFixD0, bRunOldQMNaming);

    ReadFitRangeAndNonFlatBgdSys(tResultsDirectory_cLamK0, tAllFitSys, kLamK0, tCentralityType,
                                 ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, 
                                 tIncludeResidualsType, tResPrimMaxDecayType, tChargedResidualsType, 
                                 tFixD0, bRunOldQMNaming);

    td4dVec tFinalFitSysVec = ReduceCutsVector(tAllFitSys);
    tFinalVec = CombineCutSyswFitSys(tFinalCutSysVec,tFinalFitSysVec);
  }
  else tFinalVec = tFinalCutSysVec;

  //-----------------------------------------------------------------------------------------



  if(bWriteToFile)
  {
    TString tOutputLamKchName = TString::Format("%sFinalFitSystematics", tSaveDirectory_cLamcKch.Data());
    if(bIncludeFitRangeSys) tOutputLamKchName += TString("_wFitRangeSys");
    if(!bRunOldQMNaming)
    {
      LednickyFitter::AppendFitInfo(tOutputLamKchName, ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, 
                                           tIncludeResidualsType, tResPrimMaxDecayType, tChargedResidualsType, tFixD0);
      tOutputLamKchName += TString("_");
    }
    tOutputLamKchName += TString("cLamcKch.txt");

    TString tOutputLamK0Name = TString::Format("%sFinalFitSystematics", tSaveDirectory_cLamK0.Data());
    if(bIncludeFitRangeSys) tOutputLamK0Name += TString("_wFitRangeSys");
    if(!bRunOldQMNaming)
    {
      LednickyFitter::AppendFitInfo(tOutputLamK0Name, ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, 
                                           tIncludeResidualsType, tResPrimMaxDecayType, tChargedResidualsType, tFixD0);
      tOutputLamK0Name += TString("_");
    }
    tOutputLamK0Name += TString("cLamK0.txt");

    std::ofstream tOutputLamKch;
    tOutputLamKch.open(tOutputLamKchName);

    std::ofstream tOutputLamK0;
    tOutputLamK0.open(tOutputLamK0Name);

    PrintFinalVec(tFinalVec, tOutputLamKch);
      cout << "****************** Output LamKch info to file: " << tOutputLamKchName << endl;
    PrintFinalVec(tFinalVec, tOutputLamK0);
      cout << "****************** Output LamK0 info to file: " << tOutputLamK0Name << endl;

    tOutputLamKch.close();
    tOutputLamK0.close();
  }
  else PrintFinalVec(tFinalVec);


cout << "DONE" << endl;
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
