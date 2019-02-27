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
td1dVec ReadParameterValue(TString aLine, int aNValsPerCut=3)
{
  assert(aNValsPerCut <= 4);
  td1dVec tReturnVec(0);

  TObjArray* tCuts = aLine.Tokenize('|');
  assert(tCuts->GetEntries()==aNValsPerCut+1);

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

  if(aNValsPerCut==4)
  {
    TString tCut4 = ((TObjString*)tCuts->At(3))->String().Strip(TString::kBoth, ' ');

    int tBeg4 = tCut4.First(':');
    tCut4.Remove(0,tBeg4+1);
    int tEnd4 = tCut4.First("+");
    tCut4.Remove(tEnd4,tCut4.Length()-tEnd4);
    tCut4.Strip(TString::kBoth, ' ');

    tReturnVec.push_back(tCut4.Atof());
  }

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
void ReadFile(TString aFileLocation, td4dVec &aAll, int aNValsPerCut=3)
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
      td1dVec tValuesVec = ReadParameterValue(tLine, aNValsPerCut);
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
void ReadAllCutSys(TString aSystematicsDirectory, td4dVec &aAllCutSysToFill, AnalysisType aAnType, CentralityType aCentType, TString aFitInfoTString, bool aRunOldQMNaming)
{
  TString tParentResultsDate;
  if     (aSystematicsDirectory.Contains("20161027")) tParentResultsDate = TString("20161027");
  else if(aSystematicsDirectory.Contains("20180505")) tParentResultsDate = TString("20180505");
  else assert(0);

  assert(aAnType==kLamK0 || aAnType==kLamKchP || aAnType==kLamKchM);

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


  int tMaxCutLamKch=12, tMaxCutLamK0=17;
  vector<int> tCutInts(0);
  for(int i=1; i<=tMaxCutLamKch; i++)
  {
    if(i==6 || i==12) continue;
    else tCutInts.push_back(i);
  }
  for(int i=1; i<=tMaxCutLamK0; i++)
  {
    if(i==9 || i==15 || i==17) continue;
    else tCutInts.push_back(-i);
  }

  TString tGeneralAnTypeModified;
  int tCutInt=0;

  for(int iCut=0; iCut<tCutInts.size(); iCut++)
  {
    tCutInt = tCutInts[iCut];
    if     (tCutInt > 0) tGeneralAnTypeModified = "cLamcKch";
    else if(tCutInt < 0) tGeneralAnTypeModified = "cLamK0";
    else assert(0);


    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++ ";
    cout << "tCutInt = " << tCutInt;
    cout << " +++++++++++++++++++++++++++++++++++++++++++++++++++++++ " << endl << endl;

    SystematicsFileInfo tFileInfo = GetFileInfo_LamK(tCutInt, tParentResultsDate);
      TString tResultsDate = tFileInfo.resultsDate;
      TString tDirNameModifierBase1 = tFileInfo.dirNameModifierBase1;
      vector<double> tModifierValues1 = tFileInfo.modifierValues1;
      TString tDirNameModifierBase2 = tFileInfo.dirNameModifierBase2;
      vector<double> tModifierValues2 = tFileInfo.modifierValues2;

    TString tDirectoryBase = TString::Format("%sResults_TripleSystematics_%sVaried%s", aSystematicsDirectory.Data(), tGeneralAnTypeModified.Data(), tDirNameModifierBase1.Data());
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
      tFileLocationBase = TString::Format("%s%s/CfFitValues_%s%s.txt", tDirectoryBase.Data(), aFitInfoTString.Data(), cAnalysisBaseTags[aAnType], cCentralityTags[aCentType]);
    }

    ReadFile(tFileLocationBase,aAllCutSysToFill);
    cout << "Read file " << tFileLocationBase << endl << endl;
  }

}


//----------------------------------------------------------------------
void ReadFitRangeAndNonFlatBgdSys(TString aResultsDirectory, td4dVec &aAllFitSysToFill, AnalysisType aAnType, CentralityType aCentType, TString aFitInfoTString, bool aRunOldQMNaming)
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
    TString tFileLocationBase_FitRangeAndNonFlat = TString::Format("%s%s/Systematics/", aResultsDirectory.Data(), aFitInfoTString.Data());
    tFileLocationFitRangeSys = TString::Format("%sCfFitValues_VaryMaxFitKStar_%s%s.txt", tFileLocationBase_FitRangeAndNonFlat.Data(), 
                                               cAnalysisBaseTags[aAnType], cCentralityTags[aCentType]);
    tFileLocationNonFlatBgdSys = TString::Format("%sCfFitValues_VaryNonFlatBgdFitType_%s%s.txt", tFileLocationBase_FitRangeAndNonFlat.Data(), 
                                                 cAnalysisBaseTags[aAnType], cCentralityTags[aCentType]);
  }

  ReadFile(tFileLocationFitRangeSys, aAllFitSysToFill);
  ReadFile(tFileLocationNonFlatBgdSys, aAllFitSysToFill, 4);

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
  TString tParentResultsDate = "20180505";  //Parent analysis these systematics are to accompany

  bool bRunOldQMNaming = false;

  CentralityType tCentralityType = kMB;  //Probably should always be kMB
  bool bIncludeFitRangeSys = true;
  bool bWriteToFile = true;


  bool bUseStavCf=false;

  bool tShareLambdaParams = true;
  bool tAllShareSingleLambdaParam = false;

  //--Dualie sharing options
  bool tDualieShareLambda = true;
  bool tDualieShareRadii = true;

  //--Corrections
  bool ApplyMomResCorrection = true;
  bool ApplyNonFlatBackgroundCorrection = true;
  NonFlatBgdFitType tNonFlatBgdFitType_cLamcKch = kPolynomial;
  NonFlatBgdFitType tNonFlatBgdFitType_cLamK0 = kLinear;
  vector<NonFlatBgdFitType> tNonFlatBgdFitTypes{tNonFlatBgdFitType_cLamK0, tNonFlatBgdFitType_cLamK0,
                                                tNonFlatBgdFitType_cLamcKch, tNonFlatBgdFitType_cLamcKch, tNonFlatBgdFitType_cLamcKch, tNonFlatBgdFitType_cLamcKch};

  //--Residuals
  IncludeResidualsType tIncludeResidualsType = kInclude3Residuals; 
  ChargedResidualsType tChargedResidualsType = kUseXiDataAndCoulombOnlyInterp/*kUseCoulombOnlyInterpForAll*/;
  ResPrimMaxDecayType tResPrimMaxDecayType = k10fm;


  //--Fix parameters
  bool FixRadii = false;
  bool FixD0 = false;
  bool FixAllScattParams = false;
  bool FixAllLambdaTo1 = false;
  if(FixAllLambdaTo1)
  {
    tAllShareSingleLambdaParam = true;
  }
  bool FixAllNormTo1 = false;

  //--mT scaling
  bool UsemTScalingOfResidualRadii = false;
  double mTScalingPowerOfResidualRadii = -0.5;


  if(bRunOldQMNaming) tIncludeResidualsType = kIncludeNoResiduals; 

  //-------------------------------
  TString tFitInfoTString = LednickyFitter::BuildSaveNameModifier(ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, tNonFlatBgdFitTypes,
                                                                          tIncludeResidualsType, tResPrimMaxDecayType, 
                                                                          tChargedResidualsType, FixD0,
                                                                          bUseStavCf, FixAllLambdaTo1, FixAllNormTo1, FixRadii, FixAllScattParams,
                                                                          tShareLambdaParams, tAllShareSingleLambdaParam, UsemTScalingOfResidualRadii, true,
                                                                          tDualieShareLambda, tDualieShareRadii);

  TString tSystematicsDirectory = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Systematics_LamK_%s/TripleSystematics/", tParentResultsDate.Data());

  //NOTE: All needed results in LamKch directory
  TString tResultsDirectory = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",tParentResultsDate.Data());

  TString tSaveDirectory = tResultsDirectory;

  if(!bRunOldQMNaming)
  {
    tSaveDirectory += tFitInfoTString;
    tSaveDirectory += TString("/Systematics/");
  }

  //-------------------------------
  int tNAnalysisTypes = 6; //kLamK0, kALamK0, kLamKchP, kALamKchP, kLamKchM, kALamKchM
  int tNCentralityTypes = 3; //k0010, k1030, k3050
  int tNParameterTypes = 5; //kLambda, kRadius, kRef0, kImf0, kd0

  td4dVec tAllCutSys(0);
    tAllCutSys.resize(tNAnalysisTypes, td3dVec(tNCentralityTypes, td2dVec(tNParameterTypes, td1dVec(0))));

  //---------------------------------------------------------------------------------------

  ReadAllCutSys(tSystematicsDirectory, tAllCutSys, kLamKchP, tCentralityType, tFitInfoTString, bRunOldQMNaming);
  ReadAllCutSys(tSystematicsDirectory, tAllCutSys, kLamKchM, tCentralityType, tFitInfoTString, bRunOldQMNaming);
  ReadAllCutSys(tSystematicsDirectory, tAllCutSys, kLamK0, tCentralityType, tFitInfoTString, bRunOldQMNaming);

  //---------------------------------------------------------------------------------------
  td4dVec tFinalCutSysVec = ReduceCutsVector(tAllCutSys);
  //-----------------------------------------------------------------------------------------
  td4dVec tFinalVec;



  if(bIncludeFitRangeSys)
  {
    td4dVec tAllFitSys(0);
      tAllFitSys.resize(tNAnalysisTypes, td3dVec(tNCentralityTypes, td2dVec(tNParameterTypes, td1dVec(0))));

    ReadFitRangeAndNonFlatBgdSys(tResultsDirectory, tAllFitSys, kLamKchP, tCentralityType, tFitInfoTString, bRunOldQMNaming);
    ReadFitRangeAndNonFlatBgdSys(tResultsDirectory, tAllFitSys, kLamKchM, tCentralityType, tFitInfoTString, bRunOldQMNaming);
    ReadFitRangeAndNonFlatBgdSys(tResultsDirectory, tAllFitSys, kLamK0, tCentralityType, tFitInfoTString, bRunOldQMNaming);

    td4dVec tFinalFitSysVec = ReduceCutsVector(tAllFitSys);
    tFinalVec = CombineCutSyswFitSys(tFinalCutSysVec,tFinalFitSysVec);
  }
  else tFinalVec = tFinalCutSysVec;

  //-----------------------------------------------------------------------------------------



  if(bWriteToFile)
  {
    TString tOutputName = TString::Format("%sFinalFitSystematics", tSaveDirectory.Data());
    if(bIncludeFitRangeSys) tOutputName += TString("_wFitRangeSys");
    if(!bRunOldQMNaming)
    {
      tOutputName += tFitInfoTString;
    }
    tOutputName += TString(".txt");

    std::ofstream tOutput;
    tOutput.open(tOutputName);
    assert(tOutput.is_open());


    PrintFinalVec(tFinalVec, tOutput);
      cout << "****************** Output LamKch info to file: " << tOutputName << endl;
    PrintFinalVec(tFinalVec, tOutput);
      cout << "****************** Output LamK0 info to file: " << tOutputName << endl;

    tOutput.close();
  }
  else PrintFinalVec(tFinalVec);


cout << "DONE" << endl;
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
