#include "FitSystematicAnalysis.h"
class FitSystematicAnalysis;

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>

#include "TObjString.h"


SystematicsFileInfo GetFileInfo(int aNumber)
{
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

  //---------------------------------------------------------------------- 


  SystematicsFileInfo gInfoLamK01;
    gInfoLamK01.resultsDate = "20161103";
    gInfoLamK01.dirNameModifierBase1 = "_ALam_minNegDaughterToPrimVertex_";
    gInfoLamK01.modifierValues1 = vector<double> {0.05,0.10,0.20};
    gInfoLamK01.dirNameModifierBase2 = "";
    gInfoLamK01.modifierValues2 = vector<double> {};
    gInfoLamK01.allCentralities = true;

  SystematicsFileInfo gInfoLamK02;
    gInfoLamK02.resultsDate = "20161103";
    gInfoLamK02.dirNameModifierBase1 = "_ALam_minPosDaughterToPrimVertex_";
    gInfoLamK02.modifierValues1 = vector<double> {0.20,0.30,0.40};
    gInfoLamK02.dirNameModifierBase2 = "";
    gInfoLamK02.modifierValues2 = vector<double> {};
    gInfoLamK02.allCentralities = true;

  SystematicsFileInfo gInfoLamK03;
    gInfoLamK03.resultsDate = "20161026";
    gInfoLamK03.dirNameModifierBase1 = "_CLAM_maxDcaV0_";
    gInfoLamK03.modifierValues1 = vector<double> {0.40,0.50,0.60};
    gInfoLamK03.dirNameModifierBase2 = "";
    gInfoLamK03.modifierValues2 = vector<double> {};
    gInfoLamK03.allCentralities = true;

  SystematicsFileInfo gInfoLamK04;
    gInfoLamK04.resultsDate = "20161026";
    gInfoLamK04.dirNameModifierBase1 = "_CLAM_maxDcaV0Daughters_";
    gInfoLamK04.modifierValues1 = vector<double> {0.30,0.40,0.50};
    gInfoLamK04.dirNameModifierBase2 = "";
    gInfoLamK04.modifierValues2 = vector<double> {};
    gInfoLamK04.allCentralities = true;

  SystematicsFileInfo gInfoLamK05;
    gInfoLamK05.resultsDate = "20161031";
    gInfoLamK05.dirNameModifierBase1 = "_CLAM_minCosPointingAngle_";
    gInfoLamK05.modifierValues1 = vector<double> {0.9992, 0.9993, 0.9994};
    gInfoLamK05.dirNameModifierBase2 = "";
    gInfoLamK05.modifierValues2 = vector<double> {};
    gInfoLamK05.allCentralities = true;

  SystematicsFileInfo gInfoLamK06;
    gInfoLamK06.resultsDate = "20161026";
    gInfoLamK06.dirNameModifierBase1 = "_K0s_maxDcaV0_";
    gInfoLamK06.modifierValues1 = vector<double> {0.20,0.30,0.40};
    gInfoLamK06.dirNameModifierBase2 = "";
    gInfoLamK06.modifierValues2 = vector<double> {};
    gInfoLamK06.allCentralities = true;

  SystematicsFileInfo gInfoLamK07;
    gInfoLamK07.resultsDate = "20161026";
    gInfoLamK07.dirNameModifierBase1 = "_K0s_maxDcaV0Daughters_";
    gInfoLamK07.modifierValues1 = vector<double> {0.20,0.30,0.40};
    gInfoLamK07.dirNameModifierBase2 = "";
    gInfoLamK07.modifierValues2 = vector<double> {};
    gInfoLamK07.allCentralities = true;

  SystematicsFileInfo gInfoLamK08;
    gInfoLamK08.resultsDate = "20161102";
    gInfoLamK08.dirNameModifierBase1 = "_K0s_minCosPointingAngle_";
    gInfoLamK08.modifierValues1 = vector<double> {0.9992, 0.9993, 0.9994};
    gInfoLamK08.dirNameModifierBase2 = "";
    gInfoLamK08.modifierValues2 = vector<double> {};
    gInfoLamK08.allCentralities = true;

  SystematicsFileInfo gInfoLamK09;
    gInfoLamK09.resultsDate = "20161025";
    gInfoLamK09.dirNameModifierBase1 = "_K0s_minInvMassReject_";
    gInfoLamK09.modifierValues1 = vector<double> {1.112683, 1.110683, 1.106683, 1.100683};
    gInfoLamK09.dirNameModifierBase2 = "_K0s_maxInvMassReject_";
    gInfoLamK09.modifierValues2 = vector<double> {1.118683, 1.120683, 1.124683, 1.130683};
    gInfoLamK09.allCentralities = false;

  SystematicsFileInfo gInfoLamK010;
    gInfoLamK010.resultsDate = "20161103";
    gInfoLamK010.dirNameModifierBase1 = "_K0s_minNegDaughterToPrimVertex_";
    gInfoLamK010.modifierValues1 = vector<double> {0.2, 0.3, 0.4};
    gInfoLamK010.dirNameModifierBase2 = "";
    gInfoLamK010.modifierValues2 = vector<double> {};
    gInfoLamK010.allCentralities = true;

  SystematicsFileInfo gInfoLamK011;
    gInfoLamK011.resultsDate = "20161103";
    gInfoLamK011.dirNameModifierBase1 = "_K0s_minPosDaughterToPrimVertex_";
    gInfoLamK011.modifierValues1 = vector<double> {0.2, 0.3, 0.4};
    gInfoLamK011.dirNameModifierBase2 = "";
    gInfoLamK011.modifierValues2 = vector<double> {};
    gInfoLamK011.allCentralities = true;

  SystematicsFileInfo gInfoLamK012;
    gInfoLamK012.resultsDate = "20161103";
    gInfoLamK012.dirNameModifierBase1 = "_Lam_minNegDaughterToPrimVertex_";
    gInfoLamK012.modifierValues1 = vector<double> {0.2, 0.3, 0.4};
    gInfoLamK012.dirNameModifierBase2 = "";
    gInfoLamK012.modifierValues2 = vector<double> {};
    gInfoLamK012.allCentralities = true;

  SystematicsFileInfo gInfoLamK013;
    gInfoLamK013.resultsDate = "20161103";
    gInfoLamK013.dirNameModifierBase1 = "_Lam_minPosDaughterToPrimVertex_";
    gInfoLamK013.modifierValues1 = vector<double> {0.05, 0.1, 0.2};
    gInfoLamK013.dirNameModifierBase2 = "";
    gInfoLamK013.modifierValues2 = vector<double> {};
    gInfoLamK013.allCentralities = true;

  SystematicsFileInfo gInfoLamK014;
    gInfoLamK014.resultsDate = "20161106";
    gInfoLamK014.dirNameModifierBase1 = "_minAvgSepNegNeg_";
    gInfoLamK014.modifierValues1 = vector<double> {5.0, 6.0, 7.0};
    gInfoLamK014.dirNameModifierBase2 = "";
    gInfoLamK014.modifierValues2 = vector<double> {};
    gInfoLamK014.allCentralities = true;

  SystematicsFileInfo gInfoLamK015;
    gInfoLamK015.resultsDate = "20161108";
    gInfoLamK015.dirNameModifierBase1 = "_minAvgSepNegNeg_";
    gInfoLamK015.modifierValues1 = vector<double> {5.5, 6.0, 6.5};
    gInfoLamK015.dirNameModifierBase2 = "";
    gInfoLamK015.modifierValues2 = vector<double> {};
    gInfoLamK015.allCentralities = true;

  SystematicsFileInfo gInfoLamK016;
    gInfoLamK016.resultsDate = "20161106";
    gInfoLamK016.dirNameModifierBase1 = "_minAvgSepPosPos_";
    gInfoLamK016.modifierValues1 = vector<double> {5.0, 6.0, 7.0};
    gInfoLamK016.dirNameModifierBase2 = "";
    gInfoLamK016.modifierValues2 = vector<double> {};
    gInfoLamK016.allCentralities = true;

  SystematicsFileInfo gInfoLamK017;
    gInfoLamK017.resultsDate = "20161108";
    gInfoLamK017.dirNameModifierBase1 = "_minAvgSepPosPos_";
    gInfoLamK017.modifierValues1 = vector<double> {5.5, 6.0, 6.5};
    gInfoLamK017.dirNameModifierBase2 = "";
    gInfoLamK017.modifierValues2 = vector<double> {};
    gInfoLamK017.allCentralities = true;


  //----------------------------------------------------------------------

  if(aNumber==1) return gInfoLamKch1;
  else if(aNumber==2) return gInfoLamKch2;
  else if(aNumber==3) return gInfoLamKch3;
  else if(aNumber==4) return gInfoLamKch4;
  else if(aNumber==5) return gInfoLamKch5;
  else if(aNumber==6) return gInfoLamKch6;
  else if(aNumber==7) return gInfoLamKch7;
  else if(aNumber==8) return gInfoLamKch8;
  else if(aNumber==9) return gInfoLamKch9;
  else if(aNumber==10) return gInfoLamKch10;
  else if(aNumber==11) return gInfoLamKch11;
  else if(aNumber==12) return gInfoLamKch12;

  else if(aNumber==-1) return gInfoLamK01;
  else if(aNumber==-2) return gInfoLamK02;
  else if(aNumber==-3) return gInfoLamK03;
  else if(aNumber==-4) return gInfoLamK04;
  else if(aNumber==-5) return gInfoLamK05;
  else if(aNumber==-6) return gInfoLamK06;
  else if(aNumber==-7) return gInfoLamK07;
  else if(aNumber==-8) return gInfoLamK08;
  else if(aNumber==-9) return gInfoLamK09;
  else if(aNumber==-10) return gInfoLamK010;
  else if(aNumber==-11) return gInfoLamK011;
  else if(aNumber==-12) return gInfoLamK012;
  else if(aNumber==-13) return gInfoLamK013;
  else if(aNumber==-14) return gInfoLamK014;
  else if(aNumber==-15) return gInfoLamK015;
  else if(aNumber==-16) return gInfoLamK016;
  else if(aNumber==-17) return gInfoLamK017;

  else
  {
    cout << "ERROR: SystematicsFileInfo GetFileInfo" << endl;
    assert(0);
    return gInfoLamKch1;
  }
}



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

  AnalysisType tCurrentAnType;
  CentralityType tCurrentCentralityType;
  ParameterType tCurrentParameterType;

  std::string tStdString;
  TString tLine;\
  while(getline(tFileIn, tStdString))
  {
    tLine = TString(tStdString);

    if(tLine.Contains("AnalysisType")) tCurrentAnType = GetAnalysisType(tLine);
    if(tLine.Contains("Centrality Type")) tCurrentCentralityType = GetCentralityType(tLine);
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
        cout << endl;
      }
      cout << "------------------------------------------------------------" << endl;
    }
  }
}


int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------
/*
  td2dVec tLamK00010(5);
  td2dVec tLamK01030(5);
  td2dVec tLamK03050(5);

  td2dVec tALamK00010(5);
  td2dVec tALamK01030(5);
  td2dVec tALamK03050(5);

  td2dVec tLamKchP0010(5);
  td2dVec tLamKchP1030(5);
  td2dVec tLamKchP3050(5);

  td2dVec tALamKchM0010(5);
  td2dVec tALamKchM1030(5);
  td2dVec tALamKchM3050(5);

  td2dVec tLamKchM0010(5);
  td2dVec tLamKchM1030(5);
  td2dVec tLamKchM3050(5);

  td2dVec tALamKchP0010(5);
  td2dVec tALamKchP1030(5);
  td2dVec tALamKchP3050(5);

  //-------------------------------
  td3dVec tLamK0;
    tLamK0.push_back(tLamK00010);
    tLamK0.push_back(tLamK01030);
    tLamK0.push_back(tLamK03050);

  td3dVec tALamK0;
    tALamK0.push_back(tALamK00010);
    tALamK0.push_back(tALamK01030);
    tALamK0.push_back(tALamK03050);

  td3dVec tLamKchP;
    tLamKchP.push_back(tLamKchP0010);
    tLamKchP.push_back(tLamKchP1030);
    tLamKchP.push_back(tLamKchP3050);

  td3dVec tALamKchM;
    tALamKchM.push_back(tALamKchM0010);
    tALamKchM.push_back(tALamKchM1030);
    tALamKchM.push_back(tALamKchM3050);

  td3dVec tLamKchM;
    tLamKchM.push_back(tLamKchM0010);
    tLamKchM.push_back(tLamKchM1030);
    tLamKchM.push_back(tLamKchM3050);

  td3dVec tALamKchP;
    tALamKchP.push_back(tALamKchP0010);
    tALamKchP.push_back(tALamKchP1030);
    tALamKchP.push_back(tALamKchP3050);

  //-------------------------------
  td4dVec tAll;
    tAll.push_back(tLamK0);
    tAll.push_back(tALamK0);
    tAll.push_back(tLamKchP);
    tAll.push_back(tALamKchP);
    tAll.push_back(tLamKchM);
    tAll.push_back(tALamKchM);

  //-------------------------------
*/

  //-------------------------------
  int tNAnalysisTypes = 6; //kLamK0, kALamK0, kLamKchP, kALamKchP, kLamKchM, kALamKchM
  int tNCentralityTypes = 3; //k0010, k1030, k3050
  int tNParameterTypes = 5; //kLambda, kRadius, kRef0, kImf0, kd0

  td4dVec tAll(0);
    tAll.resize(tNAnalysisTypes, td3dVec(tNCentralityTypes, td2dVec(tNParameterTypes, td1dVec(0))));

  TString tFileName = "/home/jesse/Analysis/FemtoAnalysis/Results/Systematics/Results_cLamcKch_Systematics_CLAM_maxDcaV0_20161026/CfFitValues_LamKchP_MomResCrctn_NonFlatBgdCrctn.txt";

  for(int iCut=1; iCut<=12; iCut++)
  {
    if(iCut==6) continue;
    cout << "iCut = " << iCut << endl;

    SystematicsFileInfo tFileInfo = GetFileInfo(iCut);
      TString tResultsDate = tFileInfo.resultsDate;
      TString tDirNameModifierBase1 = tFileInfo.dirNameModifierBase1;
      vector<double> tModifierValues1 = tFileInfo.modifierValues1;
      TString tDirNameModifierBase2 = tFileInfo.dirNameModifierBase2;
      vector<double> tModifierValues2 = tFileInfo.modifierValues2;

    TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Systematics/Results_%s_Systematics%s","cLamcKch",tDirNameModifierBase1.Data());
    if(!tDirNameModifierBase2.IsNull())
    {
      tDirectoryBase.Remove(TString::kTrailing,'_');
      tDirectoryBase += tDirNameModifierBase2;
    }
    tDirectoryBase += TString::Format("%s/",tResultsDate.Data());

    TString tFileLocationBase1 = tDirectoryBase + TString::Format("CfFitValues_%s_MomResCrctn_NonFlatBgdCrctn.txt","LamKchP");
    TString tFileLocationBase2 = tDirectoryBase + TString::Format("CfFitValues_%s_MomResCrctn_NonFlatBgdCrctn.txt","LamKchM");

    ReadFile(tFileLocationBase1,tAll);
    ReadFile(tFileLocationBase2,tAll);
  }


  for(int iCut=1; iCut<=17; iCut++)
  {
    if(iCut==9) continue;
    int tCut = -1*iCut;
    cout << "tCut = " << tCut << endl;

    SystematicsFileInfo tFileInfo = GetFileInfo(tCut);
      TString tResultsDate = tFileInfo.resultsDate;
      TString tDirNameModifierBase1 = tFileInfo.dirNameModifierBase1;
      vector<double> tModifierValues1 = tFileInfo.modifierValues1;
      TString tDirNameModifierBase2 = tFileInfo.dirNameModifierBase2;
      vector<double> tModifierValues2 = tFileInfo.modifierValues2;

    TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Systematics/Results_%s_Systematics%s","cLamK0",tDirNameModifierBase1.Data());
    if(!tDirNameModifierBase2.IsNull())
    {
      tDirectoryBase.Remove(TString::kTrailing,'_');
      tDirectoryBase += tDirNameModifierBase2;
    }
    tDirectoryBase += TString::Format("%s/",tResultsDate.Data());

    TString tFileLocationBase = tDirectoryBase + TString::Format("CfFitValues_%s_MomResCrctn_NonFlatBgdCrctn.txt","LamK0");

    ReadFile(tFileLocationBase,tAll);
  }


  td4dVec tFinalVec = ReduceCutsVector(tAll);
  PrintFinalVec(tFinalVec);






cout << "DONE" << endl;
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
