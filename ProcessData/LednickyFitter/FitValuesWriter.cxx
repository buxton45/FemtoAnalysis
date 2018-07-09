/* FitValuesWriter.cxx */

#include "FitValuesWriter.h"

#ifdef __ROOT__
ClassImp(FitValuesWriter)
#endif




//________________________________________________________________________________________________________________
FitValuesWriter::FitValuesWriter(vector<TString> &aFitParamsTStringVec, TString &aFitInfoTString) : 
  fFitParamsTStringVec(aFitParamsTStringVec),
  fFitInfoTString(aFitInfoTString)

{
  fFitParamsTStringVec.push_back(TString("**********************************************************************************************"));
  fFitParamsTStringVec.push_back(TString(""));
  fFitParamsTStringVec.push_back(TString(""));
}





//________________________________________________________________________________________________________________
FitValuesWriter::~FitValuesWriter()
{
  //no-op
}


//________________________________________________________________________________________________________________
TString FitValuesWriter::GetFitInfoTString(TString aLine)
{
  TObjArray* tContents = aLine.Tokenize("=");
  TString tFitInfo = ((TObjString*)tContents->At(1))->String().Strip(TString::kBoth, ' ');

  return tFitInfo;
}


//________________________________________________________________________________________________________________
AnalysisType FitValuesWriter::GetAnalysisType(TString aLine)
{
  int tBeg = aLine.First("AnalysisType");
  TString tSmallLine = aLine.Remove(0,tBeg);

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

//________________________________________________________________________________________________________________
CentralityType FitValuesWriter::GetCentralityType(TString aLine)
{
  int tBeg = aLine.First("Centrality Type");
  TString tSmallLine = aLine.Remove(0,tBeg);

  TObjArray* tContents = tSmallLine.Tokenize("=");
  TString tCentName = ((TObjString*)tContents->At(1))->String().Strip(TString::kBoth, ' ');

  CentralityType tCentType;
  if(tCentName.EqualTo("0-10%")) tCentType = k0010;
  else if(tCentName.EqualTo("10-30%")) tCentType = k1030;
  else if(tCentName.EqualTo("30-50%")) tCentType = k3050;
  else assert(0);

  return tCentType;
}

//________________________________________________________________________________________________________________
ParameterType FitValuesWriter::GetParamTypeFromName(TString aName)
{
  for(int i=0; i<10; i++) if(aName.EqualTo(cParameterNames[i])) return static_cast<ParameterType>(i);

  assert(0);
  return kLambda;
}


//________________________________________________________________________________________________________________
td1dVec FitValuesWriter::ReadParameterValue(TString aLine)
{
  td1dVec tReturnVec(0);

  TString tSubLine1 = aLine.Strip(TString::kBoth, ' ');
  TString tSubLine2 = TString(tSubLine1);
  TString tSubLine3 = TString(tSubLine1);

  int tBeg1 = 0;
  int tEnd1 = tSubLine1.First(':');
  tSubLine1.Remove(tEnd1, tSubLine1.Length()-tEnd1);

  int tBeg2 = tSubLine2.First(':')+1;
  tSubLine2.Remove(0, tBeg2);
  int tEnd2 = tSubLine2.First("+");
  tSubLine2.Remove(tEnd2, tSubLine2.Length()-tEnd2);

  int tBeg3 = tSubLine3.First("+")+2;
  tSubLine3.Remove(0, tBeg3);
  int tEnd3 = tSubLine3.Length();
  tSubLine3.Remove(tEnd3, tSubLine3.Length()-tEnd3);

  //------------------------------------
cout << "tSubLine1 = " << tSubLine1 << endl;
cout << "tSubLine2 = " << tSubLine2 << endl;
cout << "tSubLine3 = " << tSubLine3 << endl;


  int tParamType = (int)GetParamTypeFromName(tSubLine1);
  double tFitVal = tSubLine2.Atof();
  double tFitErr = tSubLine3.Atof();

  tReturnVec.push_back(tParamType);
  tReturnVec.push_back(tFitVal);
  tReturnVec.push_back(tFitErr);

  return tReturnVec;
}


//________________________________________________________________________________________________________________
void FitValuesWriter::InterpretFitParamsTStringVec()
{

  AnalysisType tCurrentAnType;
  CentralityType tCurrentCentralityType;
  ParameterType tCurrentParameterType;

  TString tLine;
  for(unsigned int i=0; i<fFitParamsTStringVec.size(); i++)
  {
    tLine = fFitParamsTStringVec[i];

    if(tLine.Contains("AnalysisType")) tCurrentAnType = GetAnalysisType(tLine);
    if(tLine.Contains("CentralityType")) tCurrentCentralityType = GetCentralityType(tLine);
    if(tLine.Contains("Lambda") || tLine.Contains("Radius") || tLine.Contains("Ref0") || tLine.Contains("Imf0") || tLine.Contains("d0"))
    {
      td1dVec tValuesVec = ReadParameterValue(tLine);

      cout << "i = " << i << endl;
      cout << "ParamType = " << tValuesVec[0] << "(" << cParameterNames[(int)tValuesVec[0]] << ")" << endl;
      cout << "Value     = " << tValuesVec[1] << endl;
      cout << "Error     = " << tValuesVec[2] << endl << endl;

/*
      if(tLine.Contains("Lambda")) tCurrentParameterType = kLambda;
      else if(tLine.Contains("Radius")) tCurrentParameterType = kRadius;
      else if(tLine.Contains("Ref0")) tCurrentParameterType = kRef0;
      else if(tLine.Contains("Imf0")) tCurrentParameterType = kImf0;
      else if(tLine.Contains("d0")) tCurrentParameterType = kd0;
      else assert(0);
*/
//      FillCutsVector(aAll, tValuesVec, tCurrentParameterType, tCurrentAnType, tCurrentCentralityType);
    }

  }

}


//________________________________________________________________________________________________________________
vector<vector<TString> > FitValuesWriter::ConvertMasterTo2dVec(TString aFileLocation)
{
  ifstream tFileIn(aFileLocation);
  assert(tFileIn);

  vector<TString> tFitParams1dVec(0);
  vector<vector<TString> > tAllFitParams2dVec(0);

  std::string tStdString;
  TString tLine;

  while(getline(tFileIn, tStdString))
  {
    tLine = TString(tStdString);
    if(tLine.Contains("FitInfo"))
    {
      if(tFitParams1dVec.size() > 0) tAllFitParams2dVec.push_back(tFitParams1dVec);
      tFitParams1dVec.clear();

      tFitParams1dVec.push_back(tLine);
    }
    else tFitParams1dVec.push_back(tLine);
  }
  if(tFitParams1dVec.size() > 0) tAllFitParams2dVec.push_back(tFitParams1dVec);

  tFileIn.close();
  return tAllFitParams2dVec;
}



//________________________________________________________________________________________________________________
void FitValuesWriter::WriteToMaster(TString aFileLocation)
{
  vector<vector<TString> > tMaster2dVec = ConvertMasterTo2dVec(aFileLocation);
  bool tResultIncluded = false;

  vector<TString> tVecToInclude = fFitParamsTStringVec;
  tVecToInclude.insert(tVecToInclude.begin(), TString::Format("FitInfo = %s", fFitInfoTString.Data()));

  for(unsigned int i=0; i<tMaster2dVec.size(); i++)
  {
    if( tMaster2dVec[i][0].EqualTo(tVecToInclude[0])   //Is fit info the same?
     && tMaster2dVec[i][2].EqualTo(tVecToInclude[2])   //Is first analysis type the same?
     && tMaster2dVec[i].size()==tVecToInclude.size())  //Are there the same number of line, i.e. same number of analyses
    {
      tMaster2dVec[i] = tVecToInclude;
      tResultIncluded = true;
    }
  }
  if(!tResultIncluded) //New result did not replace something in Master, so add to end
  {
    tMaster2dVec.push_back(tVecToInclude);
  }

  //------------------------------------------------------------------
  std::ofstream tOutputFile;
  tOutputFile.open(aFileLocation);

  for(unsigned int iAn=0; iAn<tMaster2dVec.size(); iAn++)
  {
    for(unsigned int iLine=0; iLine<tMaster2dVec[iAn].size(); iLine++)
    {
      tOutputFile << TString(tMaster2dVec[iAn][iLine]) << endl;
    }
  }

}




















