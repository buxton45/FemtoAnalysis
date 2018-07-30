/* FitValuesWriterwErrs.cxx */

#include "FitValuesWriterwErrs.h"

#ifdef __ROOT__
ClassImp(FitValuesWriterwErrs)
#endif




//________________________________________________________________________________________________________________
FitValuesWriterwErrs::FitValuesWriterwErrs(TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString) : 
  FitValuesWriter(),

  fMasterFileLocation(aMasterFileLocation),
  fSystematicsFileLocation(aSystematicsFileLocation),
  fFitInfoTString(aFitInfoTString),

  fResType(kInclude3Residuals)
{

}





//________________________________________________________________________________________________________________
FitValuesWriterwErrs::~FitValuesWriterwErrs()
{
  //no-op
}


//________________________________________________________________________________________________________________
AnalysisType FitValuesWriterwErrs::GetAnalysisType(TString aLine)
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

//________________________________________________________________________________________________________________
td1dVec FitValuesWriterwErrs::ReadParameterValue(TString aLine)
{
  td1dVec tReturnVec(0);

  assert(aLine.Contains("Error") && !aLine.Contains("%"));

  TObjArray* tCuts = aLine.Tokenize('|');
  assert(tCuts->GetEntries()==4);

  TString tErr0010 = ((TObjString*)tCuts->At(0))->String().Strip(TString::kBoth, ' ');
  TString tErr1030 = ((TObjString*)tCuts->At(1))->String().Strip(TString::kBoth, ' ');
  TString tErr3050 = ((TObjString*)tCuts->At(2))->String().Strip(TString::kBoth, ' ');

  int tBeg1 = tErr0010.First('=');
  tErr0010.Remove(0,tBeg1+1);
  tErr0010.Strip(TString::kBoth, ' ');

  int tBeg2 = tErr1030.First('=');
  tErr1030.Remove(0,tBeg2+1);
  tErr1030.Strip(TString::kBoth, ' ');

  int tBeg3 = tErr3050.First('=');
  tErr3050.Remove(0,tBeg3+1);
  tErr3050.Strip(TString::kBoth, ' ');

  tReturnVec.push_back(tErr0010.Atof());
  tReturnVec.push_back(tErr1030.Atof());
  tReturnVec.push_back(tErr3050.Atof());

  return tReturnVec;
}


//________________________________________________________________________________________________________________
vector<vector<FitParameter*> > FitValuesWriterwErrs::ReadAllParameters(AnalysisType aAnType)
{
  vector<vector<FitParameter*> > tReturn2dVec(0);
  //----- First, use FitValuesWriter::GetFitResults to build tReturn2dVec without systematic errors
  for(int iCent=0; iCent<(int)kMB; iCent++)
  {
    tReturn2dVec.push_back(GetFitResults(fMasterFileLocation, fFitInfoTString, aAnType, static_cast<CentralityType>(iCent)));
  }

  //----- Now, add systematic errors
  ifstream tSysFile(fSystematicsFileLocation);
  if(!tSysFile.is_open()) cout << "FAILURE - FILE NOT OPEN: " << fSystematicsFileLocation << endl;
  assert(tSysFile.is_open());

  AnalysisType tCurrentAnType;
  ParameterType tCurrentParameterType;
  td1dVec tSysErrVals(0);

  std::string tStdString;
  TString tLine;
  while(getline(tSysFile, tStdString))
  {
    tLine = TString(tStdString);

    if(tLine.Contains("AnalysisType")) tCurrentAnType = GetAnalysisType(tLine);
    if(tLine.Contains("Lambda")) tCurrentParameterType = kLambda;
    if(tLine.Contains("Radius")) tCurrentParameterType = kRadius;
    if(tLine.Contains("Ref0")) tCurrentParameterType = kRef0;
    if(tLine.Contains("Imf0")) tCurrentParameterType = kImf0;
    if(tLine.Contains("d0")) tCurrentParameterType = kd0;
    if(tLine.Contains("Error") && !tLine.Contains("%") && tCurrentAnType==aAnType)
    {
      tSysErrVals = ReadParameterValue(tLine);
      tReturn2dVec[k0010][tCurrentParameterType]->SetFitValueSysError(tSysErrVals[0]);
      tReturn2dVec[k1030][tCurrentParameterType]->SetFitValueSysError(tSysErrVals[1]);
      tReturn2dVec[k3050][tCurrentParameterType]->SetFitValueSysError(tSysErrVals[2]);
    }
  }
  tSysFile.close();

  cout << "AnalysisType = " << cAnalysisBaseTags[aAnType] << endl;
  for(int iCent=0; iCent<tReturn2dVec.size(); iCent++)
  {
    cout << "CentralityType = " << cCentralityTags[iCent] << endl;
    for(unsigned int iParam=0; iParam<tReturn2dVec[iCent].size(); iParam++)
    {
      cout << TString::Format("%s = %0.3f +- %0.3f +- %0.3f", cParameterNames[tReturn2dVec[iCent][iParam]->GetType()], 
                                                              tReturn2dVec[iCent][iParam]->GetFitValue(), 
                                                              tReturn2dVec[iCent][iParam]->GetFitValueError(), 
                                                              tReturn2dVec[iCent][iParam]->GetFitValueSysError()) ;
      cout << endl;
    }
    cout << endl;
  }

  return tReturn2dVec;
}





