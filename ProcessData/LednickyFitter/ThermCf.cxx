/* ThermCf.cxx */

#include "ThermCf.h"

#ifdef __ROOT__
ClassImp(ThermCf)
#endif




//________________________________________________________________________________________________________________
ThermCf::ThermCf(TString aFileName, TString aCfDescriptor, AnalysisType aAnalysisType, CentralityType aCentralityType, bool aCombineConj, ThermEventsType aThermEventsType, int aRebin, double aMinNorm, double aMaxNorm, bool aUseNumRotPar2InsteadOfDen) : 
  fFileName(aFileName),
  fCfDescriptor(aCfDescriptor),

  fAnalysisType(aAnalysisType),
  fCentralityType(aCentralityType),

  fCombineConjugates(aCombineConj),
  fCombineLamKchPM(false),
  fThermEventsType(aThermEventsType), 

  fCombineImpactParams(true),
  fImpactParam(-1),

  fRebin(aRebin),
  fMinNorm(aMinNorm),
  fMaxNorm(aMaxNorm),

  fUseNumRotPar2InsteadOfDen(aUseNumRotPar2InsteadOfDen),

  fThermCfHeavy(nullptr)

{
  if(fCombineLamKchPM) assert(fAnalysisType==kLamKchP || fAnalysisType==kALamKchM ||
                              fAnalysisType==kLamKchM || fAnalysisType==kALamKchP);

  BuildThermCf();
}





//________________________________________________________________________________________________________________
ThermCf::~ThermCf()
{
  //no-op
}




//________________________________________________________________________________________________________________
void ThermCf::SetStyleAndColor(TH1* aHist, int aMarkerStyle, int aColor, double aMarkerSize)
{
  aHist->SetLineColor(aColor);
  aHist->SetMarkerColor(aColor);
  aHist->SetMarkerStyle(aMarkerStyle);
  aHist->SetMarkerSize(aMarkerSize);
}

//________________________________________________________________________________________________________________
TH1* ThermCf::GetThermHist(TString aFileLocation, TString aHistName)
{
  TFile *tFile = TFile::Open(aFileLocation);

  TH1 *tReturnHist = (TH1*)tFile->Get(aHistName);
  TH1 *tReturnHistClone = (TH1*)tReturnHist->Clone();
  tReturnHistClone->SetDirectory(0);

  tFile->Close();
  delete tFile;

  return tReturnHistClone;
}

//________________________________________________________________________________________________________________
CfHeavy* ThermCf::CombineTwoCfHeavy(TString aName, CfHeavy* aCfHeavy1, CfHeavy* aCfHeavy2)
{
  vector<CfLite*> tCfLiteVec1 = aCfHeavy1->GetCfLiteCollection();
  vector<CfLite*> tCfLiteVec2 = aCfHeavy2->GetCfLiteCollection();

  vector<CfLite*> tCfLiteVec(0);
  for(unsigned int i=0; i<tCfLiteVec1.size(); i++) tCfLiteVec.push_back(tCfLiteVec1[i]);
  for(unsigned int i=0; i<tCfLiteVec2.size(); i++) tCfLiteVec.push_back(tCfLiteVec2[i]);

  double tMinNorm = aCfHeavy1->GetMinNorm();
  assert(tMinNorm == aCfHeavy2->GetMinNorm());
  double tMaxNorm = aCfHeavy1->GetMaxNorm();
  assert(tMaxNorm == aCfHeavy2->GetMaxNorm());

  CfHeavy* tCfHeavy = new CfHeavy(aName, aName, tCfLiteVec, tMinNorm, tMaxNorm);
  return tCfHeavy;
}


//________________________________________________________________________________________________________________
CfHeavy* ThermCf::GetThermHeavyCf(TString aFileName, TString aCfDescriptor, AnalysisType aAnType, int aImpactParam, bool aCombineConj, bool aUseAdamEvents, int aRebin, double aMinNorm, double aMaxNorm, bool aUseNumRotPar2InsteadOfDen)
{
  AnalysisType tConjAnType;
  if     (aAnType==kLamK0)    {tConjAnType=kALamK0;}
  else if(aAnType==kALamK0)   {tConjAnType=kLamK0;}
  else if(aAnType==kLamKchP)  {tConjAnType=kALamKchM;}
  else if(aAnType==kALamKchM) {tConjAnType=kLamKchP;}
  else if(aAnType==kLamKchM)  {tConjAnType=kALamKchP;}
  else if(aAnType==kALamKchP) {tConjAnType=kLamKchM;}
  else assert(0);

  //--------------------------------
  TString tDirectory;
  if(!aUseAdamEvents) tDirectory = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/", aImpactParam);
  else                tDirectory = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/AdamCPUResults/lhyqid3v_LHCPbPb_2760_b%d/", aImpactParam);
  TString tFileLocation = TString::Format("%s%s", tDirectory.Data(), aFileName.Data());
  //--------------------------------
  TH1* tNum1 = GetThermHist(tFileLocation, TString::Format("Num%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[aAnType]));
  TH1* tDen1;
  if(!aUseNumRotPar2InsteadOfDen) tDen1 = GetThermHist(tFileLocation, TString::Format("Den%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[aAnType]));
  else tDen1 = GetThermHist(tFileLocation, TString::Format("Num%s_RotatePar2%s", aCfDescriptor.Data(), cAnalysisBaseTags[aAnType]));
  CfLite* tCfLite1 = new CfLite(TString::Format("CfLite_%s", cAnalysisBaseTags[aAnType]), 
                                TString::Format("CfLite_%s", cAnalysisBaseTags[aAnType]), 
                                tNum1, tDen1, aMinNorm, aMaxNorm);
  tCfLite1->Rebin(aRebin);

  TH1* tNum2 = GetThermHist(tFileLocation, TString::Format("Num%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[tConjAnType]));
  TH1* tDen2;
  if(!aUseNumRotPar2InsteadOfDen) tDen2 = GetThermHist(tFileLocation, TString::Format("Den%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[tConjAnType]));
  else tDen2 = GetThermHist(tFileLocation, TString::Format("Num%s_RotatePar2%s", aCfDescriptor.Data(), cAnalysisBaseTags[tConjAnType]));
  CfLite* tCfLite2 = new CfLite(TString::Format("CfLite_%s", cAnalysisBaseTags[tConjAnType]), 
                                TString::Format("CfLite_%s", cAnalysisBaseTags[tConjAnType]), 
                                tNum2, tDen2, aMinNorm, aMaxNorm);
  tCfLite2->Rebin(aRebin);

  CfHeavy* tCfHeavy;
  if(!aCombineConj)
  {
    vector<CfLite*> tCfLiteVec {tCfLite1};
    tCfHeavy = new CfHeavy(TString::Format("CfHeavy_%s", cAnalysisBaseTags[aAnType]), 
                           TString::Format("CfHeavy_%s", cAnalysisBaseTags[aAnType]), 
                           tCfLiteVec, aMinNorm, aMaxNorm);
  }
  else
  {
    vector<CfLite*> tCfLiteVec {tCfLite1, tCfLite2};
    tCfHeavy = new CfHeavy(TString::Format("CfHeavy_%s_%s", cAnalysisBaseTags[aAnType], cAnalysisBaseTags[tConjAnType]), 
                           TString::Format("CfHeavy_%s_%s", cAnalysisBaseTags[aAnType], cAnalysisBaseTags[tConjAnType]), 
                           tCfLiteVec, aMinNorm, aMaxNorm);
  }

  return tCfHeavy;
}


//________________________________________________________________________________________________________________
CfHeavy* ThermCf::GetThermHeavyCf(TString aFileName, TString aCfDescriptor, AnalysisType aAnType, int aImpactParam, bool aCombineConj, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, bool aUseNumRotPar2InsteadOfDen)
{
  CfHeavy* tCfHeavy;
  if(aEventsType==kMe)        tCfHeavy = GetThermHeavyCf(aFileName, aCfDescriptor, aAnType, aImpactParam, aCombineConj, false, aRebin, aMinNorm, aMaxNorm, aUseNumRotPar2InsteadOfDen);
  else if(aEventsType==kAdam) tCfHeavy = GetThermHeavyCf(aFileName, aCfDescriptor, aAnType, aImpactParam, aCombineConj, true, aRebin, aMinNorm, aMaxNorm, aUseNumRotPar2InsteadOfDen);
  else
  {
    CfHeavy* tCfHeavy_Me   = GetThermHeavyCf(aFileName, aCfDescriptor, aAnType, aImpactParam, aCombineConj, false, aRebin, aMinNorm, aMaxNorm, aUseNumRotPar2InsteadOfDen);
    CfHeavy* tCfHeavy_Adam = GetThermHeavyCf(aFileName, aCfDescriptor, aAnType, aImpactParam, aCombineConj, true, aRebin, aMinNorm, aMaxNorm, aUseNumRotPar2InsteadOfDen);

    TString tName = tCfHeavy_Me->GetHeavyCfName();
    tName += TString(cThermEventsTypeTags[aEventsType]);

    tCfHeavy = CombineTwoCfHeavy(tName, tCfHeavy_Me, tCfHeavy_Adam);
  }
  return tCfHeavy;
}


//________________________________________________________________________________________________________________
TH1* ThermCf::GetThermCf(TString aFileName, TString aCfDescriptor, AnalysisType aAnType, int aImpactParam, bool aCombineConj, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, int aMarkerStyle, int aColor, bool aUseNumRotPar2InsteadOfDen)
{
  CfHeavy* tCfHeavy;
  if(aEventsType==kMe)        tCfHeavy = GetThermHeavyCf(aFileName, aCfDescriptor, aAnType, aImpactParam, aCombineConj, false, aRebin, aMinNorm, aMaxNorm, aUseNumRotPar2InsteadOfDen);
  else if(aEventsType==kAdam) tCfHeavy = GetThermHeavyCf(aFileName, aCfDescriptor, aAnType, aImpactParam, aCombineConj, true, aRebin, aMinNorm, aMaxNorm, aUseNumRotPar2InsteadOfDen);
  else
  {
    CfHeavy* tCfHeavy_Me   = GetThermHeavyCf(aFileName, aCfDescriptor, aAnType, aImpactParam, aCombineConj, false, aRebin, aMinNorm, aMaxNorm, aUseNumRotPar2InsteadOfDen);
    CfHeavy* tCfHeavy_Adam = GetThermHeavyCf(aFileName, aCfDescriptor, aAnType, aImpactParam, aCombineConj, true, aRebin, aMinNorm, aMaxNorm, aUseNumRotPar2InsteadOfDen);

    TString tName = tCfHeavy_Me->GetHeavyCfName();
    tName += TString(cThermEventsTypeTags[aEventsType]);

    tCfHeavy = CombineTwoCfHeavy(tName, tCfHeavy_Me, tCfHeavy_Adam);
  }

  TH1* tReturnHist = tCfHeavy->GetHeavyCf();

  SetStyleAndColor(tReturnHist, aMarkerStyle, aColor);
  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1* ThermCf::GetThermCf(AnalysisType aAnType, int aImpactParam, bool aCombineConj, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, int aMarkerStyle, int aColor, bool aUseNumRotPar2InsteadOfDen)
{
  TString tFileName = "CorrelationFunctions_RandomEPs_NumWeight1.root";
  TString tCfDescriptor = "Full";
  return GetThermCf(tFileName, tCfDescriptor, aAnType, aImpactParam, aCombineConj, aEventsType, aRebin, aMinNorm, aMaxNorm, aMarkerStyle, aColor, aUseNumRotPar2InsteadOfDen);
}

//________________________________________________________________________________________________________________
CfHeavy* ThermCf::GetCentralityCombinedThermCfsHeavy(TString aFileName, TString aCfDescriptor, AnalysisType aAnType, CentralityType aCentType, bool aCombineConj, bool aUseAdamEvents, int aRebin, double aMinNorm, double aMaxNorm, bool aUseNumRotPar2InsteadOfDen)
{
  AnalysisType tConjAnType;
  if     (aAnType==kLamK0)    {tConjAnType=kALamK0;}
  else if(aAnType==kALamK0)   {tConjAnType=kLamK0;}
  else if(aAnType==kLamKchP)  {tConjAnType=kALamKchM;}
  else if(aAnType==kALamKchM) {tConjAnType=kLamKchP;}
  else if(aAnType==kLamKchM)  {tConjAnType=kALamKchP;}
  else if(aAnType==kALamKchP) {tConjAnType=kLamKchM;}
  else assert(0);

  //--------------------------------
  vector<int> tImpactParams;
  if     (aCentType == k0010) tImpactParams = vector<int>{3};
  else if(aCentType == k1030) tImpactParams = vector<int>{5, 7};
  else if(aCentType == k3050) tImpactParams = vector<int>{8, 9};
  else assert(0);
  //--------------------------------
  TString tDirectory, tFileLocation;
  vector<CfLite*> tCfLiteVec;
  for(unsigned int iIP=0; iIP < tImpactParams.size(); iIP++)
  {
    if(!aUseAdamEvents) tDirectory = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/", tImpactParams[iIP]);
    else                tDirectory = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/AdamCPUResults/lhyqid3v_LHCPbPb_2760_b%d/", tImpactParams[iIP]);
    tFileLocation = TString::Format("%s%s", tDirectory.Data(), aFileName.Data());
    //--------------------------------
    TH1* tNum1 = GetThermHist(tFileLocation, TString::Format("Num%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[aAnType]));
    TH1* tDen1;
    if(!aUseNumRotPar2InsteadOfDen) tDen1 = GetThermHist(tFileLocation, TString::Format("Den%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[aAnType]));
    else tDen1 = GetThermHist(tFileLocation, TString::Format("Num%s_RotatePar2%s", aCfDescriptor.Data(), cAnalysisBaseTags[aAnType]));
    CfLite* tCfLite1 = new CfLite(TString::Format("CfLite_%s_%d", cAnalysisBaseTags[aAnType], tImpactParams[iIP]), 
                                  TString::Format("CfLite_%s_%d", cAnalysisBaseTags[aAnType], tImpactParams[iIP]), 
                                  tNum1, tDen1, aMinNorm, aMaxNorm);
    tCfLite1->Rebin(aRebin);
    tCfLiteVec.push_back(tCfLite1);

    if(aCombineConj)
    {
      TH1* tNum2 = GetThermHist(tFileLocation, TString::Format("Num%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[tConjAnType]));
      TH1* tDen2;
      if(!aUseNumRotPar2InsteadOfDen) tDen2 = GetThermHist(tFileLocation, TString::Format("Den%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[tConjAnType]));
      else tDen2 = GetThermHist(tFileLocation, TString::Format("Num%s_RotatePar2%s", aCfDescriptor.Data(), cAnalysisBaseTags[tConjAnType]));
      CfLite* tCfLite2 = new CfLite(TString::Format("CfLite_%s_%d", cAnalysisBaseTags[tConjAnType], tImpactParams[iIP]), 
                                    TString::Format("CfLite_%s_%d", cAnalysisBaseTags[tConjAnType], tImpactParams[iIP]), 
                                    tNum2, tDen2, aMinNorm, aMaxNorm);
      tCfLite2->Rebin(aRebin);
      tCfLiteVec.push_back(tCfLite2);
    }
  }

  CfHeavy* tCfHeavy = new CfHeavy(TString::Format("CfHeavy_%s_%s_%s", cAnalysisBaseTags[aAnType], cAnalysisBaseTags[tConjAnType], cCentralityTags[aCentType]), 
                                  TString::Format("CfHeavy_%s_%s_%s", cAnalysisBaseTags[aAnType], cAnalysisBaseTags[tConjAnType], cCentralityTags[aCentType]), 
                                  tCfLiteVec, aMinNorm, aMaxNorm);

  return tCfHeavy;
}

//________________________________________________________________________________________________________________
CfHeavy* ThermCf::GetCentralityCombinedThermCfsHeavy(TString aFileName, TString aCfDescriptor, AnalysisType aAnType, CentralityType aCentType, bool aCombineConj, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, bool aUseNumRotPar2InsteadOfDen)
{
  CfHeavy* tCfHeavy;
  if(aEventsType==kMe)        tCfHeavy = GetCentralityCombinedThermCfsHeavy(aFileName, aCfDescriptor, aAnType, aCentType, aCombineConj, false, aRebin, aMinNorm, aMaxNorm, aUseNumRotPar2InsteadOfDen);
  else if(aEventsType==kAdam) tCfHeavy = GetCentralityCombinedThermCfsHeavy(aFileName, aCfDescriptor, aAnType, aCentType, aCombineConj, true, aRebin, aMinNorm, aMaxNorm, aUseNumRotPar2InsteadOfDen);
  else
  {
    CfHeavy* tCfHeavy_Me   = GetCentralityCombinedThermCfsHeavy(aFileName, aCfDescriptor, aAnType, aCentType, aCombineConj, false, aRebin, aMinNorm, aMaxNorm, aUseNumRotPar2InsteadOfDen);
    CfHeavy* tCfHeavy_Adam = GetCentralityCombinedThermCfsHeavy(aFileName, aCfDescriptor, aAnType, aCentType, aCombineConj, true, aRebin, aMinNorm, aMaxNorm, aUseNumRotPar2InsteadOfDen);

    TString tName = tCfHeavy_Me->GetHeavyCfName();
    tName += TString(cThermEventsTypeTags[aEventsType]);

    tCfHeavy = CombineTwoCfHeavy(tName, tCfHeavy_Me, tCfHeavy_Adam);
  }
  return tCfHeavy;
}


//________________________________________________________________________________________________________________
TH1* ThermCf::GetCentralityCombinedThermCfs(TString aFileName, TString aCfDescriptor, AnalysisType aAnType, CentralityType aCentType, bool aCombineConj, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, int aMarkerStyle, int aColor, bool aUseNumRotPar2InsteadOfDen)
{
  CfHeavy* tCfHeavy = GetCentralityCombinedThermCfsHeavy(aFileName, aCfDescriptor, aAnType, aCentType, aCombineConj, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseNumRotPar2InsteadOfDen);
  TH1* tReturnHist = tCfHeavy->GetHeavyCf();
  SetStyleAndColor(tReturnHist, aMarkerStyle, aColor);
  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1* ThermCf::GetCentralityCombinedThermCfs(AnalysisType aAnType, CentralityType aCentType, bool aCombineConj, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, int aMarkerStyle, int aColor, bool aUseNumRotPar2InsteadOfDen)
{
  TString tFileName = "CorrelationFunctions_RandomEPs_NumWeight1.root";
  TString tCfDescriptor = "Full";
  return GetCentralityCombinedThermCfs(tFileName, tCfDescriptor, aAnType, aCentType, aCombineConj, aEventsType, aRebin, aMinNorm, aMaxNorm, aMarkerStyle, aColor, aUseNumRotPar2InsteadOfDen);
}


//________________________________________________________________________________________________________________
CfHeavy* ThermCf::GetLamKchPMCombinedThermCfsHeavy(TString aFileName, TString aCfDescriptor, CentralityType aCentType, bool aUseAdamEvents, int aRebin, double aMinNorm, double aMaxNorm, bool aUseNumRotPar2InsteadOfDen)
{
  vector<int> tImpactParams;
  if     (aCentType == k0010) tImpactParams = vector<int>{3};
  else if(aCentType == k1030) tImpactParams = vector<int>{5, 7};
  else if(aCentType == k3050) tImpactParams = vector<int>{8, 9};
  else assert(0);
  //--------------------------------
  TString tDirectory, tFileLocation;
  vector<CfLite*> tCfLiteVec;
  for(unsigned int iIP=0; iIP < tImpactParams.size(); iIP++)
  {
    if(!aUseAdamEvents) tDirectory = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/", tImpactParams[iIP]);
    else                tDirectory = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/AdamCPUResults/lhyqid3v_LHCPbPb_2760_b%d/", tImpactParams[iIP]);
    tFileLocation = TString::Format("%s%s", tDirectory.Data(), aFileName.Data());
    //--------------------------------
    TH1* tNum1 = GetThermHist(tFileLocation, TString::Format("Num%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[kLamKchP]));
    TH1* tDen1;
    if(!aUseNumRotPar2InsteadOfDen) tDen1 = GetThermHist(tFileLocation, TString::Format("Den%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[kLamKchP]));
    else tDen1 = GetThermHist(tFileLocation, TString::Format("Num%s_RotatePar2%s", aCfDescriptor.Data(), cAnalysisBaseTags[kLamKchP]));
    CfLite* tCfLite1 = new CfLite(TString::Format("CfLite_%s_%d", cAnalysisBaseTags[kLamKchP], tImpactParams[iIP]), 
                                  TString::Format("CfLite_%s_%d", cAnalysisBaseTags[kLamKchP], tImpactParams[iIP]), 
                                  tNum1, tDen1, aMinNorm, aMaxNorm);
    tCfLite1->Rebin(aRebin);
    tCfLiteVec.push_back(tCfLite1);

    TH1* tNum2 = GetThermHist(tFileLocation, TString::Format("Num%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[kALamKchM]));
    TH1* tDen2;
    if(!aUseNumRotPar2InsteadOfDen) tDen2 = GetThermHist(tFileLocation, TString::Format("Den%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[kALamKchM]));
    else tDen2 = GetThermHist(tFileLocation, TString::Format("Num%s_RotatePar2%s", aCfDescriptor.Data(), cAnalysisBaseTags[kALamKchM]));
    CfLite* tCfLite2 = new CfLite(TString::Format("CfLite_%s_%d", cAnalysisBaseTags[kALamKchM], tImpactParams[iIP]), 
                                  TString::Format("CfLite_%s_%d", cAnalysisBaseTags[kALamKchM], tImpactParams[iIP]), 
                                  tNum2, tDen2, aMinNorm, aMaxNorm);
    tCfLite2->Rebin(aRebin);
    tCfLiteVec.push_back(tCfLite2);
    //--------------------------------
    TH1* tNum3 = GetThermHist(tFileLocation, TString::Format("Num%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[kLamKchM]));
    TH1* tDen3;
    if(!aUseNumRotPar2InsteadOfDen) tDen3 = GetThermHist(tFileLocation, TString::Format("Den%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[kLamKchM]));
    else tDen3 = GetThermHist(tFileLocation, TString::Format("Num%s_RotatePar2%s", aCfDescriptor.Data(), cAnalysisBaseTags[kLamKchM]));
    CfLite* tCfLite3 = new CfLite(TString::Format("CfLite_%s_%d", cAnalysisBaseTags[kLamKchM], tImpactParams[iIP]), 
                                  TString::Format("CfLite_%s_%d", cAnalysisBaseTags[kLamKchM], tImpactParams[iIP]), 
                                  tNum3, tDen3, aMinNorm, aMaxNorm);
    tCfLite3->Rebin(aRebin);
    tCfLiteVec.push_back(tCfLite3);

    TH1* tNum4 = GetThermHist(tFileLocation, TString::Format("Num%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[kALamKchP]));
    TH1* tDen4;
    if(!aUseNumRotPar2InsteadOfDen) tDen4 = GetThermHist(tFileLocation, TString::Format("Den%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[kALamKchP]));
    else tDen4 = GetThermHist(tFileLocation, TString::Format("Num%s_RotatePar2%s", aCfDescriptor.Data(), cAnalysisBaseTags[kALamKchP]));
    CfLite* tCfLite4 = new CfLite(TString::Format("CfLite_%s_%d", cAnalysisBaseTags[kALamKchP], tImpactParams[iIP]), 
                                  TString::Format("CfLite_%s_%d", cAnalysisBaseTags[kALamKchP], tImpactParams[iIP]), 
                                  tNum4, tDen4, aMinNorm, aMaxNorm);
    tCfLite4->Rebin(aRebin);
    tCfLiteVec.push_back(tCfLite4);
  }

  CfHeavy* tCfHeavy = new CfHeavy(TString::Format("CfHeavy_LamKchPM_%s", cCentralityTags[aCentType]), 
                                  TString::Format("CfHeavy_LamKchPM_%s", cCentralityTags[aCentType]), 
                                  tCfLiteVec, aMinNorm, aMaxNorm);
  return tCfHeavy;
}


//________________________________________________________________________________________________________________
CfHeavy* ThermCf::GetLamKchPMCombinedThermCfsHeavy(TString aFileName, TString aCfDescriptor, CentralityType aCentType, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, bool aUseNumRotPar2InsteadOfDen)
{
  CfHeavy* tCfHeavy;
  if(aEventsType==kMe)        tCfHeavy = GetLamKchPMCombinedThermCfsHeavy(aFileName, aCfDescriptor, aCentType, false, aRebin, aMinNorm, aMaxNorm, aUseNumRotPar2InsteadOfDen);
  else if(aEventsType==kAdam) tCfHeavy = GetLamKchPMCombinedThermCfsHeavy(aFileName, aCfDescriptor, aCentType, true, aRebin, aMinNorm, aMaxNorm, aUseNumRotPar2InsteadOfDen);
  else
  {
    CfHeavy* tCfHeavy_Me   = GetLamKchPMCombinedThermCfsHeavy(aFileName, aCfDescriptor, aCentType, false, aRebin, aMinNorm, aMaxNorm, aUseNumRotPar2InsteadOfDen);
    CfHeavy* tCfHeavy_Adam = GetLamKchPMCombinedThermCfsHeavy(aFileName, aCfDescriptor, aCentType, true, aRebin, aMinNorm, aMaxNorm, aUseNumRotPar2InsteadOfDen);

    TString tName = tCfHeavy_Me->GetHeavyCfName();
    tName += TString(cThermEventsTypeTags[aEventsType]);

    tCfHeavy = CombineTwoCfHeavy(tName, tCfHeavy_Me, tCfHeavy_Adam);
  }
  return tCfHeavy;
}


//________________________________________________________________________________________________________________
TH1* ThermCf::GetLamKchPMCombinedThermCfs(TString aFileName, TString aCfDescriptor, CentralityType aCentType, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, int aMarkerStyle, int aColor, bool aUseNumRotPar2InsteadOfDen)
{
  CfHeavy* tCfHeavy = GetLamKchPMCombinedThermCfsHeavy(aFileName, aCfDescriptor, aCentType, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseNumRotPar2InsteadOfDen);
  TH1* tReturnHist = tCfHeavy->GetHeavyCf();
  SetStyleAndColor(tReturnHist, aMarkerStyle, aColor);
  return tReturnHist;
}


//________________________________________________________________________________________________________________
void ThermCf::BuildThermCf()
{
  if(!fCombineImpactParams)
  {
    assert(fImpactParam > 0);
    fThermCfHeavy = GetThermHeavyCf(fFileName, fCfDescriptor, fAnalysisType, fImpactParam, fCombineConjugates, fThermEventsType, fRebin, fMinNorm, fMaxNorm, fUseNumRotPar2InsteadOfDen);
  }
  else if(fCombineLamKchPM)
  {
    assert(fAnalysisType==kLamKchP || fAnalysisType==kALamKchM ||
           fAnalysisType==kLamKchM || fAnalysisType==kALamKchP);
    assert(fCombineImpactParams);

    fThermCfHeavy = GetLamKchPMCombinedThermCfsHeavy(fFileName, fCfDescriptor, fCentralityType, fThermEventsType, fRebin, fMinNorm, fMaxNorm, fUseNumRotPar2InsteadOfDen);
  }
  else
  {
    fThermCfHeavy = GetCentralityCombinedThermCfsHeavy(fFileName, fCfDescriptor, fAnalysisType, fCentralityType, fCombineConjugates, fThermEventsType, fRebin, fMinNorm, fMaxNorm, fUseNumRotPar2InsteadOfDen);
  }

}


//________________________________________________________________________________________________________________
TH1* ThermCf::GetThermCf(int aMarkerStyle, int aColor, double aMarkerSize)
{
  BuildThermCf();
  TH1* tReturnHist = fThermCfHeavy->GetHeavyCf();
  SetStyleAndColor(tReturnHist, aMarkerStyle, aColor, aMarkerSize);
  return tReturnHist;
}


