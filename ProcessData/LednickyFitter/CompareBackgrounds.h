#include "FitGenerator.h"
class FitGenerator;

//________________________________________________________________________________________________________________
void SetStyleAndColor(TH1* aHist, int aMarkerStyle, int aColor, double aMarkerSize=0.75)
{
  aHist->SetLineColor(aColor);
  aHist->SetMarkerColor(aColor);
  aHist->SetMarkerStyle(aMarkerStyle);
  aHist->SetMarkerSize(aMarkerSize);
}

//_________________________________________________________________________________________
TH1D* Get1dTHERMHist(TString FileName, TString HistName)
{
  TFile f1(FileName);
  TH1D *ReturnHist = (TH1D*)f1.Get(HistName);

  TH1D *ReturnHistClone = (TH1D*)ReturnHist->Clone();
  ReturnHistClone->SetDirectory(0);

  return ReturnHistClone;
}

//_________________________________________________________________________________________
CfHeavy* CombineTwoCfHeavy(TString aName, CfHeavy* aCfHeavy1, CfHeavy* aCfHeavy2)
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
CfHeavy* GetTHERMHeavyCf(TString aFileName, TString aCfDescriptor, AnalysisType aAnType, int aImpactParam=8, bool aCombineConj=true, bool aUseAdamEvents=false, int aRebin=2, double aMinNorm=0.32, double aMaxNorm=0.40)
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
  TH1D* tNum1 = Get1dTHERMHist(tFileLocation, TString::Format("Num%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[aAnType]));
  TH1D* tDen1 = Get1dTHERMHist(tFileLocation, TString::Format("Den%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[aAnType]));
  CfLite* tCfLite1 = new CfLite(TString::Format("CfLite_%s", cAnalysisBaseTags[aAnType]), 
                                TString::Format("CfLite_%s", cAnalysisBaseTags[aAnType]), 
                                tNum1, tDen1, aMinNorm, aMaxNorm);
  tCfLite1->Rebin(aRebin);

  TH1D* tNum2 = Get1dTHERMHist(tFileLocation, TString::Format("Num%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[tConjAnType]));
  TH1D* tDen2 = Get1dTHERMHist(tFileLocation, TString::Format("Den%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[tConjAnType]));
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
TH1* GetTHERMCf(TString aFileName, TString aCfDescriptor, AnalysisType aAnType, int aImpactParam=8, bool aCombineConj=true, ThermEventsType aEventsType=kMe, int aRebin=2, double aMinNorm=0.32, double aMaxNorm=0.40, int aMarkerStyle=20, int aColor=1)
{
  CfHeavy* tCfHeavy;
  if(aEventsType==kMe)        tCfHeavy = GetTHERMHeavyCf(aFileName, aCfDescriptor, aAnType, aImpactParam, aCombineConj, false, aRebin, aMinNorm, aMaxNorm);
  else if(aEventsType==kAdam) tCfHeavy = GetTHERMHeavyCf(aFileName, aCfDescriptor, aAnType, aImpactParam, aCombineConj, true, aRebin, aMinNorm, aMaxNorm);
  else
  {
    CfHeavy* tCfHeavy_Me   = GetTHERMHeavyCf(aFileName, aCfDescriptor, aAnType, aImpactParam, aCombineConj, false, aRebin, aMinNorm, aMaxNorm);
    CfHeavy* tCfHeavy_Adam = GetTHERMHeavyCf(aFileName, aCfDescriptor, aAnType, aImpactParam, aCombineConj, true, aRebin, aMinNorm, aMaxNorm);

    TString tName = tCfHeavy_Me->GetHeavyCfName();
    tName += TString(cThermEventsTypeTags[aEventsType]);

    tCfHeavy = CombineTwoCfHeavy(tName, tCfHeavy_Me, tCfHeavy_Adam);
  }

  TH1* tReturnHist = tCfHeavy->GetHeavyCf();

  SetStyleAndColor(tReturnHist, aMarkerStyle, aColor);
  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1* GetTHERMCf(AnalysisType aAnType, int aImpactParam=8, bool aCombineConj=true, ThermEventsType aEventsType=kMe, int aRebin=2, double aMinNorm=0.32, double aMaxNorm=0.40, int aMarkerStyle=20, int aColor=1)
{
  TString tFileName = "CorrelationFunctions_RandomEPs_NumWeight1.root";
  TString tCfDescriptor = "Full";
  return GetTHERMCf(tFileName, tCfDescriptor, aAnType, aImpactParam, aCombineConj, aEventsType, aRebin, aMinNorm, aMaxNorm, aMarkerStyle, aColor);
}

//________________________________________________________________________________________________________________
CfHeavy* GetCombinedTHERMCfsHeavy(TString aFileName, TString aCfDescriptor, AnalysisType aAnType, CentralityType aCentType, bool aCombineConj=true, bool aUseAdamEvents=false, int aRebin=2, double aMinNorm=0.32, double aMaxNorm=0.40)
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
    TH1D* tNum1 = Get1dTHERMHist(tFileLocation, TString::Format("Num%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[aAnType]));
    TH1D* tDen1 = Get1dTHERMHist(tFileLocation, TString::Format("Den%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[aAnType]));
    CfLite* tCfLite1 = new CfLite(TString::Format("CfLite_%s_%d", cAnalysisBaseTags[aAnType], tImpactParams[iIP]), 
                                  TString::Format("CfLite_%s_%d", cAnalysisBaseTags[aAnType], tImpactParams[iIP]), 
                                  tNum1, tDen1, aMinNorm, aMaxNorm);
    tCfLite1->Rebin(aRebin);
    tCfLiteVec.push_back(tCfLite1);

    if(aCombineConj)
    {
      TH1D* tNum2 = Get1dTHERMHist(tFileLocation, TString::Format("Num%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[tConjAnType]));
      TH1D* tDen2 = Get1dTHERMHist(tFileLocation, TString::Format("Den%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[tConjAnType]));
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
TH1* GetCombinedTHERMCfs(TString aFileName, TString aCfDescriptor, AnalysisType aAnType, CentralityType aCentType, bool aCombineConj=true, ThermEventsType aEventsType=kMe, int aRebin=2, double aMinNorm=0.32, double aMaxNorm=0.40, int aMarkerStyle=20, int aColor=1)
{
  CfHeavy* tCfHeavy;
  if(aEventsType==kMe)        tCfHeavy = GetCombinedTHERMCfsHeavy(aFileName, aCfDescriptor, aAnType, aCentType, aCombineConj, false, aRebin, aMinNorm, aMaxNorm);
  else if(aEventsType==kAdam) tCfHeavy = GetCombinedTHERMCfsHeavy(aFileName, aCfDescriptor, aAnType, aCentType, aCombineConj, true, aRebin, aMinNorm, aMaxNorm);
  else
  {
    CfHeavy* tCfHeavy_Me   = GetCombinedTHERMCfsHeavy(aFileName, aCfDescriptor, aAnType, aCentType, aCombineConj, false, aRebin, aMinNorm, aMaxNorm);
    CfHeavy* tCfHeavy_Adam = GetCombinedTHERMCfsHeavy(aFileName, aCfDescriptor, aAnType, aCentType, aCombineConj, true, aRebin, aMinNorm, aMaxNorm);

    TString tName = tCfHeavy_Me->GetHeavyCfName();
    tName += TString(cThermEventsTypeTags[aEventsType]);

    tCfHeavy = CombineTwoCfHeavy(tName, tCfHeavy_Me, tCfHeavy_Adam);
  }

  TH1* tReturnHist = tCfHeavy->GetHeavyCf();
  SetStyleAndColor(tReturnHist, aMarkerStyle, aColor);
  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1* GetCombinedTHERMCfs(AnalysisType aAnType, CentralityType aCentType, bool aCombineConj=true, ThermEventsType aEventsType=kMe, int aRebin=2, double aMinNorm=0.32, double aMaxNorm=0.40, int aMarkerStyle=20, int aColor=1)
{
  TString tFileName = "CorrelationFunctions_RandomEPs_NumWeight1.root";
  TString tCfDescriptor = "Full";
  return GetCombinedTHERMCfs(tFileName, tCfDescriptor, aAnType, aCentType, aCombineConj, aEventsType, aRebin, aMinNorm, aMaxNorm, aMarkerStyle, aColor);
}


//________________________________________________________________________________________________________________
CfHeavy* GetLamKchPMCombinedTHERMCfsHeavy(TString aFileName, TString aCfDescriptor, CentralityType aCentType, bool aUseAdamEvents=false, int aRebin=2, double aMinNorm=0.32, double aMaxNorm=0.40)
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
    TH1D* tNum1 = Get1dTHERMHist(tFileLocation, TString::Format("Num%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[kLamKchP]));
    TH1D* tDen1 = Get1dTHERMHist(tFileLocation, TString::Format("Den%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[kLamKchP]));
    CfLite* tCfLite1 = new CfLite(TString::Format("CfLite_%s_%d", cAnalysisBaseTags[kLamKchP], tImpactParams[iIP]), 
                                  TString::Format("CfLite_%s_%d", cAnalysisBaseTags[kLamKchP], tImpactParams[iIP]), 
                                  tNum1, tDen1, aMinNorm, aMaxNorm);
    tCfLite1->Rebin(aRebin);
    tCfLiteVec.push_back(tCfLite1);

    TH1D* tNum2 = Get1dTHERMHist(tFileLocation, TString::Format("Num%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[kALamKchM]));
    TH1D* tDen2 = Get1dTHERMHist(tFileLocation, TString::Format("Den%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[kALamKchM]));
    CfLite* tCfLite2 = new CfLite(TString::Format("CfLite_%s_%d", cAnalysisBaseTags[kALamKchM], tImpactParams[iIP]), 
                                  TString::Format("CfLite_%s_%d", cAnalysisBaseTags[kALamKchM], tImpactParams[iIP]), 
                                  tNum2, tDen2, aMinNorm, aMaxNorm);
    tCfLite2->Rebin(aRebin);
    tCfLiteVec.push_back(tCfLite2);
    //--------------------------------
    TH1D* tNum3 = Get1dTHERMHist(tFileLocation, TString::Format("Num%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[kLamKchM]));
    TH1D* tDen3 = Get1dTHERMHist(tFileLocation, TString::Format("Den%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[kLamKchM]));
    CfLite* tCfLite3 = new CfLite(TString::Format("CfLite_%s_%d", cAnalysisBaseTags[kLamKchM], tImpactParams[iIP]), 
                                  TString::Format("CfLite_%s_%d", cAnalysisBaseTags[kLamKchM], tImpactParams[iIP]), 
                                  tNum3, tDen3, aMinNorm, aMaxNorm);
    tCfLite3->Rebin(aRebin);
    tCfLiteVec.push_back(tCfLite3);

    TH1D* tNum4 = Get1dTHERMHist(tFileLocation, TString::Format("Num%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[kALamKchP]));
    TH1D* tDen4 = Get1dTHERMHist(tFileLocation, TString::Format("Den%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[kALamKchP]));
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
TH1* GetLamKchPMCombinedTHERMCfs(TString aFileName, TString aCfDescriptor, CentralityType aCentType, ThermEventsType aEventsType=kMe, int aRebin=2, double aMinNorm=0.32, double aMaxNorm=0.40, int aMarkerStyle=20, int aColor=1)
{
  CfHeavy* tCfHeavy;
  if(aEventsType==kMe)        tCfHeavy = GetLamKchPMCombinedTHERMCfsHeavy(aFileName, aCfDescriptor, aCentType, false, aRebin, aMinNorm, aMaxNorm);
  else if(aEventsType==kAdam) tCfHeavy = GetLamKchPMCombinedTHERMCfsHeavy(aFileName, aCfDescriptor, aCentType, true, aRebin, aMinNorm, aMaxNorm);
  else
  {
    CfHeavy* tCfHeavy_Me   = GetLamKchPMCombinedTHERMCfsHeavy(aFileName, aCfDescriptor, aCentType, false, aRebin, aMinNorm, aMaxNorm);
    CfHeavy* tCfHeavy_Adam = GetLamKchPMCombinedTHERMCfsHeavy(aFileName, aCfDescriptor, aCentType, true, aRebin, aMinNorm, aMaxNorm);

    TString tName = tCfHeavy_Me->GetHeavyCfName();
    tName += TString(cThermEventsTypeTags[aEventsType]);

    tCfHeavy = CombineTwoCfHeavy(tName, tCfHeavy_Me, tCfHeavy_Adam);
  }

  TH1* tReturnHist = tCfHeavy->GetHeavyCf();
  SetStyleAndColor(tReturnHist, aMarkerStyle, aColor);
  return tReturnHist;
}



