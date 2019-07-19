///////////////////////////////////////////////////////////////////////////
// CfHeavy:                                                         //
///////////////////////////////////////////////////////////////////////////


#include "CfHeavy.h"

#ifdef __ROOT__
ClassImp(CfHeavy)
#endif



//________________________________________________________________________________________________________________





//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________






//________________________________________________________________________________________________________________
CfHeavy::CfHeavy(TString aHeavyCfName, TString aHeavyCfTitle, vector<CfLite*> &aCfLiteCollection, double aMinNorm, double aMaxNorm) : 
  fHeavyCfName(aHeavyCfName),
  fHeavyCfTitle(aHeavyCfTitle),
  fCfLiteCollection(aCfLiteCollection),
  fCollectionSize(aCfLiteCollection.size()),
  fHeavyCf(0),
  fHeavyCfwErrorsByHand(0),
  fMinNorm(aMinNorm),
  fMaxNorm(aMaxNorm)

{

  CombineCfs(fHeavyCfName,fHeavyCfTitle);

  fHeavyCf->SetDirectory(0);

}



//________________________________________________________________________________________________________________
CfHeavy::~CfHeavy()
{

}


//________________________________________________________________________________________________________________
CfHeavy::CfHeavy(const CfHeavy& aHeavy) :

  fHeavyCfName(aHeavy.fHeavyCfName),
  fHeavyCfTitle(aHeavy.fHeavyCfTitle),
  fCollectionSize(aHeavy.fCollectionSize),
  fMinNorm(aHeavy.fMinNorm),
  fMaxNorm(aHeavy.fMaxNorm)
{
  //copy constructor
  for(int i=0; i< aHeavy.fCollectionSize; i++)
  {
    CfLite* aCopyLite = new CfLite( *(aHeavy.fCfLiteCollection[i]) );
    fCfLiteCollection.push_back(aCopyLite);
  }

  if(aHeavy.fHeavyCf) fHeavyCf = (TH1*)aHeavy.fHeavyCf->Clone();
  else fHeavyCf = 0;

  if(aHeavy.fHeavyCfwErrorsByHand) fHeavyCfwErrorsByHand = (TH1*)aHeavy.fHeavyCfwErrorsByHand->Clone();
  else fHeavyCfwErrorsByHand = 0;

}

//________________________________________________________________________________________________________________
CfHeavy& CfHeavy::operator=(const CfHeavy& aHeavy)
{
  //assignment operator
  if(this == &aHeavy) return *this;

  fHeavyCfName = aHeavy.fHeavyCfName;
  fHeavyCfTitle = aHeavy.fHeavyCfTitle;
  fCollectionSize = aHeavy.fCollectionSize;
  fMinNorm = aHeavy.fMinNorm;
  fMaxNorm = aHeavy.fMaxNorm;

  for(int i=0; i< aHeavy.fCollectionSize; i++)
  {
    CfLite* aCopyLite = new CfLite( *(aHeavy.fCfLiteCollection[i]) );
    fCfLiteCollection.push_back(aCopyLite);
  }

  if(aHeavy.fHeavyCf) fHeavyCf = (TH1*)aHeavy.fHeavyCf->Clone();
  else fHeavyCf = 0;

  if(aHeavy.fHeavyCfwErrorsByHand) fHeavyCfwErrorsByHand = (TH1*)aHeavy.fHeavyCfwErrorsByHand->Clone();
  else fHeavyCfwErrorsByHand = 0;

  return *this;
}


//________________________________________________________________________________________________________________
void CfHeavy::CombineCfs()
{
  CombineCfs(fHeavyCfName,fHeavyCfTitle);
}



//________________________________________________________________________________________________________________
void CfHeavy::CombineCfs(TString aReturnName, TString aReturnTitle)
{
  double tScale = 0.;
  int tCounter = 0;
  double tTempScale = 0.;

  fHeavyCf = (TH1*)fCfLiteCollection[0]->Cf()->Clone(aReturnName);

  //-----Check to see if Sumw2 has already been classed, and if not, call it
  if(!fHeavyCf->GetSumw2N()) {fHeavyCf->Sumw2();}

  fHeavyCf->SetTitle(aReturnTitle);
  tTempScale = fCfLiteCollection[0]->GetNumScale();

  tScale += tTempScale;
  tCounter++;

  fHeavyCf->Scale(tTempScale);

  for(unsigned int i=1; i<fCfLiteCollection.size(); i++)
  {
    assert(fCfLiteCollection[0]->Num()->GetBinWidth(1)==fCfLiteCollection[i]->Num()->GetBinWidth(1));
    assert(fCfLiteCollection[0]->Num()->GetNbinsX()==fCfLiteCollection[i]->Num()->GetNbinsX());

    tTempScale = fCfLiteCollection[i]->GetNumScale();
  
    fHeavyCf->Add((TH1*)fCfLiteCollection[i]->Cf()->Clone(),tTempScale);

    tScale += tTempScale;
    tCounter++;
  }

  fHeavyCf->Scale(1./tScale);

  if(!fHeavyCf->GetSumw2N()) {fHeavyCf->Sumw2();}

}

//________________________________________________________________________________________________________________
void CfHeavy::Rebin(int aRebinFactor)
{
  for(unsigned int i=0; i<fCfLiteCollection.size(); i++)
  {
    fCfLiteCollection[i]->Rebin(aRebinFactor,fMinNorm,fMaxNorm);
  }

  //refresh the results
  CombineCfs();
}

//________________________________________________________________________________________________________________
void CfHeavy::Rebin(int aNGroups, vector<double> &aGroups)
{
  for(unsigned int i=0; i<fCfLiteCollection.size(); i++)
  {
    fCfLiteCollection[i]->Rebin(aNGroups,aGroups);
  }

  //refresh the results
  CombineCfs();
}


//________________________________________________________________________________________________________________
void CfHeavy::Rebin(int aRebinFactor, double aMinNorm, double aMaxNorm)
{
  fMinNorm = aMinNorm;
  fMaxNorm = aMaxNorm;

  for(unsigned int i=0; i<fCfLiteCollection.size(); i++)
  {
    fCfLiteCollection[i]->Rebin(aRebinFactor,fMinNorm,fMaxNorm);
  }

  //refresh the results
  CombineCfs(TString::Format("%s_CustomRebin", fHeavyCf->GetName()), TString::Format("%s_CustomRebin", fHeavyCf->GetTitle()));
}


//________________________________________________________________________________________________________________
void CfHeavy::AddCfLite(CfLite* aCfLite)
{
  fCfLiteCollection.push_back(aCfLite);
  //refresh the results
  CombineCfs();
}


//________________________________________________________________________________________________________________
TObjArray* CfHeavy::GetNumCollection()
{
  TObjArray* ReturnCollection = new TObjArray();
  ReturnCollection->SetOwner(true);

  for(unsigned int i=0; i<fCfLiteCollection.size(); i++)
  {
    ReturnCollection->Add(fCfLiteCollection[i]->Num()->Clone());
  }

  return ReturnCollection;
}

//________________________________________________________________________________________________________________
TObjArray* CfHeavy::GetDenCollection()
{
  TObjArray* ReturnCollection = new TObjArray();
  ReturnCollection->SetOwner(true);

  for(unsigned int i=0; i<fCfLiteCollection.size(); i++)
  {
    ReturnCollection->Add(fCfLiteCollection[i]->Den()->Clone());
  }

  return ReturnCollection;
}


//________________________________________________________________________________________________________________
TObjArray* CfHeavy::GetCfCollection()
{
  TObjArray* ReturnCollection = new TObjArray();
  ReturnCollection->SetOwner(true);

  for(unsigned int i=0; i<fCfLiteCollection.size(); i++)
  {
    ReturnCollection->Add(fCfLiteCollection[i]->Cf()->Clone());
  }

  return ReturnCollection;
}


//________________________________________________________________________________________________________________
void CfHeavy::SaveAllCollectionsAndCf(TString aPreName, TString aPostName, TFile *aFile)
{
  assert(aFile->IsOpen());


  TObjArray* tNumCollection = GetNumCollection();
  TObjArray* tDenCollection = GetDenCollection();
  TObjArray* tCfCollection = GetCfCollection();

  TString tNumCollName = "f";
    tNumCollName += aPreName;
    tNumCollName += "NumCollection_";
    tNumCollName += aPostName;

  TString tDenCollName = "f";
    tDenCollName += aPreName;
    tDenCollName += "DenCollection_";
    tDenCollName += aPostName;

  TString tCfCollName = "f";
    tCfCollName += aPreName;
    tCfCollName += "CfCollection_";
    tCfCollName += aPostName;

  TString tCfName = "f";
    tCfName += aPreName;
    tCfName += "Cf_";
    tCfName += aPostName;
    tCfName += "_Tot";
  

  tNumCollection->Write(tNumCollName,TObject::kSingleKey);
  tDenCollection->Write(tDenCollName,TObject::kSingleKey);
  tCfCollection->Write(tCfCollName,TObject::kSingleKey);
  
  fHeavyCf->Write(tCfName);

}


//________________________________________________________________________________________________________________
TH1* CfHeavy::GetSimplyAddedNumDen(TString aReturnName, bool aGetNum)
{
  TH1* tReturnHist;
  if(aGetNum) tReturnHist = (TH1*)fCfLiteCollection[0]->Num()->Clone(aReturnName);
  else tReturnHist = (TH1*)fCfLiteCollection[0]->Den()->Clone(aReturnName);
  tReturnHist->SetTitle(aReturnName);
  //-----Check to see if Sumw2 has already been classed, and if not, call it
  if(!tReturnHist->GetSumw2N()) {tReturnHist->Sumw2();}

  for(unsigned int i=1; i<fCfLiteCollection.size(); i++)
  {
    if(aGetNum) tReturnHist->Add(fCfLiteCollection[i]->Num());
    else tReturnHist->Add(fCfLiteCollection[i]->Den());
  }

  if(!tReturnHist->GetSumw2N()) {tReturnHist->Sumw2();}
  return tReturnHist;
}

//________________________________________________________________________________________________________________
void CfHeavy::BuildHeavyCfwErrorsByHand()
{
  fHeavyCfwErrorsByHand = (TH1*)fHeavyCf->Clone(TString::Format("%swErrorsByHand", fHeavyCfName.Data()));
  fHeavyCfwErrorsByHand->SetTitle(TString::Format("%swErrorsByHand", fHeavyCfTitle.Data()));
  fHeavyCfwErrorsByHand->Sumw2(false);  //This doesn't matter, because after SetBinError is called, this is set to true

  for(unsigned int i=0; i<fCfLiteCollection.size(); i++) fCfLiteCollection[i]->BuildCfwErrorsByHand();
  for(unsigned int i=1; i<fCfLiteCollection.size(); i++) assert(fCfLiteCollection[i-1]->CfwErrorsByHand()->GetNbinsX() == fCfLiteCollection[i]->CfwErrorsByHand()->GetNbinsX());

  int tNbins = fCfLiteCollection[0]->CfwErrorsByHand()->GetNbinsX();
  assert(tNbins == fHeavyCfwErrorsByHand->GetNbinsX());

  double tSumOfWeights = 0.;
  for(unsigned int iCf=0; iCf<fCfLiteCollection.size(); iCf++) tSumOfWeights += fCfLiteCollection[iCf]->GetNumScale();

  for(int iBin=1; iBin<=tNbins; iBin++)
  {
    double tWeight=0., tVarSq=0., tCfVal=0., tCfErr=0., tTerm1=0.;
    for(unsigned int iCf=0; iCf<fCfLiteCollection.size(); iCf++)
    {
      tWeight = fCfLiteCollection[iCf]->GetNumScale();
      tCfVal = fCfLiteCollection[iCf]->CfwErrorsByHand()->GetBinContent(iBin);
      tCfErr = fCfLiteCollection[iCf]->CfwErrorsByHand()->GetBinError(iBin);

      tTerm1 = (1./tSumOfWeights)*tWeight*tCfVal*tCfErr;
      tVarSq += tTerm1*tTerm1;
    }
    assert(abs(sqrt(tVarSq) - fHeavyCf->GetBinError(iBin)) < std::numeric_limits< double >::min());
    fHeavyCfwErrorsByHand->SetBinError(iBin, sqrt(tVarSq));
  }

  for(int i=1; i<=tNbins; i++)
  {
    assert(abs(fHeavyCfwErrorsByHand->GetBinContent(i) - fHeavyCf->GetBinContent(i)) < std::numeric_limits< double >::min());
    assert(abs(fHeavyCfwErrorsByHand->GetBinError(i) - fHeavyCf->GetBinError(i)) < std::numeric_limits< double >::min());
  }
}


//________________________________________________________________________________________________________________
void CfHeavy::DivideCfByThermBgd(TH1* aThermBgd)
{
  assert(fHeavyCf->GetBinWidth(1)==aThermBgd->GetBinWidth(1));
  assert(aThermBgd->GetNbinsX() >= fHeavyCf->GetNbinsX());

  if(aThermBgd->GetNbinsX() > fHeavyCf->GetNbinsX()) aThermBgd->SetBins(fHeavyCf->GetNbinsX(), 
                                                                        fHeavyCf->GetXaxis()->GetBinLowEdge(1), 
                                                                        fHeavyCf->GetXaxis()->GetBinUpEdge(fHeavyCf->GetNbinsX()));
  assert(aThermBgd->GetNbinsX() == fHeavyCf->GetNbinsX());
  fHeavyCf->Divide(aThermBgd);
}



//________________________________________________________________________________________________________________
double CfHeavy::GetTotalNumScale()
{
  double tTotalScale = 0.;
  for(unsigned int i=0; i<fCfLiteCollection.size(); i++) tTotalScale += fCfLiteCollection[i]->GetNumScale();
  return tTotalScale;
}



