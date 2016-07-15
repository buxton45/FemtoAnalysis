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

/*
  int tMinNormBin = 0;
  int tMaxNormBin = 0;
*/


  fHeavyCf = (TH1*)fCfLiteCollection[0]->Cf()->Clone(aReturnName);

  //-----Check to see if Sumw2 has already been classed, and if not, call it
  if(!fHeavyCf->GetSumw2N()) {fHeavyCf->Sumw2();}

  fHeavyCf->SetTitle(aReturnTitle);

/*
  tMinNormBin = fCfLiteCollection[0]->Num()->FindBin(fMinNorm);
  tMaxNormBin = fCfLiteCollection[0]->Num()->FindBin(fMaxNorm);

  tTempScale = fCfLiteCollection[0]->Num()->Integral(tMinNormBin,tMaxNormBin);
*/
  tTempScale = fCfLiteCollection[0]->GetNumScale();

  tScale += tTempScale;
  tCounter++;

  fHeavyCf->Scale(tTempScale);

  for(unsigned int i=1; i<fCfLiteCollection.size(); i++)
  {
/*
    tMinNormBin = fCfLiteCollection[i]->Num()->FindBin(fMinNorm);
    tMaxNormBin = fCfLiteCollection[i]->Num()->FindBin(fMaxNorm);
    
    tTempScale = fCfLiteCollection[i]->Num()->Integral(tMinNormBin,tMaxNormBin);
*/
    tTempScale = fCfLiteCollection[i]->GetNumScale();
  
    fHeavyCf->Add(fCfLiteCollection[i]->Cf(),tTempScale);

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
void CfHeavy::Rebin(int aRebinFactor, double aMinNorm, double aMaxNorm)
{
  fMinNorm = aMinNorm;
  fMaxNorm = aMaxNorm;

  for(unsigned int i=0; i<fCfLiteCollection.size(); i++)
  {
    fCfLiteCollection[i]->Rebin(aRebinFactor,fMinNorm,fMaxNorm);
  }

  //refresh the results
  CombineCfs();
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
    ReturnCollection->Add(fCfLiteCollection[i]->Num());
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
    ReturnCollection->Add(fCfLiteCollection[i]->Den());
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
    ReturnCollection->Add(fCfLiteCollection[i]->Cf());
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


