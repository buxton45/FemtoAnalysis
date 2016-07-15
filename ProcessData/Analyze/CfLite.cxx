///////////////////////////////////////////////////////////////////////////
// CfLite:                                                               //
///////////////////////////////////////////////////////////////////////////


#include "CfLite.h"

#ifdef __ROOT__
ClassImp(CfLite)
#endif



//________________________________________________________________________________________________________________





//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________






//________________________________________________________________________________________________________________
CfLite::CfLite(TString aName, TString aTitle, TH1* aNum, TH1* aDen, double aMinNorm, double aMaxNorm) :
  fCfName(aName),
  fCfTitle(aTitle),
  fNum(aNum),
  fDen(aDen),
  fCf(0),
  fMinNorm(aMinNorm),
  fMaxNorm(aMaxNorm),
  fNumScale(0),
  fDenScale(0)

{

  BuildCf(fCfName,fCfTitle,fMinNorm,fMaxNorm);

  fNum->SetDirectory(0);
  fDen->SetDirectory(0);
  fCf->SetDirectory(0);

  if(!fNum->GetSumw2N()) {fNum->Sumw2();}
  if(!fDen->GetSumw2N()) {fDen->Sumw2();}


}



//________________________________________________________________________________________________________________
CfLite::~CfLite()
{

}


//________________________________________________________________________________________________________________
CfLite::CfLite(const CfLite& aLite) : 
  fCfName(aLite.fCfName),
  fCfTitle(aLite.fCfTitle),
  fMinNorm(aLite.fMinNorm),
  fMaxNorm(aLite.fMaxNorm),
  fNumScale(aLite.fNumScale),
  fDenScale(aLite.fDenScale)
{
  //copy constructor
  if(aLite.fNum) fNum = (TH1*)aLite.fNum->Clone(); 
  else fNum = 0;

  if(aLite.fDen) fDen = (TH1*)aLite.fDen->Clone(); 
  else fDen = 0;

  if(aLite.fCf) fCf = (TH1*)aLite.fCf->Clone(); 
  else fCf = 0;
}

//________________________________________________________________________________________________________________
CfLite& CfLite::operator=(const CfLite& aLite)
{
  //assignment operator
  if(this == &aLite) return *this;

  fCfName = aLite.fCfName;
  fCfTitle = aLite.fCfTitle;
  fMinNorm = aLite.fMinNorm;
  fMaxNorm = aLite.fMaxNorm;
  fNumScale = aLite.fNumScale;
  fDenScale = aLite.fDenScale;

  if(aLite.fNum) fNum = (TH1*)aLite.fNum->Clone(); 
  else fNum = 0;

  if(aLite.fDen) fDen = (TH1*)aLite.fDen->Clone(); 
  else fDen = 0;

  if(aLite.fCf) fCf = (TH1*)aLite.fCf->Clone(); 
  else fCf = 0;

  return *this;
}



//________________________________________________________________________________________________________________
void CfLite::BuildCf(double aMinNorm, double aMaxNorm)
{
  BuildCf(fCfName,fCfTitle,fMinNorm,fMaxNorm);
}





//________________________________________________________________________________________________________________
void CfLite::BuildCf(TString aName, TString aTitle, double aMinNorm, double aMaxNorm)
{
  int tMinNormBin = fNum->FindBin(aMinNorm);
  int tMaxNormBin = fNum->FindBin(aMaxNorm);
  fNumScale = fNum->Integral(tMinNormBin,tMaxNormBin);

  tMinNormBin = fDen->FindBin(aMinNorm);
  tMaxNormBin = fDen->FindBin(aMaxNorm);
  fDenScale = fDen->Integral(tMinNormBin,tMaxNormBin);

  fCf = (TH1*)fNum->Clone(aName);

  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!fCf->GetSumw2N())
  {
    cout << "Sumw2 NOT already called on " << aName << ", so calling it now" << endl;
    fCf->Sumw2();
  }

  fCf->Divide(fDen);
  fCf->Scale(fDenScale/fNumScale);
  fCf->SetTitle(aTitle);

  if(!fCf->GetSumw2N()) {fCf->Sumw2();}

}





//________________________________________________________________________________________________________________
void CfLite::Rebin(int aRebinFactor)
{
  Rebin(aRebinFactor,fMinNorm,fMaxNorm);
}
//________________________________________________________________________________________________________________
void CfLite::Rebin(int aRebinFactor, double aMinNorm, double aMaxNorm)
{
  fNum->Rebin(aRebinFactor);
  fDen->Rebin(aRebinFactor);

  BuildCf(aMinNorm,aMaxNorm);
}









