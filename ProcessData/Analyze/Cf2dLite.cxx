///////////////////////////////////////////////////////////////////////////
// Cf2dLite:                                                             //
///////////////////////////////////////////////////////////////////////////


#include "Cf2dLite.h"

#ifdef __ROOT__
ClassImp(Cf2dLite)
#endif



//________________________________________________________________________________________________________________





//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________






//________________________________________________________________________________________________________________
Cf2dLite::Cf2dLite(TString aDaughterCfsBaseName, TH2* aMotherNum2d, TH2* aMotherDen2d, AxisType aProjectionAxis, vector<vector<int> > &aProjectionBins, double aMinNorm, double aMaxNorm) :
  fMotherNum2d(aMotherNum2d),
  fMotherDen2d(aMotherDen2d),
  fNbinsX(0),
  fNbinsY(0),
  fXName(0),
  fYName(0),
  fProjectionAxis(aProjectionAxis),
  fProjectionBins(aProjectionBins),
  fMinNorm(aMinNorm),
  fMaxNorm(aMaxNorm),
  fNDaughterCfs(0),
  fDaughterCfsBaseName(aDaughterCfsBaseName),
  fDaughterCfs(0)

{

  assert(fMotherNum2d->GetNbinsX() == fMotherDen2d->GetNbinsX());
  assert(fMotherNum2d->GetNbinsY() == fMotherDen2d->GetNbinsY());

  fNbinsX = fMotherNum2d->GetNbinsX();
  fNbinsY = fMotherNum2d->GetNbinsY();

  fNDaughterCfs = fProjectionBins.size();

  DoProjection();

}



//________________________________________________________________________________________________________________
Cf2dLite::~Cf2dLite()
{

}


//________________________________________________________________________________________________________________
void Cf2dLite::DoProjection(int aRebinFactor)
{
  fDaughterCfs.clear();

  int tProjBinLow = 0;
  int tProjBinHigh = 0;
  
  for(int i=0; i<fNDaughterCfs; i++)
  {
    tProjBinLow = fProjectionBins[i][0];
    tProjBinHigh = fProjectionBins[i][1];

    TString tNumName = "Num";
    TString tDenName = "Den";
    TString tCfName = fDaughterCfsBaseName;
    tCfName+=i;

    tNumName+=fDaughterCfsBaseName;
    tNumName+=i;

    tDenName+=fDaughterCfsBaseName;
    tDenName+=i;

    TH1* tempNum;
    TH1* tempDen;

    if(fProjectionAxis == kXaxis)
    {
      tempNum = fMotherNum2d->ProjectionX(tNumName,tProjBinLow,tProjBinHigh);
      tempDen = fMotherDen2d->ProjectionX(tDenName,tProjBinLow,tProjBinHigh);
    }
    else if(fProjectionAxis == kYaxis)
    {
      tempNum = fMotherNum2d->ProjectionY(tNumName,tProjBinLow,tProjBinHigh);
      tempDen = fMotherDen2d->ProjectionY(tDenName,tProjBinLow,tProjBinHigh);
    }
    
    CfLite *tempCfLite = new CfLite(tCfName,tCfName,tempNum,tempDen,fMinNorm,fMaxNorm);
    if(aRebinFactor != 1) {tempCfLite->Rebin(aRebinFactor,fMinNorm,fMaxNorm);}
    fDaughterCfs.push_back(tempCfLite);

  }

}


//________________________________________________________________________________________________________________
void Cf2dLite::Rebin(int aRebinFactor)
{
  DoProjection(aRebinFactor);
}



//________________________________________________________________________________________________________________
CfLite* Cf2dLite::GetDaughterCf(int aDaughterCf)
{
  assert(aDaughterCf <= fNDaughterCfs);
  return fDaughterCfs[aDaughterCf];
}




