/* ThermChargedResidual.cxx */

#include "ThermChargedResidual.h"


#ifdef __ROOT__
ClassImp(ThermChargedResidual)
#endif





//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________
ThermChargedResidual::ThermChargedResidual(AnalysisType aResType) :
  fResidualType(aResType),
  fPartType1(kPDGNull),
  fPartType2(kPDGNull),

  f3dCoulombOnlyInterpWfs(nullptr),

  fInterpKStarMagMin(-1.),
  fInterpKStarMagMax(-1.),

  fInterpRStarMagMin(-1.),
  fInterpRStarMagMax(-1.),

  fInterpThetaMin(-1.),
  fInterpThetaMax(-1.)
{
  cout << "Building ThermChargedResidual: " << cAnalysisBaseTags[fResidualType] << endl;
  SetPartTypes();
  LoadCoulombOnlyInterpWfs();
}

//________________________________________________________________________________________________________________
ThermChargedResidual::ThermChargedResidual(const ThermChargedResidual& aRes) :
  fResidualType(aRes.fResidualType),
  fPartType1(aRes.fPartType1),
  fPartType2(aRes.fPartType2),

  fInterpKStarMagMin(aRes.fInterpKStarMagMin),
  fInterpKStarMagMax(aRes.fInterpKStarMagMax),

  fInterpRStarMagMin(aRes.fInterpRStarMagMin),
  fInterpRStarMagMax(aRes.fInterpRStarMagMax),

  fInterpThetaMin(aRes.fInterpThetaMin),
  fInterpThetaMax(aRes.fInterpThetaMax)
{
  if(aRes.f3dCoulombOnlyInterpWfs) f3dCoulombOnlyInterpWfs = new TH3D(*aRes.f3dCoulombOnlyInterpWfs);
  else f3dCoulombOnlyInterpWfs = nullptr;
}

//________________________________________________________________________________________________________________
ThermChargedResidual& ThermChargedResidual::operator=(const ThermChargedResidual& aRes)
{
  if(this == &aRes) return *this;

  fResidualType = aRes.fResidualType;
  fPartType1 = aRes.fPartType1;
  fPartType2 = aRes.fPartType2;

  fInterpKStarMagMin = aRes.fInterpKStarMagMin;
  fInterpKStarMagMax = aRes.fInterpKStarMagMax;

  fInterpRStarMagMin = aRes.fInterpRStarMagMin;
  fInterpRStarMagMax = aRes.fInterpRStarMagMax;

  fInterpThetaMin = aRes.fInterpThetaMin;
  fInterpThetaMax = aRes.fInterpThetaMax;

  if(aRes.f3dCoulombOnlyInterpWfs) f3dCoulombOnlyInterpWfs = new TH3D(*aRes.f3dCoulombOnlyInterpWfs);
  else f3dCoulombOnlyInterpWfs = nullptr;

  return *this;
}

//________________________________________________________________________________________________________________
ThermChargedResidual::~ThermChargedResidual()
{
/*no-op*/
}


//________________________________________________________________________________________________________________
void ThermChargedResidual::SetPartTypes()
{
  switch(fResidualType) {
  //LamKchP-------------------------------
  case kResXiCKchP:
    fPartType1 = kPDGXiC;
    fPartType2 = kPDGKchP;
    break;

  case kResSigStPKchP:
    fPartType1 = kPDGSigStP;
    fPartType2 = kPDGKchP;
    break;

  case kResSigStMKchP:
    fPartType1 = kPDGSigStM;
    fPartType2 = kPDGKchP;
    break;

  //ALamKchM-------------------------------
  case kResAXiCKchM:
    fPartType1 = kPDGAXiC;
    fPartType2 = kPDGKchM;
    break;

  case kResASigStMKchM:
    fPartType1 = kPDGASigStM;
    fPartType2 = kPDGKchM;
    break;

  case kResASigStPKchM:
    fPartType1 = kPDGASigStP;
    fPartType2 = kPDGKchM;
    break;

  //LamKchM-------------------------------
  case kResXiCKchM:
    fPartType1 = kPDGXiC;
    fPartType2 = kPDGKchM;
    break;

  case kResSigStPKchM:
    fPartType1 = kPDGSigStP;
    fPartType2 = kPDGKchM;
    break;


  case kResSigStMKchM:
    fPartType1 = kPDGSigStM;
    fPartType2 = kPDGKchM;
    break;



  //ALamKchP-------------------------------
  case kResAXiCKchP:
    fPartType1 = kPDGAXiC;
    fPartType2 = kPDGKchP;
    break;

  case kResASigStMKchP:
    fPartType1 = kPDGASigStM;
    fPartType2 = kPDGKchP;
    break;

  case kResASigStPKchP:
    fPartType1 = kPDGASigStP;
    fPartType2 = kPDGKchP;
    break;


  default:
    cout << "ERROR: ThermChargedResidual::SetPartTypes:  fResidualType = " << fResidualType << " is not appropriate" << endl << endl;
    assert(0);
  }
}

//________________________________________________________________________________________________________________
void ThermChargedResidual::LoadCoulombOnlyInterpWfs(TString aFileDirectory)
{
  TString aFileName = aFileDirectory + TString::Format("3dCoulombOnlyInterpWfs_%s.root", cAnalysisBaseTags[fResidualType]);
  TFile aFile(aFileName);
  TH3D* t3dCoulombOnlyInterpWfs = (TH3D*)aFile.Get(TString::Format("t3dCoulombOnlyInterpWfs_%s", cAnalysisBaseTags[fResidualType]));
  assert(t3dCoulombOnlyInterpWfs);
  f3dCoulombOnlyInterpWfs = (TH3D*)t3dCoulombOnlyInterpWfs->Clone();
  f3dCoulombOnlyInterpWfs->SetDirectory(0);

  //NOTE: Min values should technically also use GetBinCenter, but I keep GetBinLowEdge because
  //if value less than min, I do not want WfSq=1 and I want TH3::Interpolate to print error

  fInterpKStarMagMin = f3dCoulombOnlyInterpWfs->GetXaxis()->GetBinLowEdge(1);
  fInterpKStarMagMax = f3dCoulombOnlyInterpWfs->GetXaxis()->GetBinCenter(f3dCoulombOnlyInterpWfs->GetNbinsX());

  fInterpRStarMagMin = f3dCoulombOnlyInterpWfs->GetYaxis()->GetBinLowEdge(1);
  fInterpRStarMagMax = f3dCoulombOnlyInterpWfs->GetYaxis()->GetBinCenter(f3dCoulombOnlyInterpWfs->GetNbinsY());

  fInterpThetaMin = f3dCoulombOnlyInterpWfs->GetZaxis()->GetBinLowEdge(1);
  fInterpThetaMax = f3dCoulombOnlyInterpWfs->GetZaxis()->GetBinUpEdge(f3dCoulombOnlyInterpWfs->GetNbinsZ());
}

//________________________________________________________________________________________________________________
bool ThermChargedResidual::CanInterp(double aKStarMag, double aRStarMag, double aTheta)
{
  if(aKStarMag < fInterpKStarMagMin || aKStarMag > fInterpKStarMagMax) return false;
  if(aRStarMag < fInterpRStarMagMin || aRStarMag > fInterpRStarMagMax) return false;
  if(aTheta < fInterpThetaMin || aTheta > fInterpThetaMax) return false;
  return true;
}

//________________________________________________________________________________________________________________
bool ThermChargedResidual::CanInterp(TVector3* aKStar3Vec, TVector3* aRStar3Vec)
{
  double tTheta = aKStar3Vec->Angle(*aRStar3Vec);
  double tKStarMag = aKStar3Vec->Mag();
  double tRStarMag = aRStar3Vec->Mag();

  return CanInterp(tKStarMag, tRStarMag, tTheta);
}




//________________________________________________________________________________________________________________
double ThermChargedResidual::InterpolateWfSquared(double aKStarMag, double aRStarMag, double aTheta)
{
  if(CanInterp(aKStarMag, aRStarMag, aTheta)) return f3dCoulombOnlyInterpWfs->Interpolate(aKStarMag, aRStarMag, aTheta);
  else return 1.;
}

//________________________________________________________________________________________________________________
double ThermChargedResidual::InterpolateWfSquared(TVector3* aKStar3Vec, TVector3* aRStar3Vec)
{
  double tTheta = aKStar3Vec->Angle(*aRStar3Vec);
  double tKStarMag = aKStar3Vec->Mag();
  double tRStarMag = aRStar3Vec->Mag();

  return InterpolateWfSquared(tKStarMag, tRStarMag, tTheta);
}


