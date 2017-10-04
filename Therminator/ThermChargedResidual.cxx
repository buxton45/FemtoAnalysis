/* ThermChargedResidual.cxx */

#include "ThermChargedResidual.h"


#ifdef __ROOT__
ClassImp(ThermChargedResidual)
#endif





//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________
ThermChargedResidual::ThermChargedResidual(AnalysisType aAnType) :
  fResidualType(aAnType),
  fPartType1(kPDGNull),
  fPartType2(kPDGNull),
  fBohrRadius(0.),
  fTurnOffCoulomb(false),

  fInterpHistFile(0), fInterpHistFileLednickyHFunction(0),

  fLednickyHFunctionHist(0),

  fGTildeRealHist(0),
  fGTildeImagHist(0),

  fHyperGeo1F1RealHist(0),
  fHyperGeo1F1ImagHist(0),

  fMinInterpKStar(0), fMinInterpRStar(0), fMinInterpTheta(0),
  fMaxInterpKStar(0), fMaxInterpRStar(0), fMaxInterpTheta(0)
{
  SetPartTypes();
  fBohrRadius = GetBohrRadius(fResidualType);
  LoadInterpHistFile();
}

//________________________________________________________________________________________________________________
ThermChargedResidual::ThermChargedResidual(const ThermChargedResidual& aRes) :
  fResidualType(aRes.fResidualType),
  fPartType1(aRes.fPartType1),
  fPartType2(aRes.fPartType2),
  fBohrRadius(aRes.fBohrRadius),
  fTurnOffCoulomb(aRes.fTurnOffCoulomb),

  fInterpHistFile(nullptr),
  fInterpHistFileLednickyHFunction(nullptr),

  fMinInterpKStar(aRes.fMinInterpKStar), fMinInterpRStar(aRes.fMinInterpRStar), fMinInterpTheta(aRes.fMinInterpTheta),
  fMaxInterpKStar(aRes.fMaxInterpKStar), fMaxInterpRStar(aRes.fMaxInterpRStar), fMaxInterpTheta(aRes.fMaxInterpTheta)
{

  if(aRes.fLednickyHFunctionHist) fLednickyHFunctionHist = new TH1D(*aRes.fLednickyHFunctionHist);
  else fLednickyHFunctionHist = nullptr;

  if(aRes.fGTildeRealHist) fGTildeRealHist = new TH2D(*aRes.fGTildeRealHist);
  else fGTildeRealHist = nullptr;

  if(aRes.fGTildeImagHist) fGTildeImagHist = new TH2D(*aRes.fGTildeImagHist);
  else fGTildeImagHist = nullptr;

  if(aRes.fHyperGeo1F1RealHist) fHyperGeo1F1RealHist = new TH3D(*aRes.fHyperGeo1F1RealHist);
  else fHyperGeo1F1RealHist = nullptr;

  if(aRes.fHyperGeo1F1ImagHist) fHyperGeo1F1ImagHist = new TH3D(*aRes.fHyperGeo1F1ImagHist);
  else fHyperGeo1F1ImagHist = nullptr;
}

//________________________________________________________________________________________________________________
ThermChargedResidual& ThermChargedResidual::operator=(const ThermChargedResidual& aRes)
{
  if(this == &aRes) return *this;

  fResidualType = aRes.fResidualType;
  fPartType1 = aRes.fPartType1;
  fPartType2 = aRes.fPartType2;
  fBohrRadius = aRes.fBohrRadius;
  fTurnOffCoulomb = aRes.fTurnOffCoulomb;

  fInterpHistFile = nullptr;
  fInterpHistFileLednickyHFunction = nullptr;

  fMinInterpKStar = aRes.fMinInterpKStar; fMinInterpRStar = aRes.fMinInterpRStar; fMinInterpTheta = aRes.fMinInterpTheta;
  fMaxInterpKStar = aRes.fMaxInterpKStar; fMaxInterpRStar = aRes.fMaxInterpRStar; fMaxInterpTheta = aRes.fMaxInterpTheta;


  if(aRes.fLednickyHFunctionHist) fLednickyHFunctionHist = new TH1D(*aRes.fLednickyHFunctionHist);
  else fLednickyHFunctionHist = nullptr;

  if(aRes.fGTildeRealHist) fGTildeRealHist = new TH2D(*aRes.fGTildeRealHist);
  else fGTildeRealHist = nullptr;

  if(aRes.fGTildeImagHist) fGTildeImagHist = new TH2D(*aRes.fGTildeImagHist);
  else fGTildeImagHist = nullptr;

  if(aRes.fHyperGeo1F1RealHist) fHyperGeo1F1RealHist = new TH3D(*aRes.fHyperGeo1F1RealHist);
  else fHyperGeo1F1RealHist = nullptr;

  if(aRes.fHyperGeo1F1ImagHist) fHyperGeo1F1ImagHist = new TH3D(*aRes.fHyperGeo1F1ImagHist);
  else fHyperGeo1F1ImagHist = nullptr;

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
double ThermChargedResidual::GetBohrRadius(AnalysisType aAnalysisType)
{
  double tReturnBohrRadius = -1.;

  switch(aAnalysisType) {
  case kXiKchP:
  case kAXiKchM:
  case kResXiCKchP:
  case kResAXiCKchM:
    tReturnBohrRadius = -gBohrRadiusXiK;  //attractive
    break;

  case kXiKchM:
  case kAXiKchP:
  case kResXiCKchM:
  case kResAXiCKchP:
    tReturnBohrRadius = gBohrRadiusXiK;  //repulsive
    break;

  //-----------------------------------------------------

  case kResOmegaKchP:
  case kResAOmegaKchM:
    tReturnBohrRadius = -gBohrRadiusOmegaK;  //attractive
    break;

  case kResOmegaKchM:
  case kResAOmegaKchP:
    tReturnBohrRadius = gBohrRadiusOmegaK;  //repulsive
    break;

  //-----------------------------------------------------

  case kResSigStPKchM:
  case kResASigStMKchP:
    tReturnBohrRadius = -gBohrRadiusSigStPK; //attractive
    break;

  case kResSigStPKchP:
  case kResASigStMKchM:
    tReturnBohrRadius = gBohrRadiusSigStPK; //repulsive
    break;

  case kResSigStMKchP:
  case kResASigStPKchM:
    tReturnBohrRadius = -gBohrRadiusSigStMK;  //attractive
    break;

  case kResSigStMKchM:
  case kResASigStPKchP:
    tReturnBohrRadius = gBohrRadiusSigStMK;  //repulsive
    break;

  //-----------------------------------------------------

  default:
    tReturnBohrRadius = 1000000000;
//    cout << "ERROR: WaveFunction::SetCurrentBohrRadius: Invalid aAnalysisType = " << aAnalysisType << endl;
//    assert(0);
  }
  return tReturnBohrRadius;
}




//________________________________________________________________________________________________________________
void ThermChargedResidual::LoadLednickyHFunctionFile(TString aFileBaseName)
{
  TString tFileName = aFileBaseName+TString::Format("_%s.root", cAnalysisBaseTags[fResidualType]);
  fInterpHistFileLednickyHFunction = TFile::Open(tFileName);

  fLednickyHFunctionHist = (TH1D*)fInterpHistFileLednickyHFunction->Get("LednickyHFunction");
    fLednickyHFunctionHist->SetDirectory(0);

  fInterpHistFileLednickyHFunction->Close();

  if(fMinInterpKStar==0 && fMaxInterpKStar==0)
  {
    fMinInterpKStar = fLednickyHFunctionHist->GetXaxis()->GetBinCenter(1);
    fMaxInterpKStar = fLednickyHFunctionHist->GetXaxis()->GetBinCenter(fLednickyHFunctionHist->GetNbinsX());
  }
  else
  {
    assert(fMinInterpKStar == fLednickyHFunctionHist->GetXaxis()->GetBinCenter(1));
    assert(fMaxInterpKStar == fLednickyHFunctionHist->GetXaxis()->GetBinCenter(fLednickyHFunctionHist->GetNbinsX()));
  }
}

//________________________________________________________________________________________________________________
void ThermChargedResidual::LoadInterpHistFile(TString aFileBaseName, TString aLednickyHFunctionFileBaseName)
{
  LoadLednickyHFunctionFile(aLednickyHFunctionFileBaseName);

  TString aFileName = aFileBaseName+TString::Format("_%s.root", cAnalysisBaseTags[fResidualType]);
  fInterpHistFile = TFile::Open(aFileName);
  //--------------------------------------------------------------
  fHyperGeo1F1RealHist = (TH3D*)fInterpHistFile->Get("HyperGeo1F1Real");
  fHyperGeo1F1ImagHist = (TH3D*)fInterpHistFile->Get("HyperGeo1F1Imag");
    fHyperGeo1F1RealHist->SetDirectory(0);
    fHyperGeo1F1ImagHist->SetDirectory(0);

  fGTildeRealHist = (TH2D*)fInterpHistFile->Get("GTildeReal");
  fGTildeImagHist = (TH2D*)fInterpHistFile->Get("GTildeImag");
    fGTildeRealHist->SetDirectory(0);
    fGTildeImagHist->SetDirectory(0);

  fInterpHistFile->Close();

  if(fMinInterpKStar==0 && fMaxInterpKStar==0)
  {
    fMinInterpKStar = fHyperGeo1F1RealHist->GetXaxis()->GetBinCenter(1);
    fMaxInterpKStar = fHyperGeo1F1RealHist->GetXaxis()->GetBinCenter(fHyperGeo1F1RealHist->GetNbinsX());
  }
  else
  {
    assert(fMinInterpKStar == fHyperGeo1F1RealHist->GetXaxis()->GetBinCenter(1));
    assert(fMaxInterpKStar == fHyperGeo1F1RealHist->GetXaxis()->GetBinCenter(fHyperGeo1F1RealHist->GetNbinsX()));
  }

  fMinInterpRStar = fHyperGeo1F1RealHist->GetYaxis()->GetBinCenter(1);
  fMaxInterpRStar = fHyperGeo1F1RealHist->GetYaxis()->GetBinCenter(fHyperGeo1F1RealHist->GetNbinsY());

  fMinInterpTheta = fHyperGeo1F1RealHist->GetZaxis()->GetBinCenter(1);
  fMaxInterpTheta = fHyperGeo1F1RealHist->GetZaxis()->GetBinCenter(fHyperGeo1F1RealHist->GetNbinsZ());
}


//________________________________________________________________________________________________________________
double ThermChargedResidual::GetEta(double aKStar)
{
  if(fTurnOffCoulomb) return 0.;
  else return pow(((aKStar/hbarc)*fBohrRadius),-1);
}

//________________________________________________________________________________________________________________
double ThermChargedResidual::GetGamowFactor(double aKStar)
{
  if(fTurnOffCoulomb) return 1.;
  else
  {
    double tEta = GetEta(aKStar);
    tEta *= TMath::TwoPi();  //eta always comes with 2Pi here
    double tGamow = tEta*pow((TMath::Exp(tEta)-1),-1);
    return tGamow;
  }
}


//________________________________________________________________________________________________________________
complex<double> ThermChargedResidual::GetExpTerm(double aKStar, double aRStar, double aTheta)
{
  complex<double> tReturnValue = exp(-ImI*(aKStar/hbarc)*aRStar*cos(aTheta));
  return tReturnValue;
}

//________________________________________________________________________________________________________________
complex<double> ThermChargedResidual::BuildScatteringLength(double aKStar, double aReF0, double aImF0, double aD0)
{
  complex<double> tF0 (aReF0,aImF0);
  double tKStar = aKStar/hbarc;
  complex<double> tScattAmp;

  if(aReF0==0. && aImF0==0. && aD0==0.) tScattAmp = complex<double>(0.,0.);
  else if(fTurnOffCoulomb)
  {
    complex<double> tLastTerm (0.,tKStar);
    tScattAmp = pow((1./tF0) + 0.5*aD0*tKStar*tKStar - tLastTerm,-1);
  }
  else
  {
    double tLednickyHFunction = Interpolator::LinearInterpolate(fLednickyHFunctionHist,aKStar);
    double tImagChi = GetGamowFactor(aKStar)/(2.*GetEta(aKStar));
    complex<double> tLednickyChi (tLednickyHFunction,tImagChi);

    tScattAmp = pow((1./tF0) + 0.5*aD0*tKStar*tKStar - 2.*tLednickyChi/fBohrRadius,-1);
  }

  return tScattAmp;
}


//________________________________________________________________________________________________________________
double ThermChargedResidual::InterpolateWfSquared(double aKStarMag, double aRStarMag, double aTheta, double aReF0, double aImF0, double aD0)
{

  bool tDebug = true; //debug means use personal interpolation methods, instead of standard root ones

  double tGamow = GetGamowFactor(aKStarMag);
  complex<double> tExpTermComplex = GetExpTerm(aKStarMag,aRStarMag,aTheta);

  complex<double> tScattLenComplexConj;

  complex<double> tHyperGeo1F1Complex;
  complex<double> tGTildeComplexConj;

  if(!fTurnOffCoulomb)
  {
    double tHyperGeo1F1Real, tHyperGeo1F1Imag, tGTildeReal, tGTildeImag;
    //-------------------------------------
    if(tDebug)
    {
      tHyperGeo1F1Real = Interpolator::TrilinearInterpolate(fHyperGeo1F1RealHist,aKStarMag,aRStarMag,aTheta);
      tHyperGeo1F1Imag = Interpolator::TrilinearInterpolate(fHyperGeo1F1ImagHist,aKStarMag,aRStarMag,aTheta);

      tGTildeReal = Interpolator::BilinearInterpolate(fGTildeRealHist,aKStarMag,aRStarMag);
      tGTildeImag = Interpolator::BilinearInterpolate(fGTildeImagHist,aKStarMag,aRStarMag);
    }

    else
    {
      tHyperGeo1F1Real = fHyperGeo1F1RealHist->Interpolate(aKStarMag,aRStarMag,aTheta);
      tHyperGeo1F1Imag = fHyperGeo1F1ImagHist->Interpolate(aKStarMag,aRStarMag,aTheta);

      tGTildeReal = fGTildeRealHist->Interpolate(aKStarMag,aRStarMag);
      tGTildeImag = fGTildeImagHist->Interpolate(aKStarMag,aRStarMag);
    }

    tHyperGeo1F1Complex = complex<double> (tHyperGeo1F1Real,tHyperGeo1F1Imag);
    tGTildeComplexConj = complex<double> (tGTildeReal,-tGTildeImag);
    //-------------------------------------

    complex<double> tScattLenComplex = BuildScatteringLength(aKStarMag,aReF0,aImF0,aD0);
    tScattLenComplexConj = std::conj(tScattLenComplex);
  }

  else
  {
    tHyperGeo1F1Complex = complex<double> (1.,0.);
    tGTildeComplexConj = exp(-ImI*(aKStarMag/hbarc)*aRStarMag);

    complex<double> tScattLenComplex = BuildScatteringLength(aKStarMag,aReF0,aImF0,aD0);
    tScattLenComplexConj = std::conj(tScattLenComplex);
  }

  //-------------------------------------------

  complex<double> tResultComplex = tGamow*( norm(tHyperGeo1F1Complex) + norm(tScattLenComplexConj)*norm(tGTildeComplexConj)/(aRStarMag*aRStarMag) + 2.*real(tExpTermComplex*tHyperGeo1F1Complex*tScattLenComplexConj*tGTildeComplexConj/aRStarMag) );

  if(imag(tResultComplex) > std::numeric_limits< double >::min()) cout << "\t\t\t !!!!!!!!! Imaginary value in ThermChargedResidual::InterpolateWfSquared !!!!!" << endl;
  assert(imag(tResultComplex) < std::numeric_limits< double >::min());

  return real(tResultComplex);

}

//________________________________________________________________________________________________________________
double ThermChargedResidual::InterpolateWfSquared(TVector3* aKStar3Vec, TVector3* aRStar3Vec, double aReF0, double aImF0, double aD0)
{
  double tTheta = aKStar3Vec->Angle(*aRStar3Vec);
  double tKStarMag = aKStar3Vec->Mag();
  double tRStarMag = aRStar3Vec->Mag();

  return InterpolateWfSquared(tKStarMag, tRStarMag, tTheta, aReF0, aImF0, aD0);
}


