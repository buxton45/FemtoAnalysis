///////////////////////////////////////////////////////////////////////////
// WaveFunction:                                                         //
///////////////////////////////////////////////////////////////////////////

#include "WaveFunction.h"

#ifdef __ROOT__
ClassImp(WaveFunction)
#endif



//globals

//________________________________________________________________________
WaveFunction::WaveFunction() :
  fSession(),
  fCurrentAnalysisType(kXiKchP),
  fCurrentBohrRadius(gBohrRadiusXiK),
  fTurnOffCoulomb(false)
{
  fSession->InitializeSession();
}


//________________________________________________________________________
WaveFunction::~WaveFunction()
{
  fSession->EndSession();
}


//________________________________________________________________________
void WaveFunction::SetCurrentAnalysisType(AnalysisType aAnalysisType) 
{
  fCurrentAnalysisType = aAnalysisType;

//  assert(fCurrentAnalysisType == kXiKchP || fCurrentAnalysisType == kAXiKchM || fCurrentAnalysisType == kXiKchM || fCurrentAnalysisType == kAXiKchP);

  if(fCurrentAnalysisType == kXiKchP || fCurrentAnalysisType == kAXiKchM) fCurrentBohrRadius = -gBohrRadiusXiK; //attractive
  else if(fCurrentAnalysisType == kXiKchM || fCurrentAnalysisType == kAXiKchP) fCurrentBohrRadius = gBohrRadiusXiK; //repulsive
  else fCurrentBohrRadius = 1000000000;
}

//________________________________________________________________________
void WaveFunction::SetCurrentBohrRadius(AnalysisType aAnalysisType) 
{
  switch(aAnalysisType) {
  case kXiKchP:
  case kAXiKchM:
  case kResXiCKchP:
  case kResAXiCKchM:
    fCurrentBohrRadius = -gBohrRadiusXiK;
    break;

  case kXiKchM:
  case kAXiKchP:
  case kResXiCKchM:
  case kResAXiCKchP:
    fCurrentBohrRadius = gBohrRadiusXiK;
    break;


  case kResOmegaKchP:
  case kResAOmegaKchM:
    fCurrentBohrRadius = -gBohrRadiusOmegaK;
    break;


  case kResOmegaKchM:
  case kResAOmegaKchP:
    fCurrentBohrRadius = gBohrRadiusOmegaK;
    break;


  default:
    cout << "ERROR: WaveFunction::SetCurrentBohrRadius: Invalid aAnalysisType = " << aAnalysisType << endl;
    assert(0);
  }
}

//________________________________________________________________________
void WaveFunction::SetCurrentBohrRadius(double aBohrRadius) 
{
  fCurrentBohrRadius= aBohrRadius;
}


//________________________________________________________________________
complex<double> WaveFunction::GetTest(complex<double> aCmplx)
{
  complex<double> aResult = fSession->GetDiGamma(aCmplx);
  return aResult;
}


//________________________________________________________________________
double WaveFunction::GetEta(double aKStar)
{
  if(fTurnOffCoulomb) return 0.;
  else return pow(((aKStar/hbarc)*fCurrentBohrRadius),-1);
}


//________________________________________________________________________
double WaveFunction::GetLowerCaseXi(TVector3* aKStar3Vec, TVector3* aRStar3Vec)
{
  //Probably better to use TVecot3 dot product method, but this is fine for now
  double tKStarMag = aKStar3Vec->Mag();
  double tRStarMag = aRStar3Vec->Mag();

  double tTheta = aKStar3Vec->Angle(*aRStar3Vec);
  double tXi = (tKStarMag/hbarc)*tRStarMag*(1+TMath::Cos(tTheta));

  return tXi;
}


//________________________________________________________________________
double WaveFunction::GetLowerCaseXi(double aKStarMag, double aRStarMag, double aTheta)
{
  return (aKStarMag/hbarc)*aRStarMag*(1+TMath::Cos(aTheta));
}


//________________________________________________________________________
double WaveFunction::GetCoulombPhaseShift(double aKStar, double aL)
{
  double tEta = GetEta(aKStar);
  std::complex<double> tComplex (aL+1,tEta);
  double tShift = std::arg(fSession->GetGamma(tComplex));
  return tShift;
}

//________________________________________________________________________
double WaveFunction::GetCoulombSWavePhaseShift(double aKStar)
{
  if(fTurnOffCoulomb) return 0.;
  else return GetCoulombPhaseShift(aKStar,0);
}


//________________________________________________________________________
double WaveFunction::GetGamowFactor(double aKStar)
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


//________________________________________________________________________
double WaveFunction::GetLednickyHFunction(double aKStar)
{
  if(fTurnOffCoulomb) return 0.;
  else
  {
    double tEta = GetEta(aKStar);
    complex<double> tEtaCmplxPos (0.,tEta);
    complex<double> tEtaCmplxNeg (0.,-tEta);

    complex<double> tReturnValueCmplx = 0.5*(fSession->GetDiGamma(tEtaCmplxPos) + fSession->GetDiGamma(tEtaCmplxNeg) - log(tEta*tEta));

    //Note:  tReturnValueCmplx will be real, but the silly compiler yells about returning a real using complex (GetDiGamma) functions
    //Nonetheless, let's be absolutely certain that the imaginary part is zero
    bool tFail = false;
    if(std::abs(imag(tReturnValueCmplx)) > std::numeric_limits< double >::min()) {cout << "\t\t\t !!!!!!!!! Imaginary value in WaveFunction::GetLednickyHFunction !!!!!" << endl; tFail=true;}
    if(std::isnan(std::abs(imag(tReturnValueCmplx)))) {cout << "\t\t\t !!!!!!!!! NaN Imaginary value in WaveFunction::GetLednickyHFunction !!!!!" << endl; tFail=true;}
    if(tFail)
    {
      cout << "aKStar = " << aKStar << endl;
      cout << "tEta = " << tEta << endl;
      cout << "fSession->GetDiGamma(tEtaCmplxPos) = " << fSession->GetDiGamma(tEtaCmplxPos) << endl;
      cout << "fSession->GetDiGamma(tEtaCmplxNeg) = " << fSession->GetDiGamma(tEtaCmplxNeg) << endl;
      cout << "log(tEta*tEta) = " << log(tEta*tEta) << endl;
      cout << "real(tReturnValueCmplx) = " << real(tReturnValueCmplx) << endl;
      cout << "imag(tReturnValueCmplx) = " << imag(tReturnValueCmplx) << endl;
      cout << endl;
    }
    assert(std::abs(imag(tReturnValueCmplx)) < std::numeric_limits< double >::min());
    assert(!std::isnan(std::abs(imag(tReturnValueCmplx))));

    double tReturnValue = real(tReturnValueCmplx);
    return tReturnValue;
  }
}

//________________________________________________________________________
complex<double> WaveFunction::GetLednickyChiFunction(double aKStar)
{
  //WARNING: If fTurnOffCoulomb=true, this will return a NaN
  //TODO Probably eliminate LednickyChiFunction in favor of separate HFunction and Gamow
  double tReal = GetLednickyHFunction(aKStar);
  double tImag = GetGamowFactor(aKStar)/(2.*GetEta(aKStar));

  complex<double> tChi (tReal,tImag);
  return tChi;
}

//________________________________________________________________________
complex<double> WaveFunction::GetScatteringLength(double aKStar, double aReF0, double aImF0, double aD0)
{
  complex<double> tF0 (aReF0,aImF0);
  double tKStar = aKStar/hbarc;
  complex<double> tScattAmp;

  if(fTurnOffCoulomb)
  {
    complex<double> tLastTerm (0.,tKStar);
    tScattAmp = pow((1./tF0) + 0.5*aD0*tKStar*tKStar - tLastTerm,-1);
  }
  else
  {
    complex<double> tChi = GetLednickyChiFunction(aKStar);
    tScattAmp = pow((1./tF0) + 0.5*aD0*tKStar*tKStar - 2.*tChi/fCurrentBohrRadius,-1);
  }

  return tScattAmp;
}

//________________________________________________________________________
complex<double> WaveFunction::GetGTilde(double aKStar, double aRStar)
{
  complex<double> tGTilde;

  if(fTurnOffCoulomb)
  {
    tGTilde = exp(ImI*(aKStar/hbarc)*aRStar);
  }
  else
  {
    double tSWaveShift = GetCoulombSWavePhaseShift(aKStar);
    double tEta = GetEta(aKStar);
    double tRho = (aKStar/hbarc)*aRStar;

    tGTilde = TMath::Sqrt(GetGamowFactor(aKStar))*fSession->GetCoulombHpmFunction(tSWaveShift,tEta,tRho,MathematicaSession::kHPlus,0.);
  }

  return tGTilde;
}

//________________________________________________________________________
complex<double> WaveFunction::GetGTildeConjugate(double aKStar, double aRStar)
{
  complex<double> tGTildeConj;

  if(fTurnOffCoulomb)
  {
    tGTildeConj = exp(-ImI*(aKStar/hbarc)*aRStar);
  }
  else
  {
    double tSWaveShift = GetCoulombSWavePhaseShift(aKStar);
    double tEta = GetEta(aKStar);
    double tRho = (aKStar/hbarc)*aRStar;

    tGTildeConj = TMath::Sqrt(GetGamowFactor(aKStar))*fSession->GetCoulombHpmFunction(tSWaveShift,tEta,tRho,MathematicaSession::kHMinus,0.);
  }

  return tGTildeConj;
}

//________________________________________________________________________
complex<double> WaveFunction::GetWaveFunction(TVector3* aKStar3Vec, TVector3* aRStar3Vec, double aReF0, double aImF0, double aD0)
{
  double tKStarMag = aKStar3Vec->Mag();
  double tRStarMag = aRStar3Vec->Mag();

  complex<double> tMultFact = std::exp(ImI*GetCoulombSWavePhaseShift(tKStarMag))*TMath::Sqrt(GetGamowFactor(tKStarMag));

  double tEta = GetEta(tKStarMag);
  double tXi = GetLowerCaseXi(aKStar3Vec,aRStar3Vec);

  complex<double> tA (0.,-tEta);
  complex<double> tB (1.,0.);
  complex<double> tZ (0.,tXi);

  complex<double> tTerm1;
  if(fTurnOffCoulomb) tTerm1 = std::exp(-(ImI/hbarc)*aKStar3Vec->Dot(*aRStar3Vec));
  else tTerm1 = std::exp(-(ImI/hbarc)*aKStar3Vec->Dot(*aRStar3Vec))*fSession->GetHyperGeo1F1(tA,tB,tZ);
  complex<double> tTerm2 = GetScatteringLength(tKStarMag,aReF0,aImF0,aD0)*GetGTilde(tKStarMag,tRStarMag)/tRStarMag;

  complex<double> tWaveFunction = tMultFact*(tTerm1+tTerm2);

  return tWaveFunction;
}


//________________________________________________________________________
complex<double> WaveFunction::GetWaveFunction(double aKStarMag, double aRStarMag, double aTheta, double aReF0, double aImF0, double aD0)
{
  double tKStarDotRStar = (aKStarMag/hbarc)*aRStarMag*cos(aTheta);

  complex<double> tMultFact = std::exp(ImI*GetCoulombSWavePhaseShift(aKStarMag))*TMath::Sqrt(GetGamowFactor(aKStarMag));

  double tEta = GetEta(aKStarMag);
  double tXi = GetLowerCaseXi(aKStarMag,aRStarMag,aTheta);

  complex<double> tA (0.,-tEta);
  complex<double> tB (1.,0.);
  complex<double> tZ (0.,tXi);

  complex<double> tTerm1;
  if(fTurnOffCoulomb) tTerm1 = std::exp(-ImI*tKStarDotRStar);
  else tTerm1 = std::exp(-ImI*tKStarDotRStar)*fSession->GetHyperGeo1F1(tA,tB,tZ);
  complex<double> tTerm2 = GetScatteringLength(aKStarMag,aReF0,aImF0,aD0)*GetGTilde(aKStarMag,aRStarMag)/aRStarMag;

  complex<double> tWaveFunction = tMultFact*(tTerm1+tTerm2);

  return tWaveFunction;
}



