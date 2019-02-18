// file SimulatedLednickyCf.cxx

#include "SimulatedLednickyCf.h"

#ifdef __ROOT__
ClassImp(SimulatedLednickyCf)
#endif


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
SimulatedLednickyCf::SimulatedLednickyCf(double aKStarBinSize, double aMaxBuildKStar, int aNPairsPerKStarBin) :
  fGenerator(std::chrono::system_clock::now().time_since_epoch().count()),
  fNPairsPerKStarBin(aNPairsPerKStarBin),
  fKStarBinSize(aKStarBinSize),
  fMaxBuildKStar(aMaxBuildKStar),

  fCurrentRadius(1.),
  fCurrentMuOut(0.),
  fPair3dVec(0)
{
  BuildPair3dVec(fNPairsPerKStarBin, fKStarBinSize);
}


//________________________________________________________________________________________________________________
SimulatedLednickyCf::~SimulatedLednickyCf()
{}



//________________________________________________________________________________________________________________
void SimulatedLednickyCf::SetRandomKStar3Vec(TVector3* aKStar3Vec, double aKStarMagMin, double aKStarMagMax)
{
  std::uniform_real_distribution<double> tKStarMagDistribution(aKStarMagMin,aKStarMagMax);
  std::uniform_real_distribution<double> tUnityDistribution(0.,1.);

  double tKStarMag = tKStarMagDistribution(fGenerator);
  double tU = tUnityDistribution(fGenerator);
  double tV = tUnityDistribution(fGenerator);

  double tTheta = acos(2.*tV-1.); //polar angle
  double tPhi = 2.*M_PI*tU; //azimuthal angle

  aKStar3Vec->SetMagThetaPhi(tKStarMag,tTheta,tPhi);
}


//________________________________________________________________________________________________________________
void SimulatedLednickyCf::BuildPair3dVec(int aNPairsPerKStarBin, double aBinSize)
{
  ChronoTimer tTimer(kSec);
  tTimer.Start();

  fNPairsPerKStarBin = aNPairsPerKStarBin;
  fPair3dVec.clear();

  double tBinSize = aBinSize;
  int tNBinsKStar = std::round(fMaxBuildKStar/tBinSize);  //TODO make this general, ie subtract 1 if fMaxBuildKStar is on bin edge (maybe, maybe not bx of iKStarBin<tNBinsKStar)

  //Create the source Gaussians
  double tRoot2 = sqrt(2.);  //need this scale to get 4 on denominator of exp in normal dist instead of 2
  double tRadius = 1.0;
  fCurrentRadius = tRadius;
  fCurrentMuOut = 0.;
  std::normal_distribution<double> tROutSource(0.,tRoot2*tRadius);
  std::normal_distribution<double> tRSideSource(0.,tRoot2*tRadius);
  std::normal_distribution<double> tRLongSource(0.,tRoot2*tRadius);

  TVector3* tKStar3Vec = new TVector3(0.,0.,0.);

  double tKStarMagMin, tKStarMagMax;
  td1dVec tTempPair(7);
  td2dVec tTemp2dVec;

  //------------------------------------
//TODO Check randomization
//TODO Make sure I am grabbing from correct tBin.  Must work even when I rebin things
  for(int iKStarBin=0; iKStarBin<tNBinsKStar; iKStarBin++)
  {
    tKStarMagMin = iKStarBin*tBinSize;
//    if(iKStarBin==0) tKStarMagMin=0.004;  //TODO here and in ChargedResidualCf
    tKStarMagMax = (iKStarBin+1)*tBinSize;
    tTemp2dVec.clear();
    for(int iPair=0; iPair<fNPairsPerKStarBin; iPair++)
    {
      SetRandomKStar3Vec(tKStar3Vec,tKStarMagMin,tKStarMagMax);

      tTempPair[0] = tKStar3Vec->X();
      tTempPair[1] = tKStar3Vec->Y();
      tTempPair[2] = tKStar3Vec->Z();

      //TODO: for now, spherically symmetric
      tTempPair[3] = tROutSource(fGenerator);
      tTempPair[4] = tRSideSource(fGenerator);
      tTempPair[5] = tRLongSource(fGenerator);

      tTempPair[6] = tTempPair[3];

      tTemp2dVec.push_back(tTempPair);
    }
    fPair3dVec.push_back(tTemp2dVec);
  }

  delete tKStar3Vec;

  tTimer.Stop();
  cout << "BuildPair3dVec finished: ";
  tTimer.PrintInterval();
}

//________________________________________________________________________________________________________________
void SimulatedLednickyCf::UpdatePairRadiusParameter(double aNewRadius, double aMuOut)
{
  //RSide and RLong can just be rescaled, as was the procedure before
  //As for ROut, since I am allowing a MuOut now, things are a little more complicated
  //  I think the easiest way is to always start from a Gaussian with radius=1 and mu=0, and to shift from there

  double tScaleFactor = aNewRadius/fCurrentRadius;
  fCurrentRadius = aNewRadius;
  fCurrentMuOut = aMuOut;

//TODO Make sure I am grabbing from correct tBin.  Must work even when I rebin things
  for(int iKStarBin=0; iKStarBin<(int)fPair3dVec.size(); iKStarBin++)
  {
    for(int iPair=0; iPair<(int)fPair3dVec[iKStarBin].size(); iPair++)
    {
        fPair3dVec[iKStarBin][iPair][4] *= tScaleFactor;  //RSide
        fPair3dVec[iKStarBin][iPair][5] *= tScaleFactor;  //RLong

        //I think correct procedure here should be to rescale first, then shift
        fPair3dVec[iKStarBin][iPair][3] = aNewRadius*fPair3dVec[iKStarBin][iPair][6] + fCurrentMuOut;
    }
  }
}



//________________________________________________________________________________________________________________
bool SimulatedLednickyCf::AreParamsSameExcludingLambda(double *aCurrent, double *aNew, int aNEntries)
{
//TODO NOTE, want to exclude Lambda parameter here, becuase it's always changing
  bool tAreSame = true;
  for(int i=1; i<aNEntries; i++)
  {
    if(abs(aCurrent[i]-aNew[i]) > std::numeric_limits< double >::min()) tAreSame = false;
  }

  if(!tAreSame)
  {
    for(int i=0; i<aNEntries; i++) aCurrent[i] = aNew[i];
  }

  return tAreSame;
}

//________________________________________________________________________________________________________________
void SimulatedLednickyCf::AdjustLambdaParam(td1dVec &aCoulombCf, double aOldLambda, double aNewLambda)
{
  double tRawValue, tCurrentValue;
  for(unsigned int i=0; i<aCoulombCf.size(); i++)
  {
    tCurrentValue = aCoulombCf[i];
    tRawValue = (1.0/aOldLambda)*(tCurrentValue-(1.0-aOldLambda));

    aCoulombCf[i] = aNewLambda*tRawValue + (1.0-aNewLambda);
  }
}

//________________________________________________________________________________________________________________
complex<double> SimulatedLednickyCf::GetStrongOnlyWaveFunction(const TVector3 &aKStar3Vec, const TVector3 &aRStar3Vec, const double aRef0, const double aImf0, const double ad0)
{
  complex<double> tf0 (aRef0, aImf0);

  double tKdotR = aKStar3Vec.Dot(aRStar3Vec);
    tKdotR /= hbarc;
  double tKStarMag = aKStar3Vec.Mag();
    tKStarMag /= hbarc;
  double tRStarMag = aRStar3Vec.Mag();

  complex<double> tScattLenLastTerm (0., tKStarMag);
  complex<double> tScattAmp = pow((1./tf0) + 0.5*ad0*tKStarMag*tKStarMag - tScattLenLastTerm,-1);

  complex<double> tReturnWf = exp(-ImI*tKdotR) + tScattAmp*exp(ImI*tKStarMag*tRStarMag)/tRStarMag;
  return tReturnWf;
}

//________________________________________________________________________________________________________________
double SimulatedLednickyCf::GetStrongOnlyWaveFunctionSq(const TVector3 &aKStar3Vec, const TVector3 &aRStar3Vec, const double aRef0, const double aImf0, const double ad0)
{
  complex<double> tWf = GetStrongOnlyWaveFunction(aKStar3Vec, aRStar3Vec, aRef0, aImf0, ad0);
  double tWfSq = norm(tWf);
  return tWfSq;
}


//________________________________________________________________________________________________________________
double SimulatedLednickyCf::GetFitCfContent(double aKStarMagMin,/* double aKStarMagMax,*/ double *par)
{
  omp_set_num_threads(6);

  // par[0] = Lambda 
  // par[1] = Radius
  // par[2] = Ref0
  // par[3] = Imf0
  // par[4] = d0
  // par[5] = norm
  // par[6] = muOut

  if(abs(par[1]-fCurrentRadius) > std::numeric_limits< double >::min() ||
     abs(par[6]-fCurrentMuOut) > std::numeric_limits< double >::min()) 
  {
    UpdatePairRadiusParameter(par[1], par[6]); 
  }

  //TODO make this general
  //This is messing up around aKStarMagMin = 0.29, or bin 57/58
  //Probably fixed with use of std::round, but need to double check
//  int tBin = std::round(aKStarMagMin/fKStarBinSize);
  int tBin = std::floor(aKStarMagMin/fKStarBinSize);

  //KStarOut  = fPair3dVec[tBin][i][0]
  //KStarSide = fPair3dVec[tBin][i][1]
  //KStarLong = fPair3dVec[tBin][i][2]

  //RStarOut  = fPair3dVec[tBin][i][3]
  //RStarSide = fPair3dVec[tBin][i][4]
  //RStarLong = fPair3dVec[tBin][i][5]

  int tCounter = 0;
  double tReturnCfContent = 0.;

  int tMaxKStarCalls;
  tMaxKStarCalls = fPair3dVec[tBin].size();

  double tWaveFunctionSq;
  TVector3 tKStar3Vec = TVector3(0.,0.,0.);
  TVector3 tRStar3Vec = TVector3(0.,0.,0.);

//  #pragma omp parallel for reduction(+: tCounter) reduction(+: tReturnCfContent) private(tWaveFunctionSq) firstprivate(tKStar3Vec, tRStar3Vec)
  for(int i=0; i<tMaxKStarCalls; i++)
  {
    tKStar3Vec.SetXYZ(fPair3dVec[tBin][i][0], fPair3dVec[tBin][i][1], fPair3dVec[tBin][i][2]);
    tRStar3Vec.SetXYZ(fPair3dVec[tBin][i][3], fPair3dVec[tBin][i][4], fPair3dVec[tBin][i][5]);

    tWaveFunctionSq = GetStrongOnlyWaveFunctionSq(tKStar3Vec, tRStar3Vec, par[2], par[3], par[4]);
    tReturnCfContent += tWaveFunctionSq;
    tCounter++;
  }

  tReturnCfContent /= tCounter;
  tReturnCfContent = (par[0]*tReturnCfContent + (1.-par[0]));  //C = (Lam*C_gen + (1-Lam));
  return tReturnCfContent;
}

