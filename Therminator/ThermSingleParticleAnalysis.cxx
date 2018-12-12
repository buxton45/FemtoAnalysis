/* ThermSingleParticleAnalysis.cxx */

#include "ThermSingleParticleAnalysis.h"

#ifdef __ROOT__
ClassImp(ThermSingleParticleAnalysis)
#endif


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
ThermSingleParticleAnalysis::ThermSingleParticleAnalysis(ParticlePDGType aParticlePDGType) :
  fParticlePDGType(aParticlePDGType),
  fBuildUniqueParents(false),
  fUniqueParents(0),
  fAllFathers(0),

  fParents(nullptr),
  fRadii(nullptr),
  f2dRadiiVsPid(nullptr),
  f2dRadiiVsBeta(nullptr),
  f3dRadii(nullptr),

  fTransverseEmission(nullptr),
  fTransverseEmissionPrimaryOnly(nullptr),

  fTransverseEmissionVsTau(nullptr),
  fTransverseEmissionVsTauPrimaryOnly(nullptr)
{
  fParents = new TH1D(TString::Format("%sParents",GetPDGRootName(fParticlePDGType)), 
                      TString::Format("%sParents",GetPDGRootName(fParticlePDGType)), 
                      200, 0, 200);

  fRadii = new TH1D(TString::Format("%sRadii",GetPDGRootName(fParticlePDGType)), 
                      TString::Format("%sRadii",GetPDGRootName(fParticlePDGType)), 
                      1000, 0, 1000);

  f2dRadiiVsPid = new TH2D(TString::Format("%s2dRadiiVsPid",GetPDGRootName(fParticlePDGType)),
                           TString::Format("%s2dRadiiVsPid",GetPDGRootName(fParticlePDGType)),
                           200, 0, 200,
                           1000, 0, 1000);

  f2dRadiiVsBeta = new TH2D(TString::Format("%s2dRadiiVsBeta",GetPDGRootName(fParticlePDGType)),
                           TString::Format("%s2dRadiiVsBeta",GetPDGRootName(fParticlePDGType)),
                           100, 0, 1.,
                           1000, 0, 1000);

  f3dRadii = new TH3D(TString::Format("%s3dRadii",GetPDGRootName(fParticlePDGType)),
                      TString::Format("%s3dRadii",GetPDGRootName(fParticlePDGType)),
                      200, 0, 200, 
                      100, 0, 1., 
                      1000, 0, 1000);

  fTransverseEmission = new TH3D(TString::Format("%sTransverseEmission",GetPDGRootName(fParticlePDGType)),
                                 TString::Format("%sTransverseEmission",GetPDGRootName(fParticlePDGType)),
                                 400, -20., 20.,
                                 400, -20., 20.,
                                 200, -1., 1.);

  fTransverseEmissionPrimaryOnly = new TH3D(TString::Format("%sTransverseEmissionPrimaryOnly",GetPDGRootName(fParticlePDGType)),
                                           TString::Format("%sTransverseEmissionPrimaryOnly",GetPDGRootName(fParticlePDGType)),
                                           400, -20., 20.,
                                           400, -20., 20.,
                                           200, -1., 1.);

  fTransverseEmissionVsTau = new TH3D(TString::Format("%sTransverseEmissionVsTau",GetPDGRootName(fParticlePDGType)),
                                           TString::Format("%sTransverseEmissionVsTau",GetPDGRootName(fParticlePDGType)),
                                           200, 0, 20.,
                                           200, 0, 20.,
                                           100, 0., 1.);

  fTransverseEmissionVsTauPrimaryOnly = new TH3D(TString::Format("%sTransverseEmissionVsTauPrimaryOnly",GetPDGRootName(fParticlePDGType)),
                                                 TString::Format("%sTransverseEmissionVsTauPrimaryOnly",GetPDGRootName(fParticlePDGType)),
                                                 200, 0, 20.,
                                                 200, 0, 20.,
                                                 100, 0., 1.);

  //-------------------------------
  switch(fParticlePDGType) {
  case kPDGLam:
  case kPDGALam:
    fAllFathers = cAllLambdaFathers;
    break;

  case kPDGKchP:
  case kPDGKchM:
    fAllFathers = cAllKchFathers;
    break;

  case kPDGK0:
    fAllFathers = cAllK0ShortFathers;
    break;

  case kPDGProt:
  case kPDGAntiProt:
    fAllFathers = cAllProtonFathers;
    break;

  default:
    cout << "Error in ThermSingleParticleAnalysis constructor, invalide fParticlePDGType = " << fParticlePDGType << " selected." << endl;
    assert(0);
  }
}



//________________________________________________________________________________________________________________
ThermSingleParticleAnalysis::~ThermSingleParticleAnalysis()
{
  delete fParents;
  delete fRadii;
  delete f2dRadiiVsPid;
  delete f2dRadiiVsBeta;
  delete f3dRadii;

  delete fTransverseEmission;
  delete fTransverseEmissionPrimaryOnly;

  delete fTransverseEmissionVsTau;
  delete fTransverseEmissionVsTauPrimaryOnly;
}

//________________________________________________________________________________________________________________
double ThermSingleParticleAnalysis::GetSampledCTau(double aMeanCTau)
{
  double tLambda = 1.0/aMeanCTau;
  std::default_random_engine tGenerator (std::clock());  //std::clock() is seed
  std::exponential_distribution<double> tExpDistribution(tLambda);
  double tReturnCTau = tExpDistribution(tGenerator);
//TODO
//cout << "aMeanCTau = " << aMeanCTau << endl;
//cout << "tReturnCTau = " << tReturnCTau << endl << endl;
  return tReturnCTau;
}


//________________________________________________________________________________________________________________
double ThermSingleParticleAnalysis::GetLabDecayLength(double aMeanCTau, double aMass, double aE, double aMagP)
{
  double tGamma = aE/aMass;
  double tBeta = aMagP/aE;
  return tBeta*tGamma*GetSampledCTau(aMeanCTau);
}

/*
//________________________________________________________________________________________________________________
double ThermSingleParticleAnalysis::GetLabDecayLength(double aMeanCTau, double aMass, double aE, double aMagP)
{
  double tGamma = aE/aMass;
  double tBeta = aMagP/aE;
  double tLambda = 1.0/(tGamma*tBeta*aMeanCTau);
  std::default_random_engine tGenerator (std::clock());  //std::clock() is seed
  std::exponential_distribution<double> tExpDistribution(tLambda);
  double tReturnDecayLength = tExpDistribution(tGenerator);

  return tReturnDecayLength;
}
*/

//________________________________________________________________________________________________________________
void ThermSingleParticleAnalysis::MapAndFillParents(ThermParticle &aParticle)
{
  int tBin=-1;
  int tFatherType = aParticle.GetFatherPID();
  for(unsigned int i=0; i<fAllFathers.size(); i++) if(tFatherType==fAllFathers[i]) tBin=i;

  if(!fBuildUniqueParents)  //If true, I am likely looking for new parents, so I don't want these asserts to be tripped
  {
    if(tBin==-1) cout << "FAILURE IMMINENT: tFatherType = " << tFatherType << endl;
    assert(tBin>-1);
  }
  fParents->Fill(tBin);
}


//________________________________________________________________________________________________________________
void ThermSingleParticleAnalysis::MapAndFillRadiiHistograms(ThermParticle &aParticle)
{
  int tBin=-1;
  int tFatherType = aParticle.GetFatherPID();
  for(unsigned int i=0; i<fAllFathers.size(); i++) if(tFatherType==fAllFathers[i]) tBin=i;

  if(!fBuildUniqueParents)  //If true, I am likely looking for new parents, so I don't want these asserts to be tripped
  {
    if(tBin==-1) cout << "FAILURE IMMINENT: tFatherType = " << tFatherType << endl;
    assert(tBin>-1);
  }

  double tDecayLength;
/*
  if(aParticle.IsPrimordial()) tDecayLength = 0.;
  else tDecayLength = GetLabDecayLength(GetParticleDecayLength(tFatherType), aParticle.GetFatherMass(), aParticle.GetFatherE(), aParticle.GetFatherMagP());
*/
  tDecayLength = sqrt(aParticle.GetX()*aParticle.GetX() + aParticle.GetY()*aParticle.GetY() + aParticle.GetZ()*aParticle.GetZ());
/*
  if(!aParticle.IsPrimordial())
  {
    double tParticleOrigin, tParentOrigin, tDiff, tLabDecayLength;
    tParticleOrigin = sqrt(aParticle.GetX()*aParticle.GetX() + aParticle.GetY()*aParticle.GetY() + aParticle.GetZ()*aParticle.GetZ());
    tParentOrigin = sqrt(aParticle.GetFatherX()*aParticle.GetFatherX() + aParticle.GetFatherY()*aParticle.GetFatherY() + aParticle.GetFatherZ()*aParticle.GetFatherZ());
    tDiff = sqrt((aParticle.GetFatherX()-aParticle.GetX())*(aParticle.GetFatherX()-aParticle.GetX()) + 
                 (aParticle.GetFatherY()-aParticle.GetY())*(aParticle.GetFatherY()-aParticle.GetY()) +
                 (aParticle.GetFatherZ()-aParticle.GetZ())*(aParticle.GetFatherZ()-aParticle.GetZ()));
    tLabDecayLength = GetLabDecayLength(GetParticleDecayLength(tFatherType), aParticle.GetFatherMass(), aParticle.GetFatherE(), aParticle.GetFatherMagP());
    cout << "tParticleOrigin = " << tParticleOrigin << endl;
    cout << "tParentOrigin = " << tParentOrigin << endl;
    cout << "tDiff = " << tDiff << endl;
    cout << "tLabDecayLength = " << tLabDecayLength << endl << endl;
  }
*/
//  double tBeta = aParticle.GetFatherMagP()/aParticle.GetFatherE();
  double tBeta = aParticle.GetMagP()/aParticle.GetE();

  fRadii->Fill(tDecayLength);
  f2dRadiiVsPid->Fill(tBin, tDecayLength);
  f2dRadiiVsBeta->Fill(tBeta, tDecayLength);
  f3dRadii->Fill(tBin, tBeta, tDecayLength);

  double tBetaX = aParticle.GetPx()/aParticle.GetE();
  double tBetaY = aParticle.GetPy()/aParticle.GetE();

  if(tBetaY > -0.1 && tBetaY < 0.1) 
  {
    fTransverseEmission->Fill(aParticle.GetX(), aParticle.GetY(), tBetaX);
    if(aParticle.IsPrimordial()) fTransverseEmissionPrimaryOnly->Fill(aParticle.GetX(), aParticle.GetY(), tBetaX);
  }
  fTransverseEmissionVsTau->Fill(sqrt(aParticle.GetX()*aParticle.GetX() + aParticle.GetY()*aParticle.GetY()), aParticle.GetTau(), tBeta);  //TODO is GetTau() correct?
  if(aParticle.IsPrimordial()) fTransverseEmissionVsTauPrimaryOnly->Fill(sqrt(aParticle.GetX()*aParticle.GetX() + aParticle.GetY()*aParticle.GetY()), aParticle.GetTau(), tBeta);

}

//________________________________________________________________________________________________________________
void ThermSingleParticleAnalysis::FillUniqueParticleParents(ThermParticle &aParticle)
{
  bool bParentAlreadyIncluded = false;
  int tParticleFatherType = aParticle.GetFatherPID();

  for(unsigned int i=0; i<fUniqueParents.size(); i++)
  {
    if(fUniqueParents[i] == tParticleFatherType)
    {
      bParentAlreadyIncluded = true;
      break;
    }
  }
  if(!bParentAlreadyIncluded) fUniqueParents.push_back(tParticleFatherType);
}

//________________________________________________________________________________________________________________
void ThermSingleParticleAnalysis::PrintUniqueParents()
{
  std::sort(fUniqueParents.begin(), fUniqueParents.end());
  cout << GetPDGRootName(fParticlePDGType) << " Single Particle Analysis" << endl;
  cout << "fUniqueParents.size() = " << fUniqueParents.size() << endl;
  for(unsigned int i=0; i<fUniqueParents.size()-1; i++) cout << fUniqueParents[i] << ", ";
  cout << fUniqueParents[fUniqueParents.size()-1] << endl;
  cout << endl << endl << endl;
}

//________________________________________________________________________________________________________________
void ThermSingleParticleAnalysis::ProcessEventForV0(ThermEvent &aEvent)
{
  vector<ThermV0Particle> aParticleCollection = aEvent.GetV0ParticleCollection(fParticlePDGType);

  ThermV0Particle tParticle;
  for(unsigned int iPar=0; iPar<aParticleCollection.size(); iPar++)
  {
    tParticle = aParticleCollection[iPar];
    if(fBuildUniqueParents) FillUniqueParticleParents(tParticle);
    MapAndFillRadiiHistograms(tParticle);
    MapAndFillParents(tParticle);
  }
}

//________________________________________________________________________________________________________________
void ThermSingleParticleAnalysis::ProcessEventForParticle(ThermEvent &aEvent)
{
  vector<ThermParticle> aParticleCollection = aEvent.GetParticleCollection(fParticlePDGType);

  ThermParticle tParticle;
  for(unsigned int iPar=0; iPar<aParticleCollection.size(); iPar++)
  {
    tParticle = aParticleCollection[iPar];
    if(fBuildUniqueParents) FillUniqueParticleParents(tParticle);
    MapAndFillRadiiHistograms(tParticle);
    MapAndFillParents(tParticle);
  }
}




//________________________________________________________________________________________________________________
void ThermSingleParticleAnalysis::ProcessEvent(ThermEvent &aEvent)
{
  if(fParticlePDGType==kPDGLam || fParticlePDGType==kPDGALam || fParticlePDGType==kPDGK0) ProcessEventForV0(aEvent);
  else if(fParticlePDGType==kPDGKchP || fParticlePDGType==kPDGKchM ||
          fParticlePDGType==kPDGProt || fParticlePDGType==kPDGAntiProt) ProcessEventForParticle(aEvent);
  else assert(0);
}


//________________________________________________________________________________________________________________
void ThermSingleParticleAnalysis::SaveAll(TFile *aFile)
{
  if(fBuildUniqueParents) PrintUniqueParents();

  assert(aFile->IsOpen());
  
  fParents->Write();
  fRadii->Write();
  f2dRadiiVsPid->Write();
  f2dRadiiVsBeta->Write();
  f3dRadii->Write();

  fTransverseEmission->Write();
  fTransverseEmissionPrimaryOnly->Write();

  fTransverseEmissionVsTau->Write();
  fTransverseEmissionVsTauPrimaryOnly->Write();
}

