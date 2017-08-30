/* ThermPairAnalysis.cxx */

#include "ThermPairAnalysis.h"

#ifdef __ROOT__
ClassImp(ThermPairAnalysis)
#endif


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
ThermPairAnalysis::ThermPairAnalysis(AnalysisType aAnType) :
  fAnalysisType(aAnType),

  fKStarMin(0.),
  fKStarMax(1.),
  fNBinsKStar(200),

  fBuildUniqueParents(false),

  fUniqueParents1(0),
  fUniqueParents2(0),

  fTransformStorageMapping(0),
  fTransformInfo(),
  fTransformMatrices(nullptr),

  fPairFractions(nullptr),
  fParentsMatrix(nullptr),

  fPrimaryPairInfo(0),
  fOtherPairInfo(0)
{
  InitiateTransformMatrices();

  TString tPairFractionsName = TString::Format("PairFractions%s", cAnalysisBaseTags[aAnType]);
  fPairFractions = new TH1D(tPairFractionsName, tPairFractionsName, 12, 0, 12);

  TString tParentsMatrixName = TString::Format("ParentsMatrix%s", cAnalysisBaseTags[aAnType]);
  fParentsMatrix = new TH2D(tParentsMatrixName, tParentsMatrixName, 100, 0, 100, 135, 0, 135);
}



//________________________________________________________________________________________________________________
ThermPairAnalysis::~ThermPairAnalysis()
{
/*no-op*/
}


//________________________________________________________________________________________________________________
void ThermPairAnalysis::InitiateTransformMatrices()
{
  cout << "_______________________________________________________________________________________" << endl;
  cout << "InitiateTransformMatrices called for analysis of type " << cAnalysisBaseTags[fAnalysisType] << endl;

  switch(fAnalysisType) {
  case kLamKchP:
    fTransformStorageMapping = vector<AnalysisType>{kResSig0KchP, kResXiCKchP, kResXi0KchP, kResOmegaKchP, kResSigStPKchP, 
                                                    kResSigStMKchP, kResSigSt0KchP, kResLamKSt0, kResSig0KSt0, kResXiCKSt0, kResXi0KSt0};

    fTransformInfo.emplace_back(kPDGLam, kPDGKchP, kPDGSigma, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchP, kPDGXiC, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchP, kPDGXi0, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchP, kPDGOmega, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchP, kPDGSigStP, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchP, kPDGSigStM, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchP, kPDGSigSt0, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchP, kPDGLam, kPDGKSt0);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchP, kPDGSigma, kPDGKSt0);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchP, kPDGXiC, kPDGKSt0);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchP, kPDGXi0, kPDGKSt0);

    break;

  case kALamKchM:
    fTransformStorageMapping = vector<AnalysisType>{kResASig0KchM, kResAXiCKchM, kResAXi0KchM, kResAOmegaKchM, kResASigStMKchM, 
                                                    kResASigStPKchM, kResASigSt0KchM, kResALamAKSt0, kResASig0AKSt0, kResAXiCAKSt0, kResAXi0AKSt0};

    fTransformInfo.emplace_back(kPDGALam, kPDGKchM, kPDGASigma, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchM, kPDGAXiC, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchM, kPDGAXi0, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchM, kPDGAOmega, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchM, kPDGASigStM, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchM, kPDGASigStP, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchM, kPDGASigSt0, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchM, kPDGALam, kPDGAKSt0);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchM, kPDGASigma, kPDGAKSt0);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchM, kPDGAXiC, kPDGAKSt0);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchM, kPDGAXi0, kPDGAKSt0);

    break;

  case kLamKchM:
    fTransformStorageMapping = vector<AnalysisType>{kResSig0KchM, kResXiCKchM, kResXi0KchM, kResOmegaKchM, kResSigStPKchM, 
                                                    kResSigStMKchM, kResSigSt0KchM, kResLamAKSt0, kResSig0AKSt0, kResXiCAKSt0, kResXi0AKSt0};

    fTransformInfo.emplace_back(kPDGLam, kPDGKchM, kPDGSigma, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchM, kPDGXiC, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchM, kPDGXi0, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchM, kPDGOmega, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchM, kPDGSigStP, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchM, kPDGSigStM, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchM, kPDGSigSt0, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchM, kPDGLam, kPDGAKSt0);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchM, kPDGSigma, kPDGAKSt0);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchM, kPDGXiC, kPDGAKSt0);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchM, kPDGXi0, kPDGAKSt0);

    break;

  case kALamKchP:
    fTransformStorageMapping = vector<AnalysisType>{kResASig0KchP, kResAXiCKchP, kResAXi0KchP, kResAOmegaKchP, kResASigStMKchP, 
                                                    kResASigStPKchP, kResASigSt0KchP, kResALamKSt0, kResASig0KSt0, kResAXiCKSt0, kResAXi0KSt0};

    fTransformInfo.emplace_back(kPDGALam, kPDGKchP, kPDGASigma, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchP, kPDGAXiC, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchP, kPDGAXi0, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchP, kPDGAOmega, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchP, kPDGASigStM, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchP, kPDGASigStP, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchP, kPDGASigSt0, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchP, kPDGALam, kPDGKSt0);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchP, kPDGASigma, kPDGKSt0);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchP, kPDGAXiC, kPDGKSt0);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchP, kPDGAXi0, kPDGKSt0);

    break;

  case kLamK0:
    fTransformStorageMapping = vector<AnalysisType>{kResSig0K0, kResXiCK0, kResXi0K0, kResOmegaK0, kResSigStPK0, 
                                                    kResSigStMK0, kResSigSt0K0, kResLamKSt0ToLamK0, kResSig0KSt0ToLamK0, kResXiCKSt0ToLamK0, kResXi0KSt0ToLamK0};

    fTransformInfo.emplace_back(kPDGLam, kPDGK0, kPDGSigma, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGK0, kPDGXiC, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGK0, kPDGXi0, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGK0, kPDGOmega, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGK0, kPDGSigStP, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGK0, kPDGSigStM, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGK0, kPDGSigSt0, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGK0, kPDGLam, kPDGKSt0);
    fTransformInfo.emplace_back(kPDGLam, kPDGK0, kPDGSigma, kPDGKSt0);
    fTransformInfo.emplace_back(kPDGLam, kPDGK0, kPDGXiC, kPDGKSt0);
    fTransformInfo.emplace_back(kPDGLam, kPDGK0, kPDGXi0, kPDGKSt0);

    break;

  case kALamK0:
    fTransformStorageMapping = vector<AnalysisType>{kResASig0K0, kResAXiCK0, kResAXi0K0, kResAOmegaK0, kResASigStMK0, 
                                                    kResASigStPK0, kResASigSt0K0, kResALamKSt0ToALamK0, kResASig0KSt0ToALamK0, kResAXiCKSt0ToALamK0, kResAXi0KSt0ToALamK0};

    fTransformInfo.emplace_back(kPDGALam, kPDGK0, kPDGASigma, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGK0, kPDGAXiC, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGK0, kPDGAXi0, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGK0, kPDGAOmega, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGK0, kPDGASigStM, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGK0, kPDGASigStP, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGK0, kPDGASigSt0, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGK0, kPDGALam, kPDGKSt0);
    fTransformInfo.emplace_back(kPDGALam, kPDGK0, kPDGASigma, kPDGKSt0);
    fTransformInfo.emplace_back(kPDGALam, kPDGK0, kPDGAXiC, kPDGKSt0);
    fTransformInfo.emplace_back(kPDGALam, kPDGK0, kPDGAXi0, kPDGKSt0);

    break;

  default:
    cout << "Error in ThermPairAnalysis::InitiateTransformMatrices, invalide fAnalysisType = " << fAnalysisType << " selected." << endl;
    assert(0);
  }

  fTransformMatrices = new TObjArray();
  fTransformMatrices->SetName(TString::Format("TransformMatrices_%s", cAnalysisBaseTags[fAnalysisType]));
  TString tTempTitle;
  for(unsigned int i=0; i<fTransformStorageMapping.size(); i++)
  {
    if(fTransformStorageMapping[i] == kResLamKSt0ToLamK0 || fTransformStorageMapping[i] == kResALamKSt0ToALamK0 || 
       fTransformStorageMapping[i] == kResSig0KSt0ToLamK0 || fTransformStorageMapping[i] == kResASig0KSt0ToALamK0 || 
       fTransformStorageMapping[i] == kResXi0KSt0ToLamK0 || fTransformStorageMapping[i] == kResAXi0KSt0ToALamK0 || 
       fTransformStorageMapping[i] == kResXiCKSt0ToLamK0 || fTransformStorageMapping[i] == kResAXiCKSt0ToALamK0)
    {
      tTempTitle = TString::Format("%sTransform", cAnalysisBaseTags[fTransformStorageMapping[i]]);
    }
    else tTempTitle = TString::Format("%sTo%sTransform", cAnalysisBaseTags[fTransformStorageMapping[i]], cAnalysisBaseTags[fAnalysisType]);
    fTransformMatrices->Add(new TH2D(tTempTitle, tTempTitle, fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax));

    cout << "Added transform with name " << tTempTitle << endl;
  }
  cout << "_______________________________________________________________________________________" << endl;
}



//________________________________________________________________________________________________________________
double ThermPairAnalysis::GetFatherKStar(ThermParticle &aParticle1, ThermParticle &aParticle2, bool aUseParticleFather1, bool aUseParticleFather2)
{
  double px1, py1, pz1, mass1, E1;
  double px2, py2, pz2, mass2, E2;

  if(aUseParticleFather1)
  {
    px1 = aParticle1.GetFatherPx();
    py1 = aParticle1.GetFatherPy();
    pz1 = aParticle1.GetFatherPz();
    mass1 = aParticle1.GetFatherMass();
    E1 = aParticle1.GetFatherE();
  }
  else
  {
    px1 = aParticle1.GetPx();
    py1 = aParticle1.GetPy();
    pz1 = aParticle1.GetPz();
    mass1 = aParticle1.GetMass();
    E1 = aParticle1.GetE();
  }

  if(aUseParticleFather2)
  {
    px2 = aParticle2.GetFatherPx();
    py2 = aParticle2.GetFatherPy();
    pz2 = aParticle2.GetFatherPz();
    mass2 = aParticle2.GetFatherMass();
    E2 = aParticle2.GetFatherE();
  }
  else
  {
    px2 = aParticle2.GetPx();
    py2 = aParticle2.GetPy();
    pz2 = aParticle2.GetPz();
    mass2 = aParticle2.GetMass();
    E2 = aParticle2.GetE();
  }

  double tMinvSq = (E1+E2)*(E1+E2) - (px1+px2)*(px1+px2) - (py1+py2)*(py1+py2) - (pz1+pz2)*(pz1+pz2);
  double tQinvSq = ((mass1*mass1 - mass2*mass2)*(mass1*mass1 - mass2*mass2))/tMinvSq + tMinvSq - 2.0*(mass1*mass1 + mass2*mass2);
  double tKStar = 0.5*sqrt(tQinvSq);

  return tKStar;
}

//________________________________________________________________________________________________________________
double ThermPairAnalysis::GetKStar(ThermParticle &aParticle1, ThermParticle &aParticle2)
{
  return GetFatherKStar(aParticle1, aParticle2, false, false);
}


//________________________________________________________________________________________________________________
void ThermPairAnalysis::FillTransformMatrixParticleV0(vector<ThermParticle> &aParticleCollection, vector<ThermV0Particle> &aV0Collection, ParticlePDGType aParticleFatherType, ParticlePDGType aV0FatherType, TH2* aMatrix)
{
  ThermParticle tParticle;
  ThermV0Particle tV0;
  double tKStar, tFatherKStar;

  bool bUseParticleFather=true;
  bool bUseV0Father = true;

  for(unsigned int iV0=0; iV0<aV0Collection.size(); iV0++)
  {
    tV0 = aV0Collection[iV0];
    if((tV0.GetFatherPID() == aV0FatherType || aV0FatherType == kPDGNull) && tV0.GoodV0())
    {
      if(aV0FatherType == kPDGNull) bUseV0Father = false;  //because, by setting aV0FatherType==kPDGNull, I am saying I don't care where the V0 comes from
                                                           //In which case, I am also saying to not use the V0Father
      if(tV0.GetPID() == aV0FatherType)  //here, I want only primary V0s, which, of course, cannot have a father
      {
        assert(tV0.IsPrimordial());  //the V0 should only be primordial if that's what we're looking for
        bUseV0Father = false;
      }
      for(unsigned int iPar=0; iPar<aParticleCollection.size(); iPar++)
      {
        tParticle = aParticleCollection[iPar];
        if(tParticle.GetFatherPID() == aParticleFatherType || aParticleFatherType == kPDGNull)
        {
          if(aParticleFatherType == kPDGNull) bUseParticleFather=false; //similar explanation as above for if(aV0FatherType == kPDGNull) bUseV0Father = false;
          if(tParticle.GetPID() == aParticleFatherType)  //similar explanation as above for if(tV0.GetPID() == aV0FatherType)
          {
            assert(tParticle.IsPrimordial());
            bUseParticleFather = false;
          }

          tKStar = GetKStar(tParticle,tV0);
          tFatherKStar = GetFatherKStar(tParticle,tV0,bUseParticleFather,bUseV0Father);

          assert(tV0.DoubleCheckV0Attributes());
          aMatrix->Fill(tKStar,tFatherKStar);
        }
      }
    }
  }
}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::FillTransformMatrixV0V0(vector<ThermV0Particle> &aV01Collection, vector<ThermV0Particle> &aV02Collection, ParticlePDGType aV01FatherType, ParticlePDGType aV02FatherType, TH2* aMatrix)
{
  ThermV0Particle tV01, tV02;
  double tKStar, tFatherKStar;

  bool bUseV01Father = true;
  bool bUseV02Father = true;

  for(unsigned int iV01=0; iV01<aV01Collection.size(); iV01++)
  {
    tV01 = aV01Collection[iV01];
    if((tV01.GetFatherPID() == aV01FatherType || aV01FatherType == kPDGNull) && tV01.GoodV0())
    {
      if(aV01FatherType == kPDGNull) bUseV01Father = false;  //because, by setting aV01FatherType==kPDGNull, I am saying I don't care where V01 comes from
                                                           //In which case, I am also saying to not use the V0Father
      if(tV01.GetPID() == aV01FatherType)  //here, I want only primary V0s, which, of course, cannot have a father
      {
        assert(tV01.IsPrimordial());  //the V0 should only be primordial if that's what we're looking for
        bUseV01Father = false;
      }
      for(unsigned int iV02=0; iV02<aV02Collection.size(); iV02++)
      {
        tV02 = aV02Collection[iV02];
        if((tV02.GetFatherPID() == aV02FatherType || aV02FatherType == kPDGNull) && tV02.GoodV0() 
           && !(tV02.GetEID()==tV01.GetEID() && tV02.GetEventID()==tV02.GetEventID()) ) //For instance, if I am doing LamLam w/o mixing events, I do not want to pair a Lam with itself
        {
          if(aV02FatherType == kPDGNull) bUseV02Father=false; //similar explanation as above for if(aV01FatherType == kPDGNull) bUseV01Father = false;
          if(tV02.GetPID() == aV02FatherType)  //similar explanation as above for if(tV01.GetPID() == aV01FatherType)
          {
            assert(tV02.IsPrimordial());
            bUseV02Father = false;
          }

          tKStar = GetKStar(tV01,tV02);
          tFatherKStar = GetFatherKStar(tV01,tV02,bUseV01Father,bUseV02Father);

          assert(tV01.DoubleCheckV0Attributes() && tV02.DoubleCheckV0Attributes());  
          aMatrix->Fill(tKStar,tFatherKStar);
        }
      }
    }
  }
}



//________________________________________________________________________________________________________________
void ThermPairAnalysis::BuildTransformMatrixParticleV0(ThermEvent aEvent, vector<ThermEvent> &aMixingEventsCollection, ParticlePDGType aParticleFatherType, ParticlePDGType aV0FatherType, TH2* aMatrix)
{
  ParticlePDGType tParticleType = fTransformInfo[0].particleType2;
  ParticlePDGType tV0Type       = fTransformInfo[0].particleType1;

  vector<ThermParticle> aParticleCollection;
  vector<ThermV0Particle> aV0Collection;


  aV0Collection =  aEvent.GetV0ParticleCollection(tV0Type);
  if(!fMixEvents)  //no mixing
  {
    aParticleCollection = aEvent.GetParticleCollection(tParticleType);
    FillTransformMatrixParticleV0(aParticleCollection,aV0Collection,aParticleFatherType,aV0FatherType,aMatrix);
  }
  else
  {
    for(unsigned int iMixEv=0; iMixEv < aMixingEventsCollection.size(); iMixEv++)
    {
      aParticleCollection = aMixingEventsCollection[iMixEv].GetParticleCollection(tParticleType);
      FillTransformMatrixParticleV0(aParticleCollection,aV0Collection,aParticleFatherType,aV0FatherType,aMatrix);
    }
  }
}



//________________________________________________________________________________________________________________
void ThermPairAnalysis::BuildTransformMatrixV0V0(ThermEvent aEvent, vector<ThermEvent> &aMixingEventsCollection, ParticlePDGType aV01FatherType, ParticlePDGType aV02FatherType, TH2* aMatrix)
{
  ParticlePDGType tV01Type = fTransformInfo[0].particleType1;
  ParticlePDGType tV02Type = fTransformInfo[0].particleType2;

  vector<ThermV0Particle> aV01Collection;
  vector<ThermV0Particle> aV02Collection;


  aV01Collection =  aEvent.GetV0ParticleCollection(tV01Type);
  if(!fMixEvents)  //no mixing
  {
    aV02Collection = aEvent.GetV0ParticleCollection(tV02Type);
    FillTransformMatrixV0V0(aV01Collection,aV02Collection,aV01FatherType,aV02FatherType,aMatrix);
  }
  else
  {
    for(unsigned int iMixEv=0; iMixEv < aMixingEventsCollection.size(); iMixEv++)
    {
      aV02Collection = aMixingEventsCollection[iMixEv].GetV0ParticleCollection(tV02Type);
      FillTransformMatrixV0V0(aV01Collection,aV02Collection,aV01FatherType,aV02FatherType,aMatrix);
    }
  }
}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::BuildAllTransformMatrices(ThermEvent aEvent, vector<ThermEvent> &aMixingEventsCollection)
{
  bool bIsV0V0 = false;
  if(fTransformInfo[0].particleType2==kPDGK0 || fTransformInfo[0].particleType2==kPDGLam || fTransformInfo[0].particleType2==kPDGALam) bIsV0V0=true;
  for(unsigned int i=0; i<fTransformInfo.size(); i++)
  {
    if(bIsV0V0) BuildTransformMatrixV0V0(aEvent, aMixingEventsCollection, fTransformInfo[i].parentType1, fTransformInfo[i].parentType2, (TH2D*)fTransformMatrices->At(i));
    else BuildTransformMatrixParticleV0(aEvent, aMixingEventsCollection, fTransformInfo[i].parentType2, fTransformInfo[i].parentType1, (TH2D*)fTransformMatrices->At(i));
  }
}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::SaveAllTransformMatrices(TFile *aFile)
{
  assert(aFile->IsOpen());
  for(int i=0; i<fTransformMatrices->GetEntries(); i++) ((TH2D*)fTransformMatrices->At(i))->Write();
}




//________________________________________________________________________________________________________________
void ThermPairAnalysis::MapAndFillParentsMatrixParticleV0(TH2* aMatrix, int aV0FatherType, int aTrackFatherType)
{
  //Note: List of parent PIDs found by turning on bBuildUniqueParents switch in ThermPairAnalysis::ExtractFromAllRootFiles
  int tBinV0Father=-1, tBinTrackFather=-1;
  for(unsigned int i=0; i<cAllLambdaFathers.size(); i++) if(aV0FatherType==cAllLambdaFathers[i]) tBinV0Father=i;
  for(unsigned int i=0; i<cAllKchFathers.size(); i++) if(aTrackFatherType==cAllKchFathers[i]) tBinTrackFather=i;

  if(!fBuildUniqueParents)  //If true, I am likely looking for new parents, so I don't want these asserts to be tripped
  {
    if(tBinV0Father==-1) cout << "FAILURE IMMINENT: aV0FatherType = " << aV0FatherType << endl;
    if(tBinTrackFather==-1) cout << "FAILURE IMMINENT: aTrackFatherType = " << aTrackFatherType << endl;
    assert(tBinV0Father>-1);
    assert(tBinTrackFather>-1);
  }
  aMatrix->Fill(tBinV0Father,tBinTrackFather);
}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::MapAndFillParentsMatrixV0V0(TH2* aMatrix, int aV01FatherType, int aV02FatherType)
{
  //Note: List of parent PIDs found by turning on bBuildUniqueParents switch in ThermPairAnalysis::ExtractFromAllRootFiles
  int tBinV01Father=-1, tBinV02Father=-1;
  for(unsigned int i=0; i<cAllLambdaFathers.size(); i++) if(aV01FatherType==cAllLambdaFathers[i]) tBinV01Father=i;
  for(unsigned int i=0; i<cAllK0ShortFathers.size(); i++) if(aV02FatherType==cAllK0ShortFathers[i]) tBinV02Father=i;

  if(!fBuildUniqueParents)  //If true, I am likely looking for new parents, so I don't want these asserts to be tripped
  {
    if(tBinV01Father==-1) cout << "FAILURE IMMINENT: aV01FatherType = " << aV01FatherType << endl;
    if(tBinV02Father==-1) cout << "FAILURE IMMINENT: aV02FatherType = " << aV02FatherType << endl;
    assert(tBinV01Father>-1);
    assert(tBinV02Father>-1);
  }
  aMatrix->Fill(tBinV01Father,tBinV02Father);
}


//________________________________________________________________________________________________________________
void ThermPairAnalysis::FillUniqueParents(vector<int> &aUniqueParents, int aFatherType)
{
  bool bParentAlreadyIncluded = false;
  for(unsigned int i=0; i<aUniqueParents.size(); i++)
  {
    if(aUniqueParents[i] == aFatherType) bParentAlreadyIncluded = true;
  }
  if(!bParentAlreadyIncluded) aUniqueParents.push_back(aFatherType);
}

//________________________________________________________________________________________________________________
vector<int> ThermPairAnalysis::UniqueCombineVectors(vector<int> &aVec1, vector<int> &aVec2)
{
  vector<int> tReturnVector = aVec1;
  bool bAlreadyIncluded = false;
  for(unsigned int i=0; i<aVec2.size(); i++)
  {
    bAlreadyIncluded = false;
    for(unsigned int j=0; j<tReturnVector.size(); j++)
    {
      if(tReturnVector[j] == aVec2[i]) bAlreadyIncluded = true;
    }
    if(!bAlreadyIncluded) tReturnVector.push_back(aVec2[i]);
  }
  return tReturnVector;
}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::PrintUniqueParents()
{
  std::sort(fUniqueParents1.begin(), fUniqueParents1.end());
  cout << "Unique parents of " << fTransformInfo[0].particleType1 << endl;
  cout << "\tN(parents) = " << fUniqueParents1.size() << endl;
  for(unsigned int i=0; i<fUniqueParents1.size()-1; i++) cout << fUniqueParents1[i] << ", ";
  cout << fUniqueParents1[fUniqueParents1.size()-1] << endl;
  cout << endl << endl << endl;

  std::sort(fUniqueParents2.begin(), fUniqueParents2.end());
  cout << "Unique parents of " << fTransformInfo[0].particleType2 << endl;
  cout << "\tN(parents) = " << fUniqueParents2.size() << endl;
  for(unsigned int i=0; i<fUniqueParents2.size()-1; i++) cout << fUniqueParents2[i] << ", ";
  cout << fUniqueParents2[fUniqueParents2.size()-1] << endl;
  cout << endl << endl << endl;
}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::FillPrimaryAndOtherPairInfo(int aParentType1, int aParentType2, double aMaxPrimaryDecayLength)
{
  bool bPairAlreadyIncluded = false;

  if(IncludeAsPrimary(aParentType1, aParentType2, aMaxPrimaryDecayLength))
  {
    for(unsigned int i=0; i<fPrimaryPairInfo.size(); i++)
    {
      if(fPrimaryPairInfo[i][0].pdgType == static_cast<ParticlePDGType>(aParentType1) &&
         fPrimaryPairInfo[i][1].pdgType == static_cast<ParticlePDGType>(aParentType2)) bPairAlreadyIncluded = true;
    }
    if(!bPairAlreadyIncluded) fPrimaryPairInfo.push_back(vector<PidInfo>{GetParticlePidInfo(aParentType1),GetParticlePidInfo(aParentType2)});
  }

  //--------------------
  if(IncludeInOthers(aParentType1, aParentType2, aMaxPrimaryDecayLength))
  {
    bPairAlreadyIncluded = false;
    for(unsigned int i=0; i<fOtherPairInfo.size(); i++)
    {
      if(fOtherPairInfo[i][0].pdgType == static_cast<ParticlePDGType>(aParentType1) &&
         fOtherPairInfo[i][1].pdgType == static_cast<ParticlePDGType>(aParentType2)) bPairAlreadyIncluded = true;
    }
    if(!bPairAlreadyIncluded) fOtherPairInfo.push_back(vector<PidInfo>{GetParticlePidInfo(aParentType1),GetParticlePidInfo(aParentType2)});
  }
}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::PrintPrimaryAndOtherPairInfo()
{
  cout << "----------------------------------------- " << cAnalysisBaseTags[fAnalysisType] << " -----------------------------------------" << endl;

  cout << "---------- fPrimaryPairInfo ----------" << endl;
  cout << "\tfPrimaryPairInfo.size() = " << fPrimaryPairInfo.size() << endl;
  for(unsigned int i=0; i<fPrimaryPairInfo.size(); i++)
  {
    cout << "PID1, PID2   = " << fPrimaryPairInfo[i][0].pdgType << ", " << fPrimaryPairInfo[i][1].pdgType << endl;
    cout << "Name1, Name2 = " << fPrimaryPairInfo[i][0].name << ", " << fPrimaryPairInfo[i][1].name << endl << endl;
  }

  cout << "---------- fOtherPairInfo ----------" << endl;
  cout << "\fOtherPairInfo.size() = " << fOtherPairInfo.size() << endl;
  for(unsigned int i=0; i<fOtherPairInfo.size(); i++)
  {
    cout << "PID1, PID2   = " << fOtherPairInfo[i][0].pdgType << ", " << fOtherPairInfo[i][1].pdgType << endl;
    cout << "Name1, Name2 = " << fOtherPairInfo[i][0].name << ", " << fOtherPairInfo[i][1].name << endl << endl;
  }
  
}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::MapAndFillPairFractionHistogramParticleV0(TH1* aHistogram, int aV0FatherType, int aTrackFatherType, double aMaxPrimaryDecayLength, double tWeight)
{
  double tBin = -1.;
  if((aV0FatherType == kPDGLam || aV0FatherType == kPDGALam) && (aTrackFatherType == kPDGKchP || aTrackFatherType == kPDGKchM)) tBin = 0.;
  else if((aV0FatherType==kPDGSigma || aV0FatherType==kPDGASigma) && (aTrackFatherType == kPDGKchP || aTrackFatherType == kPDGKchM)) tBin = 1.;
  else if((aV0FatherType==kPDGXi0 || aV0FatherType==kPDGAXi0) && (aTrackFatherType == kPDGKchP || aTrackFatherType == kPDGKchM)) tBin = 2.;
  else if((aV0FatherType==kPDGXiC || aV0FatherType==kPDGAXiC) && (aTrackFatherType == kPDGKchP || aTrackFatherType == kPDGKchM)) tBin = 3.;
  else if((aV0FatherType==kPDGSigStP || aV0FatherType==kPDGASigStM) && (aTrackFatherType == kPDGKchP || aTrackFatherType == kPDGKchM)) tBin = 4.;
  else if((aV0FatherType==kPDGSigStM || aV0FatherType==kPDGASigStP) && (aTrackFatherType == kPDGKchP || aTrackFatherType == kPDGKchM)) tBin = 5.;
  else if((aV0FatherType==kPDGSigSt0 || aV0FatherType==kPDGASigSt0) && (aTrackFatherType == kPDGKchP || aTrackFatherType == kPDGKchM)) tBin = 6.;

  else if((aV0FatherType == kPDGLam || aV0FatherType == kPDGALam) && (aTrackFatherType == kPDGKSt0 || aTrackFatherType == kPDGAKSt0)) tBin = 7.;
  else if((aV0FatherType==kPDGSigma || aV0FatherType==kPDGASigma) && (aTrackFatherType == kPDGKSt0 || aTrackFatherType == kPDGAKSt0)) tBin = 8.;
  else if((aV0FatherType==kPDGXi0 || aV0FatherType==kPDGAXi0) && (aTrackFatherType == kPDGKSt0 || aTrackFatherType == kPDGAKSt0)) tBin = 9.;
  else if((aV0FatherType==kPDGXiC || aV0FatherType==kPDGAXiC) && (aTrackFatherType == kPDGKSt0 || aTrackFatherType == kPDGAKSt0)) tBin = 10.;
  else if(IncludeAsPrimary(aV0FatherType, aTrackFatherType, aMaxPrimaryDecayLength)) tBin=0.;
  else {assert(IncludeInOthers(aV0FatherType, aTrackFatherType, aMaxPrimaryDecayLength)); tBin = 11.;}

  if(tBin > -1)
  {
    tBin += 0.1;
    aHistogram->Fill(tBin, tWeight);
  }

}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::MapAndFillPairFractionHistogramV0V0(TH1* aHistogram, int aV01FatherType, int aV02FatherType, double aMaxPrimaryDecayLength, double tWeight)
{
  double tBin = -1.;
  if((aV01FatherType == kPDGLam || aV01FatherType == kPDGALam) && (aV02FatherType == kPDGK0)) tBin = 0.;
  else if((aV01FatherType==kPDGSigma || aV01FatherType==kPDGASigma) && (aV02FatherType == kPDGK0)) tBin = 1.;
  else if((aV01FatherType==kPDGXi0 || aV01FatherType==kPDGAXi0) && (aV02FatherType == kPDGK0)) tBin = 2.;
  else if((aV01FatherType==kPDGXiC || aV01FatherType==kPDGAXiC) && (aV02FatherType == kPDGK0)) tBin = 3.;
  else if((aV01FatherType==kPDGSigStP || aV01FatherType==kPDGASigStM) && (aV02FatherType == kPDGK0)) tBin = 4.;
  else if((aV01FatherType==kPDGSigStM || aV01FatherType==kPDGASigStP) && (aV02FatherType == kPDGK0)) tBin = 5.;
  else if((aV01FatherType==kPDGSigSt0 || aV01FatherType==kPDGASigSt0) && (aV02FatherType == kPDGK0)) tBin = 6.;

  else if((aV01FatherType == kPDGLam || aV01FatherType == kPDGALam) && (aV02FatherType == kPDGKSt0 || aV02FatherType == kPDGAKSt0)) tBin = 7.;
  else if((aV01FatherType==kPDGSigma || aV01FatherType==kPDGASigma) && (aV02FatherType == kPDGKSt0 || aV02FatherType == kPDGAKSt0)) tBin = 8.;
  else if((aV01FatherType==kPDGXi0 || aV01FatherType==kPDGAXi0) && (aV02FatherType == kPDGKSt0 || aV02FatherType == kPDGAKSt0)) tBin = 9.;
  else if((aV01FatherType==kPDGXiC || aV01FatherType==kPDGAXiC) && (aV02FatherType == kPDGKSt0 || aV02FatherType == kPDGAKSt0)) tBin = 10.;
  else if(IncludeAsPrimary(aV01FatherType, aV02FatherType, aMaxPrimaryDecayLength)) tBin=0.;
  else {assert(IncludeInOthers(aV01FatherType,aV02FatherType, aMaxPrimaryDecayLength)); tBin = 11.;}

  if(tBin > -1)
  {
    tBin += 0.1;
    aHistogram->Fill(tBin, tWeight);
  }
}



//________________________________________________________________________________________________________________
void ThermPairAnalysis::BuildPairFractionHistogramsParticleV0(ThermEvent aEvent, double aMaxPrimaryDecayLength)
{
  ParticlePDGType tParticleType = fTransformInfo[0].particleType2;
  ParticlePDGType tV0Type       = fTransformInfo[0].particleType1;

  vector<ThermParticle> aParticleCollection;
  vector<ThermV0Particle> aV0Collection;

  aV0Collection =  aEvent.GetV0ParticleCollection(tV0Type);
  aParticleCollection = aEvent.GetParticleCollection(tParticleType);

  ThermV0Particle tV0;

  for(unsigned int iV0=0; iV0<aV0Collection.size(); iV0++)
  {
    tV0 = aV0Collection[iV0];
    int tV0FatherType = tV0.GetFatherPID();

    if(tV0.GoodV0())
    {
      ThermParticle tParticle;
      for(unsigned int iPar=0; iPar<aParticleCollection.size(); iPar++)
      {
        tParticle = aParticleCollection[iPar];
        int tParticleFatherType = tParticle.GetFatherPID();

        MapAndFillPairFractionHistogramParticleV0(fPairFractions, tV0FatherType, tParticleFatherType, aMaxPrimaryDecayLength);
        FillPrimaryAndOtherPairInfo(tV0FatherType, tParticleFatherType, aMaxPrimaryDecayLength);
        if(fBuildUniqueParents)
        {
          FillUniqueParents(fUniqueParents2, tParticleFatherType);
          FillUniqueParents(fUniqueParents1, tV0FatherType);
        }
        MapAndFillParentsMatrixParticleV0(fParentsMatrix, tV0FatherType, tParticleFatherType);
      }
    }
  }
}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::BuildPairFractionHistogramsV0V0(ThermEvent aEvent, double aMaxPrimaryDecayLength)
{
  ParticlePDGType tV01Type = fTransformInfo[0].particleType1;
  ParticlePDGType tV02Type = fTransformInfo[0].particleType2;

  vector<ThermV0Particle> aV01Collection, aV02Collection;


  aV01Collection =  aEvent.GetV0ParticleCollection(tV01Type);
  aV02Collection =  aEvent.GetV0ParticleCollection(tV02Type);

  ThermV0Particle tV01;

  for(unsigned int iV01=0; iV01<aV01Collection.size(); iV01++)
  {
    tV01 = aV01Collection[iV01];
    int tV01FatherType = tV01.GetFatherPID();

    if(tV01.GoodV0())
    {
      ThermV0Particle tV02;
      for(unsigned int iV02=0; iV02<aV02Collection.size(); iV02++)
      {
        tV02 = aV02Collection[iV02];
        int tV02FatherType = tV02.GetFatherPID();

        if(tV02.GoodV0())
        {
          MapAndFillPairFractionHistogramV0V0(fPairFractions, tV01FatherType, tV02FatherType, aMaxPrimaryDecayLength);
          FillPrimaryAndOtherPairInfo(tV01FatherType, tV02FatherType, aMaxPrimaryDecayLength);
          if(fBuildUniqueParents)
          {
            FillUniqueParents(fUniqueParents1, tV01FatherType);
            FillUniqueParents(fUniqueParents2, tV02FatherType);
          }
          MapAndFillParentsMatrixV0V0(fParentsMatrix, tV01FatherType, tV02FatherType);
        }
      }
    }
  }
}


//________________________________________________________________________________________________________________
void ThermPairAnalysis::SavePairFractionsAndParentsMatrix(TFile *aFile)
{
  assert(aFile->IsOpen());
  
  fPairFractions->Write();
  fParentsMatrix->Write();
}









