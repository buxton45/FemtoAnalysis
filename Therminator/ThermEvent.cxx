///////////////////////////////////////////////////////////////////////////
// ThermEvent:                                                           //
///////////////////////////////////////////////////////////////////////////


#include "ThermEvent.h"

#ifdef __ROOT__
ClassImp(ThermEvent)
#endif


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________

//________________________________________________________________________________________________________________
ThermEvent::ThermEvent() :
  fGenerator(std::chrono::system_clock::now().time_since_epoch().count()),
  fEventID(0),
  fAllParticlesCollection(0),
  fAllDaughtersCollection(0),

  fLambdaCollection(0),
  fAntiLambdaCollection(0),
  fK0ShortCollection(0),

  fKchPCollection(0),
  fKchMCollection(0),

  fProtCollection(0),
  fAProtCollection(0)

{

}

//________________________________________________________________________________________________________________
ThermEvent::ThermEvent(TTree* aThermEventsTree, int aEntryNumber) :
  fGenerator(std::chrono::system_clock::now().time_since_epoch().count()),
  fEventID(0),
  fAllParticlesCollection(0),
  fAllDaughtersCollection(0),

  fLambdaCollection(0),
  fAntiLambdaCollection(0),
  fK0ShortCollection(0),

  fKchPCollection(0),
  fKchMCollection(0),

  fProtCollection(0),
  fAProtCollection(0)

{

}


//________________________________________________________________________________________________________________
ThermEvent::ThermEvent(const ThermEvent& aEvent) :
  fEventID(aEvent.fEventID),
  fAllParticlesCollection(aEvent.fAllParticlesCollection),
  fAllDaughtersCollection(aEvent.fAllDaughtersCollection),
  fLambdaCollection(aEvent.fLambdaCollection),
  fAntiLambdaCollection(aEvent.fAntiLambdaCollection),
  fK0ShortCollection(aEvent.fK0ShortCollection),
  fKchPCollection(aEvent.fKchPCollection),
  fKchMCollection(aEvent.fKchMCollection),
  fProtCollection(aEvent.fProtCollection),
  fAProtCollection(aEvent.fAProtCollection)
{

}

//________________________________________________________________________________________________________________
ThermEvent& ThermEvent::operator=(const ThermEvent& aEvent)
{
  if(this == &aEvent) return *this;

  fEventID = aEvent.fEventID;

  fAllParticlesCollection = aEvent.fAllParticlesCollection;
  fAllDaughtersCollection = aEvent.fAllDaughtersCollection;
  fLambdaCollection = aEvent.fLambdaCollection;
  fAntiLambdaCollection = aEvent.fAntiLambdaCollection;
  fK0ShortCollection = aEvent.fK0ShortCollection;
  fKchPCollection = aEvent.fKchPCollection;
  fKchMCollection = aEvent.fKchMCollection;
  fProtCollection = aEvent.fProtCollection;
  fAProtCollection = aEvent.fAProtCollection;

  return *this;
}


//________________________________________________________________________________________________________________
ThermEvent* ThermEvent::clone()
{
  return(new ThermEvent(*this));
}


//________________________________________________________________________________________________________________
ThermEvent::~ThermEvent()
{
  /* no-op */
}


//________________________________________________________________________________________________________________
bool ThermEvent::IsParticleOfInterest(ParticleCoor* aParticle)
{
  int tPID = aParticle->pid;

  if(tPID == 3122) return true;        //Lam
  else if(tPID == -3122) return true;  //ALam

  else if(tPID == 311) return true;    //K0 (K0s = 310, but apparently there are no K0s)

  else if(tPID == 321) return true;    //KchP
  else if(tPID == -321) return true;   //KchM 

  else if(tPID == 2212) return true;    //Proton
  else if(tPID == -2212) return true;   //AntiProton 

  else return false;
}

//________________________________________________________________________________________________________________
bool ThermEvent::IsDaughterOfInterest(ThermParticle &aDaughterParticle)
{
  int tFatherPID = aDaughterParticle.GetFatherPID();
 
  if(aDaughterParticle.IsPrimordial()) return false;  //this is a primordial particle with no father!
  else if(tFatherPID == kPDGLam) return true;
  else if(tFatherPID == kPDGALam) return true;
  else if(tFatherPID == kPDGK0) return true;
  else return false;
}

//________________________________________________________________________________________________________________
bool ThermEvent::IsDaughterOfInterest(ParticleCoor* aDaughterParticle)
{
  int tFatherPID = aDaughterParticle->fatherpid;
  int tFatherEID = aDaughterParticle->fathereid;

  if(tFatherEID == -1) return false;  //this is a primordial particle with no father!
  else if(tFatherPID == kPDGLam) return true;
  else if(tFatherPID == kPDGALam) return true;
  else if(tFatherPID == kPDGK0) return true;
  else return false;
}

//TODO PushBackThermParticle and PushBackThermParticleOfInterest should be able to be combined by including
//  a bool = aIsOfInterest into the arugments 
//________________________________________________________________________________________________________________
void ThermEvent::PushBackThermParticle(ParticleCoor *aParticle)
{
  fAllParticlesCollection.emplace_back(aParticle);
}


//________________________________________________________________________________________________________________
void ThermEvent::PushBackThermDaughterOfInterest(ParticleCoor *aParticle)
{
  fAllDaughtersCollection.emplace_back(aParticle);
}


//________________________________________________________________________________________________________________
//TODO
void ThermEvent::PushBackThermParticleOfInterest(ParticleCoor* aParticle)
{
  if(aParticle->pid == kPDGLam) fLambdaCollection.emplace_back(aParticle);
  else if(aParticle->pid == kPDGALam) fAntiLambdaCollection.emplace_back(aParticle);
  else if(aParticle->pid == kPDGK0) fK0ShortCollection.emplace_back(aParticle);

  else if(aParticle->pid == kPDGKchP) fKchPCollection.emplace_back(aParticle);
  else if(aParticle->pid == kPDGKchM) fKchMCollection.emplace_back(aParticle);

  else if(aParticle->pid == kPDGProt) fProtCollection.emplace_back(aParticle);
  else if(aParticle->pid == kPDGAntiProt) fAProtCollection.emplace_back(aParticle);

  else
  {
    cout << "Particle of wrong type trying to be added collection via ThermEvent::PushBackThermParticleOfInterest" << endl;
    cout << "PREPARE FOR CRASH" << endl;
    assert(0);
  }

}


//________________________________________________________________________________________________________________
void ThermEvent::ClearCollection(vector<ThermParticle> &aCollection)
{
  aCollection.clear();
  aCollection.shrink_to_fit();
}

//________________________________________________________________________________________________________________
void ThermEvent::ClearCollection(vector<ThermV0Particle> &aCollection)
{
  aCollection.clear();
  aCollection.shrink_to_fit();
}

//________________________________________________________________________________________________________________
void ThermEvent::ClearThermEvent()
{
  ClearCollection(fAllParticlesCollection);
  ClearCollection(fAllDaughtersCollection);

  ClearCollection(fLambdaCollection);
  ClearCollection(fAntiLambdaCollection);
  ClearCollection(fK0ShortCollection);

  ClearCollection(fKchPCollection);
  ClearCollection(fKchMCollection);

  ClearCollection(fProtCollection);
  ClearCollection(fAProtCollection);
}

//________________________________________________________________________________________________________________
void ThermEvent::AssertAllLambdaFathersFoundDaughters()
{
//  for(unsigned int i=0; i<fLambdaCollection.size(); i++) assert(fLambdaCollection[i].BothDaughtersFound());
//  for(unsigned int i=0; i<fAntiLambdaCollection.size(); i++) assert(fAntiLambdaCollection[i].BothDaughtersFound());

  for(unsigned int i=0; i<fLambdaCollection.size();)
  {
    if(!fLambdaCollection[i].BothDaughtersFound()) 
    {
      cout << "WARNING: !fLambdaCollection[" << i << "].BothDaughtersFound()" << endl;
      cout << "\t Deleting element..." << endl << endl;
      fLambdaCollection.erase(fLambdaCollection.begin()+i);
    }
    else i++;
  }
  for(unsigned int i=0; i<fLambdaCollection.size(); i++) assert(fLambdaCollection[i].BothDaughtersFound());
  //----------------------------------------------------
  for(unsigned int i=0; i<fAntiLambdaCollection.size();)
  {
    if(!fAntiLambdaCollection[i].BothDaughtersFound()) 
    {
      cout << "WARNING: !fAntiLambdaCollection[" << i << "].BothDaughtersFound()" << endl;
      cout << "\t Deleting element..." << endl << endl;
      fAntiLambdaCollection.erase(fAntiLambdaCollection.begin()+i);
    }
    else i++;
  }
  for(unsigned int i=0; i<fAntiLambdaCollection.size(); i++) assert(fAntiLambdaCollection[i].BothDaughtersFound());

}


//________________________________________________________________________________________________________________
void ThermEvent::AssertAllK0FathersFound0or2Daughters()
{
  for(unsigned int i=0; i<fK0ShortCollection.size(); i++)
  {
    assert(fK0ShortCollection[i].BothDaughtersFound() || (!fK0ShortCollection[i].Daughter1Found() && !fK0ShortCollection[i].Daughter2Found()));
  }
}


//________________________________________________________________________________________________________________
//TODO this and MatchDaughtersWithFathers are inefficient!
//TODO Figure out how to deep copy so this method works!
void ThermEvent::FindFatherandLoadDaughter(ThermParticle &aDaughterParticle)
{
  int tFatherPID = aDaughterParticle.GetFatherPID();
  int tFatherEID = aDaughterParticle.GetFatherEID();

  vector<ThermV0Particle> tFatherCollection; //just a copy, when I load, I must use the actual object

  if(tFatherPID == kPDGLam) tFatherCollection = fLambdaCollection;
  else if(tFatherPID == kPDGALam) tFatherCollection = fAntiLambdaCollection;
  else if(tFatherPID == kPDGK0) tFatherCollection = fK0ShortCollection;
  else assert(0);
  //---------------------------------

  int tFatherLocation = -1;
  for(unsigned int i=0; i<tFatherCollection.size(); i++)
  {
    if(tFatherCollection[i].GetEID() == tFatherEID)
    {
      tFatherLocation = i;
      break;
    }
  }
  assert(tFatherLocation >= 0);
  //---------------------------------
  if(tFatherPID == kPDGLam) fLambdaCollection[tFatherLocation].LoadDaughter(aDaughterParticle);
  else if(tFatherPID == kPDGALam) fAntiLambdaCollection[tFatherLocation].LoadDaughter(aDaughterParticle);
  else if(tFatherPID == kPDGK0) fK0ShortCollection[tFatherLocation].LoadDaughter(aDaughterParticle);
}

//________________________________________________________________________________________________________________
//TODO this and FindFatherAndLoadDaughter are inefficient!
void ThermEvent::MatchDaughtersWithFathers()
{
  for(unsigned int i=0; i<fAllDaughtersCollection.size(); i++)
  {
    if(IsDaughterOfInterest(fAllDaughtersCollection[i]))
    {
      FindFatherandLoadDaughter(fAllDaughtersCollection[i]);
    }
  }

  AssertAllLambdaFathersFoundDaughters();
  AssertAllK0FathersFound0or2Daughters();

  //No longer need fAllDaughtersCollection, so I should clear it and free up memory before it is pushed to ThermEventsCollection
  ClearCollection(fAllDaughtersCollection);
}

//________________________________________________________________________________________________________________
void ThermEvent::FindFather(ThermParticle &aParticle)
{
  if(!aParticle.IsPrimordial())
  {
    int tFatherEID = aParticle.GetFatherEID();

    int tFatherLocation = -1;
    for(unsigned int i=0; i<fAllParticlesCollection.size(); i++)
    {
      if(fAllParticlesCollection[i].GetEID() == tFatherEID)
      {
        tFatherLocation = i;
        aParticle.LoadFather(fAllParticlesCollection[i]);
        break;
      }
    }
    assert(tFatherLocation >= 0);
  }
}


//________________________________________________________________________________________________________________
void ThermEvent::FindAllFathers()
{
  for(unsigned int i=0; i<fLambdaCollection.size(); i++) FindFather(fLambdaCollection[i]);
  for(unsigned int i=0; i<fAntiLambdaCollection.size(); i++) FindFather(fAntiLambdaCollection[i]);
  for(unsigned int i=0; i<fK0ShortCollection.size(); i++) FindFather(fK0ShortCollection[i]);

  for(unsigned int i=0; i<fKchPCollection.size(); i++) FindFather(fKchPCollection[i]);
  for(unsigned int i=0; i<fKchMCollection.size(); i++) FindFather(fKchMCollection[i]);

  for(unsigned int i=0; i<fProtCollection.size(); i++) FindFather(fProtCollection[i]);
  for(unsigned int i=0; i<fAProtCollection.size(); i++) FindFather(fAProtCollection[i]);

  //No longer need fAllParticlesCollection, so I should clear it and free up memory before it is pushed to ThermEventsCollection
  //TODO this is now still needed for FlowAnalysis objects, so do not clear it yet
  //  For the most central of events, this adds ~0.5 GB to the memory consumption (5.0 GB -> 5.5 GB)
  //ClearCollection(fAllParticlesCollection);
}

//________________________________________________________________________________________________________________
void ThermEvent::EnforceKinematicCuts()
{
  //Note: This is already done for V0 collections in ThermEvent::FindFatherandLoadDaughter, by way of 
  //      ThermV0Particle::LoadDaughter (which uses ThermV0Particle::PassV0Cuts, ThermV0Particle::PassDaughterCuts)

  //Note:  //TODO? Could impose eta cut at IsParticleOfInterest and IsDaughterOfInterest level
  //       Only issue is this would also remove any fathers which are outside of the eta range, so this issue would
  //       need to be resolved first
  for(unsigned int i=0; i<fKchPCollection.size();)
  {
    if(!fKchPCollection[i].PassKinematicCuts()) fKchPCollection.erase(fKchPCollection.begin()+i);
    else i++;
  }
  for(unsigned int i=0; i<fKchMCollection.size();)
  {
    if(!fKchMCollection[i].PassKinematicCuts()) fKchMCollection.erase(fKchMCollection.begin()+i);
    else i++;
  }
  //---------------
  for(unsigned int i=0; i<fProtCollection.size();)
  {
    if(!fProtCollection[i].PassKinematicCuts()) fProtCollection.erase(fProtCollection.begin()+i);
    else i++;
  }
  for(unsigned int i=0; i<fAProtCollection.size();)
  {
    if(!fAProtCollection[i].PassKinematicCuts()) fAProtCollection.erase(fAProtCollection.begin()+i);
    else i++;
  }

}

//________________________________________________________________________________________________________________
vector<ThermV0Particle> ThermEvent::GetV0ParticleCollection(ParticlePDGType aPDGType) const
{
  if(aPDGType == kPDGLam) return fLambdaCollection;
  else if(aPDGType == kPDGALam) return fAntiLambdaCollection;
  else if(aPDGType == kPDGK0) return fK0ShortCollection;

  else assert(0);
}


//________________________________________________________________________________________________________________
vector<ThermParticle> ThermEvent::GetParticleCollection(ParticlePDGType aPDGType) const
{

  if(aPDGType == kPDGKchP) return fKchPCollection;
  else if(aPDGType == kPDGKchM) return fKchMCollection;
  else if(aPDGType == kPDGProt) return fProtCollection;
  else if(aPDGType == kPDGAntiProt) return fAProtCollection;

  else assert(0);
}

//________________________________________________________________________________________________________________
vector<ThermParticle> ThermEvent::GetGoodParticleCollectionCastAsThermParticle(ParticlePDGType aPDGType) const
{
  bool tV0Coll = false;
  vector<ThermParticle> tThermPartColl;
  vector<ThermV0Particle> tThermV0Coll;

  if(aPDGType == kPDGKchP || aPDGType == kPDGKchM || 
     aPDGType == kPDGProt || aPDGType == kPDGAntiProt) 
  {
    tThermPartColl = GetParticleCollection(aPDGType);
    tV0Coll = false;
  }
  else if(aPDGType == kPDGLam || aPDGType == kPDGALam || aPDGType == kPDGK0) 
  {
    tThermV0Coll = GetV0ParticleCollection(aPDGType);
    tV0Coll = true;
  }
  else assert(0);

  //---------------------

  vector<ThermParticle> tReturnColl(0);
  if(tV0Coll)
  {
    for(unsigned int i=0; i<tThermV0Coll.size(); i++) if(tThermV0Coll[i].GoodV0()) tReturnColl.push_back(tThermV0Coll[i]);
  }
  else
  {
    for(unsigned int i=0; i<tThermPartColl.size(); i++) tReturnColl.push_back(tThermPartColl[i]);
  }

  return tReturnColl;
}

//________________________________________________________________________________________________________________
vector<ThermParticle> ThermEvent::GetGoodParticleCollectionCastAsThermParticle(int aPID) const
{
  ParticlePDGType tPDGType = static_cast<ParticlePDGType>(aPID);
  return GetGoodParticleCollectionCastAsThermParticle(tPDGType);
}

//________________________________________________________________________________________________________________
vector<ThermParticle> ThermEvent::GetGoodParticleCollectionwConjCastAsThermParticle(int aPID) const
{
  if(aPID==311) return GetGoodParticleCollectionCastAsThermParticle(aPID);  //K0Short is its own anti-particle

  vector<ThermParticle> tColl1 = GetGoodParticleCollectionCastAsThermParticle(aPID);
  vector<ThermParticle> tColl2 = GetGoodParticleCollectionCastAsThermParticle(-1*aPID);

  vector<ThermParticle> tReturnColl(0);
  for(unsigned int i=0; i<tColl1.size(); i++) tReturnColl.push_back(tColl1[i]);
  for(unsigned int i=0; i<tColl2.size(); i++) tReturnColl.push_back(tColl2[i]);

  return tReturnColl;
}


//________________________________________________________________________________________________________________
void ThermEvent::SetV0ParticleCollection(unsigned int aEventID, ParticlePDGType aPDGType, vector<ThermV0Particle> &aCollection)
{
  assert(aEventID == fEventID);  //make sure I'm setting the collection for the correct event

  if(aPDGType == kPDGLam) fLambdaCollection = aCollection;
  else if(aPDGType == kPDGALam) fAntiLambdaCollection = aCollection;
  else if(aPDGType == kPDGK0) fK0ShortCollection = aCollection;

  else assert(0);
}


//________________________________________________________________________________________________________________
void ThermEvent::SetParticleCollection(unsigned int aEventID, ParticlePDGType aPDGType, vector<ThermParticle> &aCollection)
{
  assert(aEventID == fEventID);  //make sure I'm setting the collection for the correct event

  if(aPDGType == kPDGKchP) fKchPCollection = aCollection;
  else if(aPDGType == kPDGKchM) fKchMCollection = aCollection;
  else if(aPDGType == kPDGProt) fProtCollection = aCollection;
  else if(aPDGType == kPDGAntiProt) fAProtCollection = aCollection;

  else assert(0);
}

//________________________________________________________________________________________________________________
void ThermEvent::CheckCoECoM()
{
  if(fAllParticlesCollection.size()==0) cout << "Must call CheckCoECoM BEFORE FindAllFathers, fAllParticlesCollection.size()==0, crash imminent!" << endl;
  assert(fAllParticlesCollection.size());

  TLorentzVector tTotMom4Vec = TLorentzVector(0., 0., 0., 0.);
  for(unsigned int i=0; i<fAllParticlesCollection.size(); i++) tTotMom4Vec += fAllParticlesCollection[i].GetFourMomentum();

  cout << "_____________________________________" << endl;
  cout << "Total 4-momentum calculated:" << endl;
  cout << "PTotX = " << tTotMom4Vec.X() << endl;
  cout << "PTotY = " << tTotMom4Vec.Y() << endl;
  cout << "PTotZ = " << tTotMom4Vec.Z() << endl;
  cout << "----------" << endl;
  cout << "PTotMag = " << tTotMom4Vec.P() << endl;
  cout << "ETot = " << tTotMom4Vec.T() << endl;
  cout << "_____________________________________" << endl << endl;
}


//________________________________________________________________________________________________________________
double ThermEvent::CalculateEventPlane(vector<ThermParticle> &aCollection)
{
  unsigned int tMult = aCollection.size();
  complex<double> tImI (0., 1.);
  complex<double> tQn(0., 0.);

  for(unsigned int iPart=0; iPart<tMult; iPart++) tQn = tQn + exp(tImI*2.0*aCollection[iPart].GetPhiP());
  double tReturnEP = atan2(imag(tQn), real(tQn))/2.0;
  return tReturnEP;
}

//________________________________________________________________________________________________________________
double ThermEvent::CalculateEventPlane(vector<ThermV0Particle> &aCollection)
{
  unsigned int tMult = aCollection.size();
  complex<double> tImI (0., 1.);
  complex<double> tQn(0., 0.);

  for(unsigned int iPart=0; iPart<tMult; iPart++) tQn = tQn + exp(tImI*2.0*aCollection[iPart].GetPhiP());
  double tReturnEP = atan2(imag(tQn), real(tQn))/2.0;
  return tReturnEP;
}


//________________________________________________________________________________________________________________
void ThermEvent::RotateParticlesByRandomAzimuthalAngle(double aPhi, vector<ThermParticle> &aCollection, bool aOutputEP)
{
  double tEventPlane=0.;
  double tExpectedEP = 0.;
  if(aOutputEP)
  {
    tEventPlane = CalculateEventPlane(aCollection);
    cout << TString::Format("EP before rotation = %0.4f (%0.2f deg.)", tEventPlane, (180./TMath::Pi())*tEventPlane) << endl;
  }

  for(unsigned int iPart=0; iPart<aCollection.size(); iPart++) aCollection[iPart].TransformRotateZ(-aPhi);  //Negative sign because
                                                                                                            //want CCW rotation
  if(aOutputEP)
  {
    tExpectedEP = aPhi+tEventPlane;
    tExpectedEP = atan(tan(tExpectedEP));
    cout << TString::Format("\t aPhi for rotation = %0.4f (%0.2f deg.)", aPhi, (180./TMath::Pi())*aPhi) << endl;
    cout << TString::Format("\t aPhi + EP         = %0.4f (%0.2f deg.)", tExpectedEP, (180./TMath::Pi())*tExpectedEP) << endl;
    tEventPlane = CalculateEventPlane(aCollection);
    cout << TString::Format("EP after rotation  = %0.4f (%0.2f deg.)", tEventPlane, (180./TMath::Pi())*tEventPlane) << endl;
    cout << TString::Format("\t Diff = %0.4f", tEventPlane-tExpectedEP) << endl << endl;
    assert(abs(tEventPlane-tExpectedEP) < 0.0001);
  }
}

//________________________________________________________________________________________________________________
void ThermEvent::RotateParticlesByRandomAzimuthalAngle(double aPhi, vector<ThermV0Particle> &aCollection, bool aOutputEP)
{
  double tEventPlane=0.;
  double tExpectedEP = 0.;
  if(aOutputEP)
  {
    tEventPlane = CalculateEventPlane(aCollection);
    cout << TString::Format("EP before rotation = %0.4f (%0.2f deg.)", tEventPlane, (180./TMath::Pi())*tEventPlane) << endl;
  }

  for(unsigned int iPart=0; iPart<aCollection.size(); iPart++) aCollection[iPart].TransformRotateZ(-aPhi);  //Negative sign because
                                                                                                            //want CCW rotation
  if(aOutputEP)
  {
    tExpectedEP = aPhi+tEventPlane;
    tExpectedEP = atan(tan(tExpectedEP));
    cout << TString::Format("\t aPhi for rotation = %0.4f (%0.2f deg.)", aPhi, (180./TMath::Pi())*aPhi) << endl;
    cout << TString::Format("\t aPhi + EP         = %0.4f (%0.2f deg.)", tExpectedEP, (180./TMath::Pi())*tExpectedEP) << endl;
    tEventPlane = CalculateEventPlane(aCollection);
    cout << TString::Format("EP after rotation  = %0.4f (%0.2f deg.)", tEventPlane, (180./TMath::Pi())*tEventPlane) << endl;
    cout << TString::Format("\t Diff = %0.4f", tEventPlane-tExpectedEP) << endl << endl;
    assert(abs(tEventPlane-tExpectedEP) < 0.0001);
  }
}

//________________________________________________________________________________________________________________
void ThermEvent::RotateAllParticlesByRandomAzimuthalAngle(bool aOutputEP)
{
//  std::default_random_engine tGenerator (std::clock());  //std::clock() is seed
  std::uniform_real_distribution<double> tUnityDistribution(0.,1.);
  double tU = tUnityDistribution(fGenerator);
  double tPhi = 2.*TMath::Pi()*tU; //azimuthal angle


  RotateParticlesByRandomAzimuthalAngle(tPhi, fAllParticlesCollection, aOutputEP);
  RotateParticlesByRandomAzimuthalAngle(tPhi, fAllDaughtersCollection, aOutputEP);

  RotateParticlesByRandomAzimuthalAngle(tPhi, fLambdaCollection, aOutputEP);
  RotateParticlesByRandomAzimuthalAngle(tPhi, fAntiLambdaCollection, aOutputEP);
  RotateParticlesByRandomAzimuthalAngle(tPhi, fK0ShortCollection, aOutputEP);

  RotateParticlesByRandomAzimuthalAngle(tPhi, fKchPCollection, aOutputEP);
  RotateParticlesByRandomAzimuthalAngle(tPhi, fKchMCollection, aOutputEP);

  RotateParticlesByRandomAzimuthalAngle(tPhi, fProtCollection, aOutputEP);
  RotateParticlesByRandomAzimuthalAngle(tPhi, fAProtCollection, aOutputEP);

}

//________________________________________________________________________________________________________________
bool ThermEvent::IncludeInV3(int aV3InclusionProb1, ThermParticle& aParticle)
{
  if(aV3InclusionProb1<0) return true; //Include all

//  std::default_random_engine tGenerator (std::clock());  //std::clock() is seed
  std::normal_distribution<double> tNormDist(0., 2.);
  std::uniform_real_distribution<double> tUnityDistribution(0.,1.);

  double tRand = 100*tUnityDistribution(fGenerator);
  double tAcceptableDistFrom3 = abs(tNormDist(fGenerator));

  if(tRand < aV3InclusionProb1 && abs(aParticle.GetPt()-3.0)<tAcceptableDistFrom3) return true;
  else return false;
}

//________________________________________________________________________________________________________________
void ThermEvent::BuildArtificialV3SignalInCollection(int aV3InclusionProb1, double aPsi3, TF1* aDist, vector<ThermParticle> &aCollection)
{
//  std::default_random_engine tGenerator (std::clock());  //std::clock() is seed
  std::normal_distribution<double> tNormDist(0., 2.);
  std::uniform_real_distribution<double> tUnityDistribution(0.,1.);

  double tRand;
  double tAcceptableDistFrom3;
  double tNewPhi, tGenPhi;
  bool tInclude;
  for(unsigned int iPart=0; iPart<aCollection.size(); iPart++) 
  {
    if(aV3InclusionProb1<0) tInclude = true; //Include all
    else
    {
      tRand = 100*tUnityDistribution(fGenerator);
      tAcceptableDistFrom3 = abs(tNormDist(fGenerator));
      if(tRand < aV3InclusionProb1 && abs(aCollection[iPart].GetPt()-3.0)<tAcceptableDistFrom3) tInclude = true;
      else tInclude = false;
    }
    if(tInclude)
    {
      tGenPhi = aDist->GetRandom();
      tNewPhi = tGenPhi-aCollection[iPart].GetPhiP()+aPsi3;
      aCollection[iPart].TransformRotateZ(-tNewPhi);  //Negative sign because
                                                      //want CCW rotation
    }
  }
}

//________________________________________________________________________________________________________________
void ThermEvent::BuildArtificialV3SignalInCollection(int aV3InclusionProb1, double aPsi3, TF1* aDist, vector<ThermV0Particle> &aCollection)
{
//  std::default_random_engine tGenerator (std::clock());  //std::clock() is seed
  std::normal_distribution<double> tNormDist(0., 2.);
  std::uniform_real_distribution<double> tUnityDistribution(0.,1.);

  int tRand;
  double tAcceptableDistFrom3;
  double tNewPhi, tGenPhi;
  bool tInclude;
  for(unsigned int iPart=0; iPart<aCollection.size(); iPart++) 
  {
    if(aV3InclusionProb1<0) tInclude = true; //Include all
    else
    {
      tRand = 100*tUnityDistribution(fGenerator);
      tAcceptableDistFrom3 = abs(tNormDist(fGenerator));
      if(tRand < aV3InclusionProb1 && abs(aCollection[iPart].GetPt()-3.0)<tAcceptableDistFrom3) tInclude = true;
      else tInclude = false;
    }
    if(tInclude)
    {
      tGenPhi = aDist->GetRandom();
      tNewPhi = tGenPhi-aCollection[iPart].GetPhiP()+aPsi3;
      aCollection[iPart].TransformRotateZ(-tNewPhi);  //Negative sign because
                                                      //want CCW rotation
    }
  }
}
 


//________________________________________________________________________________________________________________
void ThermEvent::BuildArtificialV3Signal(int aV3InclusionProb1, bool aRotateEventsByRandAzAngles)
{
  gRandom->SetSeed();


//  std::default_random_engine tGenerator (std::clock());  //std::clock() is seed
  std::uniform_real_distribution<double> tUnityDistribution(0.,1.);
  double tU = tUnityDistribution(fGenerator);
  double tPsi3 = 2.*TMath::Pi()*tU; //azimuthal angle

  if(aV3InclusionProb1==-1 && !aRotateEventsByRandAzAngles) tPsi3=0.;  //In this special case, since there is no v2 and the entire
                                                                       //signal is v3, I want the third harmonic flow planes to align

  TF1 *f1 = new TF1("f1", "0.5*(cos(3*x)+1)", 0, 2.*TMath::Pi());

  BuildArtificialV3SignalInCollection(aV3InclusionProb1, tPsi3, f1, fAllParticlesCollection);
  BuildArtificialV3SignalInCollection(aV3InclusionProb1, tPsi3, f1, fAllDaughtersCollection);

  BuildArtificialV3SignalInCollection(aV3InclusionProb1, tPsi3, f1, fLambdaCollection);
  BuildArtificialV3SignalInCollection(aV3InclusionProb1, tPsi3, f1, fAntiLambdaCollection);
  BuildArtificialV3SignalInCollection(aV3InclusionProb1, tPsi3, f1, fK0ShortCollection);

  BuildArtificialV3SignalInCollection(aV3InclusionProb1, tPsi3, f1, fKchPCollection);
  BuildArtificialV3SignalInCollection(aV3InclusionProb1, tPsi3, f1, fKchMCollection);

  BuildArtificialV3SignalInCollection(aV3InclusionProb1, tPsi3, f1, fProtCollection);
  BuildArtificialV3SignalInCollection(aV3InclusionProb1, tPsi3, f1, fAProtCollection);

  delete f1;
}


//________________________________________________________________________________________________________________
void ThermEvent::BuildArtificialV2Signal(int aV3InclusionProb1, bool aRotateEventsByRandAzAngles)
{
  gRandom->SetSeed();


//  std::default_random_engine tGenerator (std::clock());  //std::clock() is seed
  std::uniform_real_distribution<double> tUnityDistribution(0.,1.);
  double tU = tUnityDistribution(fGenerator);
  double tPsi2 = 2.*TMath::Pi()*tU; //azimuthal angle

  TF1 *f1 = new TF1("f1", "0.5*(cos(2*x)+1)", 0, 2.*TMath::Pi());

  BuildArtificialV3SignalInCollection(aV3InclusionProb1, tPsi2, f1, fAllParticlesCollection);
  BuildArtificialV3SignalInCollection(aV3InclusionProb1, tPsi2, f1, fAllDaughtersCollection);

  BuildArtificialV3SignalInCollection(aV3InclusionProb1, tPsi2, f1, fLambdaCollection);
  BuildArtificialV3SignalInCollection(aV3InclusionProb1, tPsi2, f1, fAntiLambdaCollection);
  BuildArtificialV3SignalInCollection(aV3InclusionProb1, tPsi2, f1, fK0ShortCollection);

  BuildArtificialV3SignalInCollection(aV3InclusionProb1, tPsi2, f1, fKchPCollection);
  BuildArtificialV3SignalInCollection(aV3InclusionProb1, tPsi2, f1, fKchMCollection);

  BuildArtificialV3SignalInCollection(aV3InclusionProb1, tPsi2, f1, fProtCollection);
  BuildArtificialV3SignalInCollection(aV3InclusionProb1, tPsi2, f1, fAProtCollection);

  delete f1;
}



