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

  for(unsigned int i=0; i<fLambdaCollection.size(); i++)
  {
    if(!fLambdaCollection[i].BothDaughtersFound()) 
    {
      cout << "WARNING: !fLambdaCollection[" << i << "].BothDaughtersFound()" << endl;
      cout << "\t Deleting element..." << endl << endl;
      fLambdaCollection.erase(fLambdaCollection.begin()+i);
    }
  }
  for(unsigned int i=0; i<fLambdaCollection.size(); i++) assert(fLambdaCollection[i].BothDaughtersFound());
  //----------------------------------------------------
  for(unsigned int i=0; i<fAntiLambdaCollection.size(); i++)
  {
    if(!fAntiLambdaCollection[i].BothDaughtersFound()) 
    {
      cout << "WARNING: !fAntiLambdaCollection[" << i << "].BothDaughtersFound()" << endl;
      cout << "\t Deleting element..." << endl << endl;
      fAntiLambdaCollection.erase(fAntiLambdaCollection.begin()+i);
    }
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
  ClearCollection(fAllParticlesCollection);
}

//________________________________________________________________________________________________________________
vector<ThermV0Particle> ThermEvent::GetV0ParticleCollection(ParticlePDGType aPDGType)
{
  if(aPDGType == kPDGLam) return fLambdaCollection;
  else if(aPDGType == kPDGALam) return fAntiLambdaCollection;
  else if(aPDGType == kPDGK0) return fK0ShortCollection;

  else assert(0);
}


//________________________________________________________________________________________________________________
vector<ThermParticle> ThermEvent::GetParticleCollection(ParticlePDGType aPDGType)
{

  if(aPDGType == kPDGKchP) return fKchPCollection;
  else if(aPDGType == kPDGKchM) return fKchMCollection;
  else if(aPDGType == kPDGProt) return fProtCollection;
  else if(aPDGType == kPDGAntiProt) return fAProtCollection;

  else assert(0);
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
  std::default_random_engine tGenerator (std::clock());  //std::clock() is seed
  std::uniform_real_distribution<double> tUnityDistribution(0.,1.);
  double tU = tUnityDistribution(tGenerator);
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


