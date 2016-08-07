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
  fKchMCollection(0)

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
  fKchMCollection(0)

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
  fKchMCollection(aEvent.fKchMCollection)
{

}

//________________________________________________________________________________________________________________
ThermEvent& ThermEvent::operator=(ThermEvent& aEvent)
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
  cout << "ThermEvent object is being deleted!!!" << endl;
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
}

//________________________________________________________________________________________________________________
void ThermEvent::ClearCollection(vector<ThermV0Particle> &aCollection)
{
  aCollection.clear();
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
}

//________________________________________________________________________________________________________________
void ThermEvent::AssertAllLambdaFathersFoundDaughters()
{
  for(unsigned int i=0; i<fLambdaCollection.size(); i++) assert(fLambdaCollection[i].BothDaughtersFound());
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
}

//________________________________________________________________________________________________________________
void ThermEvent::FindV0Father(ThermV0Particle &aV0Particle)
{
  if(!aV0Particle.IsPrimordial())
  {
    int tFatherEID = aV0Particle.GetFatherEID();

    int tFatherLocation = -1;
    for(unsigned int i=0; i<fAllParticlesCollection.size(); i++)
    {
      if(fAllParticlesCollection[i].GetEID() == tFatherEID)
      {
        tFatherLocation = i;
        aV0Particle.LoadFather(fAllParticlesCollection[i]);
        break;
      }
    }
    assert(tFatherLocation >= 0);

  }
}


//________________________________________________________________________________________________________________
void ThermEvent::FindAllV0sFathers()
{
  for(unsigned int i=0; i<fLambdaCollection.size(); i++) FindV0Father(fLambdaCollection[i]);
  for(unsigned int i=0; i<fAntiLambdaCollection.size(); i++) FindV0Father(fAntiLambdaCollection[i]);
  for(unsigned int i=0; i<fK0ShortCollection.size(); i++) FindV0Father(fK0ShortCollection[i]);
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

  else assert(0);
}

//________________________________________________________________________________________________________________
void ThermEvent::SetV0ParticleCollection(int aEventID, ParticlePDGType aPDGType, vector<ThermV0Particle> &aCollection)
{
  assert(aEventID == fEventID);  //make sure I'm setting the collection for the correct event

  if(aPDGType == kPDGLam) fLambdaCollection = aCollection;
  else if(aPDGType == kPDGALam) fAntiLambdaCollection = aCollection;
  else if(aPDGType == kPDGK0) fK0ShortCollection = aCollection;

  else assert(0);
}


//________________________________________________________________________________________________________________
void ThermEvent::SetParticleCollection(int aEventID, ParticlePDGType aPDGType, vector<ThermParticle> &aCollection)
{
  assert(aEventID == fEventID);  //make sure I'm setting the collection for the correct event

  if(aPDGType == kPDGKchP) fKchPCollection = aCollection;
  else if(aPDGType == kPDGKchM) fKchMCollection = aCollection;

  else assert(0);
}

//________________________________________________________________________________________________________________
bool ThermEvent::DoubleCheckLamAttributes(ThermV0Particle &aV0)
{
  if(!aV0.GoodV0()) {cout << "DoubleCheckLamAttributes Fail 1" << endl; return false;}
  if(aV0.GetPID()!= kPDGLam) {cout << "DoubleCheckLamAttributes Fail 2" << endl; return false;}
  if(!aV0.BothDaughtersFound()) {cout << "DoubleCheckLamAttributes Fail 3" << endl; return false;}
  if(aV0.GetDaughter1PID() != kPDGProt) {cout << "DoubleCheckLamAttributes Fail 4" << endl; return false;}
  if(aV0.GetDaughter2PID() != kPDGPiM) {cout << "DoubleCheckLamAttributes Fail 5" << endl; return false;}

  if(aV0.GetDaughter1Mass()==0) {cout << "DoubleCheckLamAttributes Fail 6" << endl; return false;}
  if(aV0.GetDaughter1T()==0 || aV0.GetDaughter1X()==0 || aV0.GetDaughter1Y()==0 ||aV0.GetDaughter1Z()==0) {cout << "DoubleCheckLamAttributes Fail 7" << endl; return false;}
  if(aV0.GetDaughter1E()==0 || aV0.GetDaughter1Px()==0 || aV0.GetDaughter1Py()==0 ||aV0.GetDaughter1Pz()==0) {cout << "DoubleCheckLamAttributes Fail 8" << endl; return false;}

  if(aV0.GetDaughter2Mass()==0) {cout << "DoubleCheckLamAttributes Fail 9" << endl; return false;}
  if(aV0.GetDaughter2T()==0 || aV0.GetDaughter2X()==0 || aV0.GetDaughter2Y()==0 ||aV0.GetDaughter2Z()==0) {cout << "DoubleCheckLamAttributes Fail 10" << endl; return false;}
  if(aV0.GetDaughter2E()==0 || aV0.GetDaughter2Px()==0 || aV0.GetDaughter2Py()==0 ||aV0.GetDaughter2Pz()==0) {cout << "DoubleCheckLamAttributes Fail 11" << endl; return false;}

  if(aV0.GetFatherT()==0 || aV0.GetFatherX()==0 || aV0.GetFatherY()==0 ||aV0.GetFatherZ()==0) {cout << "DoubleCheckLamAttributes Fail 12" << endl; return false;}
  if(aV0.GetFatherE()==0 || aV0.GetFatherPx()==0 || aV0.GetFatherPy()==0 ||aV0.GetFatherPz()==0) {cout << "DoubleCheckLamAttributes Fail 13" << endl; return false;}


  return true;
}

//________________________________________________________________________________________________________________
bool ThermEvent::DoubleCheckV0Attributes(ThermV0Particle &aV0)
{
  bool tReturn;

  if(aV0.GetPID() == kPDGLam) tReturn = DoubleCheckLamAttributes(aV0);
//  else if(aV0.GetPID() == kPDGALam) tReturn = DoubleCheckALamAttributes(aV0);
//  else if(aV0.GetPID() == kPDGK0) tReturn = DoubleCheckK0Attributes(aV0);
  else assert(0);

  return tReturn;
}

//________________________________________________________________________________________________________________
double ThermEvent::GetKStar(ThermParticle &aParticle, ThermV0Particle &aV0)
{
  double px1, py1, pz1, mass1, E1;
  double px2, py2, pz2, mass2, E2;

  px1 = aParticle.GetPx();
  py1 = aParticle.GetPy();
  pz1 = aParticle.GetPz();
  mass1 = aParticle.GetMass();
  E1 = aParticle.GetE();

  px2 = aV0.GetPx();
  py2 = aV0.GetPy();
  pz2 = aV0.GetPz();
  mass2 = aV0.GetMass();
  E2 = aV0.GetE();

  double tMinvSq = (E1+E2)*(E1+E2) - (px1+px2)*(px1+px2) - (py1+py2)*(py1+py2) - (pz1+pz2)*(pz1+pz2);
  double tQinvSq = ((mass1*mass1 - mass2*mass2)*(mass1*mass1 - mass2*mass2))/tMinvSq + tMinvSq - 2.0*(mass1*mass1 + mass2*mass2);
  double tKStar = 0.5*sqrt(tQinvSq);

  return tKStar;
}

//________________________________________________________________________________________________________________
double ThermEvent::GetKStar(ThermV0Particle &aV01, ThermV0Particle &aV02)
{
  double px1, py1, pz1, mass1, E1;
  double px2, py2, pz2, mass2, E2;

  px1 = aV01.GetPx();
  py1 = aV01.GetPy();
  pz1 = aV01.GetPz();
  mass1 = aV01.GetMass();
  E1 = aV01.GetE();

  px2 = aV02.GetPx();
  py2 = aV02.GetPy();
  pz2 = aV02.GetPz();
  mass2 = aV02.GetMass();
  E2 = aV02.GetE();

  double tMinvSq = (E1+E2)*(E1+E2) - (px1+px2)*(px1+px2) - (py1+py2)*(py1+py2) - (pz1+pz2)*(pz1+pz2);
  double tQinvSq = ((mass1*mass1 - mass2*mass2)*(mass1*mass1 - mass2*mass2))/tMinvSq + tMinvSq - 2.0*(mass1*mass1 + mass2*mass2);
  double tKStar = 0.5*sqrt(tQinvSq);

  return tKStar;
}

//________________________________________________________________________________________________________________
double ThermEvent::GetFatherKStar(ThermParticle &aParticle, ThermV0Particle &aV0)
{
  double px1, py1, pz1, mass1, E1;
  double px2, py2, pz2, mass2, E2;

  px1 = aParticle.GetPx();
  py1 = aParticle.GetPy();
  pz1 = aParticle.GetPz();
  mass1 = aParticle.GetMass();
  E1 = aParticle.GetE();

  px2 = aV0.GetFatherPx();
  py2 = aV0.GetFatherPy();
  pz2 = aV0.GetFatherPz();
  mass2 = aV0.GetFatherMass();
  E2 = aV0.GetFatherE();

  double tMinvSq = (E1+E2)*(E1+E2) - (px1+px2)*(px1+px2) - (py1+py2)*(py1+py2) - (pz1+pz2)*(pz1+pz2);
  double tQinvSq = ((mass1*mass1 - mass2*mass2)*(mass1*mass1 - mass2*mass2))/tMinvSq + tMinvSq - 2.0*(mass1*mass1 + mass2*mass2);
  double tKStar = 0.5*sqrt(tQinvSq);

  return tKStar;
}

//________________________________________________________________________________________________________________
double ThermEvent::GetFatherKStar(ThermV0Particle &aV01, ThermV0Particle &aV02)
{
  double px1, py1, pz1, mass1, E1;
  double px2, py2, pz2, mass2, E2;

  px1 = aV01.GetFatherPx();
  py1 = aV01.GetFatherPy();
  pz1 = aV01.GetFatherPz();
  mass1 = aV01.GetFatherMass();
  E1 = aV01.GetFatherE();

  px2 = aV02.GetPx();
  py2 = aV02.GetPy();
  pz2 = aV02.GetPz();
  mass2 = aV02.GetMass();
  E2 = aV02.GetE();

  double tMinvSq = (E1+E2)*(E1+E2) - (px1+px2)*(px1+px2) - (py1+py2)*(py1+py2) - (pz1+pz2)*(pz1+pz2);
  double tQinvSq = ((mass1*mass1 - mass2*mass2)*(mass1*mass1 - mass2*mass2))/tMinvSq + tMinvSq - 2.0*(mass1*mass1 + mass2*mass2);
  double tKStar = 0.5*sqrt(tQinvSq);

  return tKStar;
}


//________________________________________________________________________________________________________________
void ThermEvent::FillTransformMatrix(TH2* aMatrix)
{
  ThermV0Particle tV0;
  ThermParticle tParticle;
  double tKStar, tFatherKStar;

  for(unsigned int iV0=0; iV0<fLambdaCollection.size(); iV0++)
  {
    tV0 = fLambdaCollection[iV0];
    if(tV0.GetFatherPID() == kPDGSigma && tV0.GoodV0())
    {
      for(unsigned int iPar=0; iPar<fKchPCollection.size(); iPar++)
      {
        tParticle = fKchPCollection[iPar];
        
        tKStar = GetKStar(tParticle,tV0);
        tFatherKStar = GetFatherKStar(tParticle,tV0);

        aMatrix->Fill(tKStar,tFatherKStar);
      }
    }
  }
}


//________________________________________________________________________________________________________________
/*
void ThermEvent::FillTransformMatrix(TH2* aMatrix)
{
  ThermV0Particle tV01, tV02;
  double tKStar, tFatherKStar;

  for(unsigned int iV01=0; iV01<fLambdaCollection.size(); iV01++)
  {
    tV01 = fLambdaCollection[iV01];
    if(tV01.GetFatherPID() == kPDGSigma && tV01.GoodV0())
    {
      for(unsigned int iV02=0; iV02<fLambdaCollection.size(); iV02++)
      {
        tV02 = fLambdaCollection[iV02];
        
        if(tV02.GoodV0() && tV02.GetEID() != tV01.GetEID())
        {
          tKStar = GetKStar(tV01,tV02);
          tFatherKStar = GetFatherKStar(tV01,tV02);

          if(DoubleCheckV0Attributes(tV01) && DoubleCheckV0Attributes(tV02))
          {
            aMatrix->Fill(tKStar,tFatherKStar);
          }
        }
      }
    }
  }
}
*/

