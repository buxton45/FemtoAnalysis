///////////////////////////////////////////////////////////////////////////
// ThermParticle:                                                        //
//             Basically just a ParticleCoor object, but this allows me  //
// to also add objects if I want.  ThermV0Particle will be built         //
// ontop of this                                                         //
///////////////////////////////////////////////////////////////////////////

#ifndef THERMPARTICLE
#define THERMPARTICLE

//includes and any constant variable declarations
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>
#include <complex>
#include <math.h>
#include <vector>
#include <ctime>
#include <random>
#include <chrono>
#include <algorithm>
#include <limits>
#include <assert.h>

#include <TMath.h>
#include <TLorentzVector.h>

using std::cout;
using std::endl;
using std::vector;


#include "Types.h"
#include "ChronoTimer.h"
#include "THGlobal.h"
#include "ParticleCoor.h"

class ThermParticle {

public:
  ThermParticle();
  ThermParticle(ParticleCoor* aParticleCoor);
  ThermParticle(const ThermParticle& aParticle);
  ThermParticle& operator=(ThermParticle& aParticle);

  virtual ~ThermParticle();

  void SetIsParticleOfInterest();

  double GetTau();
  double GetR();
  double GetRho();
  double GetPhiS();
  double GetRapidityS();

  double GetP();
  double GetPt();
  double GetMt();
  double GetPhiP();
  double GetRapidityP();
  double GetEtaP();

  void TransformToLCMS(double aBetaZ);
  void TransformRotateZ(double aPhi);
  void TransformToPRF(double aBetaT);

  TLorentzVector* GetFourPosition();
  TLorentzVector* GetFourMomentum();

  //inline
  bool IsParticleOfInterest();
  int GetEID();

  int GetFatherEID();

  int GetDecayed();
  void SetDecayed(int aDecayed);

  double GetMass();

  int GetPID();
  int GetFatherPID();

private:
  bool fParticleOfInterest;

  double fMass;
  double fT, fX, fY, fZ;
  double fE, fPx, fPy, fPz;
  int fDecayed;
  int fPID;
  int fFatherPID;
  int fRootPID;
  int fEID;
  int fFatherEID;
  int fEventID;



#ifdef __ROOT__
  ClassDef(ThermParticle, 1)
#endif
};


//inline stuff
inline bool ThermParticle::IsParticleOfInterest() {return fParticleOfInterest;}
inline int ThermParticle::GetEID() {return fEID;}

inline int ThermParticle::GetFatherEID() {return fFatherEID;}

inline int ThermParticle::GetDecayed() {return fDecayed;}
inline void ThermParticle::SetDecayed(int aDecayed) {fDecayed = aDecayed;}

inline double ThermParticle::GetMass() {return fMass;}

inline int ThermParticle::GetPID() {return fPID;}
inline int ThermParticle::GetFatherPID() {return fFatherPID;}

#endif
