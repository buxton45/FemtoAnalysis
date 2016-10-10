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
  ThermParticle(vector<double> &aVecFromTxt);
  ThermParticle(const ThermParticle& aParticle);
  ThermParticle& operator=(const ThermParticle& aParticle);
  virtual ThermParticle* clone();

  virtual ~ThermParticle();

  void SetIsParticleOfInterest();

  double GetTau() const;
  double GetR() const;
  double GetRho() const;
  double GetPhiS() const;
  double GetRapidityS() const;

  double GetP() const;
  double GetPt() const;
  double GetMt() const;
  double GetPhiP() const;
  double GetRapidityP() const;
  double GetEtaP() const;

  void TransformToLCMS(double aBetaZ);
  void TransformRotateZ(double aPhi);
  void TransformToPRF(double aBetaT);

  TLorentzVector GetFourPosition();
  TLorentzVector GetFourMomentum();

  //inline-----------------------------
  bool IsPrimordial();
  bool IsParticleOfInterest();

  double GetMass();

  double GetT();
  double GetX();
  double GetY();
  double GetZ();

  double GetE();
  double GetPx();
  double GetPy();
  double GetPz();

  int GetDecayed();
  int GetPID();
  int GetFatherPID();
  int GetRootPID();
  int GetEID();
  int GetFatherEID();
  int GetEventID();

  //----------

  void SetDecayed(int aDecayed);


protected:
  bool fPrimordial;
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
inline bool ThermParticle::IsPrimordial() {return fPrimordial;}
inline bool ThermParticle::IsParticleOfInterest() {return fParticleOfInterest;}

inline double ThermParticle::GetMass() {return fMass;}

inline double ThermParticle::GetT() {return fT;}
inline double ThermParticle::GetX() {return fX;}
inline double ThermParticle::GetY() {return fY;}
inline double ThermParticle::GetZ() {return fZ;}

inline double ThermParticle::GetE() {return fE;}
inline double ThermParticle::GetPx() {return fPx;}
inline double ThermParticle::GetPy() {return fPy;}
inline double ThermParticle::GetPz() {return fPz;}

inline int ThermParticle::GetDecayed() {return fDecayed;}
inline int ThermParticle::GetPID() {return fPID;}
inline int ThermParticle::GetFatherPID() {return fFatherPID;}
inline int ThermParticle::GetRootPID() {return fRootPID;}
inline int ThermParticle::GetEID() {return fEID;}
inline int ThermParticle::GetFatherEID() {return fFatherEID;}
inline int ThermParticle::GetEventID() {return fEventID;}

//----------
inline void ThermParticle::SetDecayed(int aDecayed) {fDecayed = aDecayed;}








#endif