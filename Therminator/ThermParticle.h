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
  void TransformRotateZ(double aPhi);  //NOTE: Clockwise rotation!!!
  void TransformToPRF(double aBetaT);

  TLorentzVector GetFourPosition() const;
  TLorentzVector GetFourMomentum() const;

  TLorentzVector GetFatherFourPosition() const;
  TLorentzVector GetFatherFourMomentum() const;

  void LoadFather(ThermParticle& aFather);

  bool PassKinematicCuts();

  //inline-----------------------------
  bool IsPrimordial() const;
  bool IsParticleOfInterest() const;

  double GetMass() const;

  double GetT() const;
  double GetX() const;
  double GetY() const;
  double GetZ() const;

  double GetE() const;
  double GetPx() const;
  double GetPy() const;
  double GetPz() const;
  double GetMagP() const;

  int GetDecayed() const;
  int GetPID() const;
  int GetFatherPID() const;
  int GetRootPID() const;
  int GetEID() const;
  int GetFatherEID() const;
  int GetEventID() const;

  //----------

  void SetDecayed(int aDecayed);

  //---------
  double GetFatherMass() const;

  double GetFatherT() const;
  double GetFatherX() const;
  double GetFatherY() const;
  double GetFatherZ() const;

  double GetFatherE() const;
  double GetFatherPx() const;
  double GetFatherPy() const;
  double GetFatherPz() const;
  double GetFatherMagP() const;

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

  double fFatherMass;
  double fFatherT, fFatherX, fFatherY, fFatherZ;
  double fFatherE, fFatherPx, fFatherPy, fFatherPz;

#ifdef __ROOT__
  ClassDef(ThermParticle, 1)
#endif
};


//inline stuff
inline bool ThermParticle::IsPrimordial() const {return fPrimordial;}
inline bool ThermParticle::IsParticleOfInterest() const {return fParticleOfInterest;}

inline double ThermParticle::GetMass() const {return fMass;}

inline double ThermParticle::GetT() const {return fT;}
inline double ThermParticle::GetX() const {return fX;}
inline double ThermParticle::GetY() const {return fY;}
inline double ThermParticle::GetZ() const {return fZ;}

inline double ThermParticle::GetE() const {return fE;}
inline double ThermParticle::GetPx() const {return fPx;}
inline double ThermParticle::GetPy() const {return fPy;}
inline double ThermParticle::GetPz() const {return fPz;}
inline double ThermParticle::GetMagP() const {return sqrt(fPx*fPx + fPy*fPy + fPz*fPz);}

inline int ThermParticle::GetDecayed() const {return fDecayed;}
inline int ThermParticle::GetPID() const {return fPID;}
inline int ThermParticle::GetFatherPID() const {return fFatherPID;}
inline int ThermParticle::GetRootPID() const {return fRootPID;}
inline int ThermParticle::GetEID() const {return fEID;}
inline int ThermParticle::GetFatherEID() const {return fFatherEID;}
inline int ThermParticle::GetEventID() const {return fEventID;}

//----------
inline void ThermParticle::SetDecayed(int aDecayed) {fDecayed = aDecayed;}
//----------

inline double ThermParticle::GetFatherMass() const {return fFatherMass;}

inline double ThermParticle::GetFatherT() const {return fFatherT;}
inline double ThermParticle::GetFatherX() const {return fFatherX;}
inline double ThermParticle::GetFatherY() const {return fFatherY;}
inline double ThermParticle::GetFatherZ() const {return fFatherZ;}

inline double ThermParticle::GetFatherE() const {return fFatherE;}
inline double ThermParticle::GetFatherPx() const {return fFatherPx;}
inline double ThermParticle::GetFatherPy() const {return fFatherPy;}
inline double ThermParticle::GetFatherPz() const {return fFatherPz;}
inline double ThermParticle::GetFatherMagP() const {return sqrt(fFatherPx*fFatherPx + fFatherPy*fFatherPy + fFatherPz*fFatherPz);}




#endif
