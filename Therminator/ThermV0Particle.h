///////////////////////////////////////////////////////////////////////////
// ThermV0Particle:                                                  //
///////////////////////////////////////////////////////////////////////////

#ifndef THERMV0PARTICLE
#define THERMV0PARTICLE

#include "ThermParticle.h"
class ThermParticle;

class ThermV0Particle : public ThermParticle {

public:
  ThermV0Particle();
  ThermV0Particle(ParticleCoor* aParticle);
  ThermV0Particle(const ThermV0Particle& aParticle);
  ThermV0Particle& operator=(const ThermV0Particle& aParticle);
  virtual ThermV0Particle* clone();

  virtual ~ThermV0Particle();

  void LoadDaughter1(ThermParticle& aDaughter);
  void LoadDaughter2(ThermParticle& aDaughter);
  void LoadDaughter(ThermParticle& aDaughter);

  //inline-----------------------------
  bool Daughter1Found();
  bool Daughter2Found();
  bool BothDaughtersFound();

  bool GoodLambda();

  int GetDaughter1PID();
  int GetDaughter2PID();

  int GetDaughter1EID();
  int GetDaughter2EID();

  double GetDaughter1Mass();

  double GetDaughter1T();
  double GetDaughter1X();
  double GetDaughter1Y();
  double GetDaughter1Z();

  double GetDaughter1E();
  double GetDaughter1Px();
  double GetDaughter1Py();
  double GetDaughter1Pz();

  double GetDaughter2Mass();

  double GetDaughter2T();
  double GetDaughter2X();
  double GetDaughter2Y();
  double GetDaughter2Z();

  double GetDaughter2E();
  double GetDaughter2Px();
  double GetDaughter2Py();
  double GetDaughter2Pz();

  //----------

  void SetDaughter1PID(int aPID);    // If daughters are charged, Daughter1 = positive
  void SetDaughter2PID(int aPID);  // If daughters are charged, Daughter2 = negative

private:
  bool fDaughter1Found;
  bool fDaughter2Found;
  bool fBothDaughtersFound;

  bool fGoodLambda;

  int fDaughter1PID;
  int fDaughter2PID;

  int fDaughter1EID;
  int fDaughter2EID;

  double fDaughter1Mass;
  double fDaughter1T, fDaughter1X, fDaughter1Y, fDaughter1Z;
  double fDaughter1E, fDaughter1Px, fDaughter1Py, fDaughter1Pz;

  double fDaughter2Mass;
  double fDaughter2T, fDaughter2X, fDaughter2Y, fDaughter2Z;
  double fDaughter2E, fDaughter2Px, fDaughter2Py, fDaughter2Pz;


#ifdef __ROOT__
  ClassDef(ThermV0Particle, 1)
#endif
};


//inline stuff
inline bool ThermV0Particle::Daughter1Found() {return fDaughter1Found;}
inline bool ThermV0Particle::Daughter2Found() {return fDaughter2Found;}
inline bool ThermV0Particle::BothDaughtersFound() {return fBothDaughtersFound;}

inline bool ThermV0Particle::GoodLambda() {return fGoodLambda;}

inline int ThermV0Particle::GetDaughter1PID() {return fDaughter1PID;}
inline int ThermV0Particle::GetDaughter2PID() {return fDaughter2PID;}

inline int ThermV0Particle::GetDaughter1EID() {return fDaughter1EID;}
inline int ThermV0Particle::GetDaughter2EID() {return fDaughter2EID;}

inline double ThermV0Particle::GetDaughter1Mass() {return fDaughter1Mass;}

inline double ThermV0Particle::GetDaughter1T() {return fDaughter1T;}
inline double ThermV0Particle::GetDaughter1X() {return fDaughter1X;}
inline double ThermV0Particle::GetDaughter1Y() {return fDaughter1Y;}
inline double ThermV0Particle::GetDaughter1Z() {return fDaughter1Z;}

inline double ThermV0Particle::GetDaughter1E() {return fDaughter1E;}
inline double ThermV0Particle::GetDaughter1Px() {return fDaughter1Px;}
inline double ThermV0Particle::GetDaughter1Py() {return fDaughter1Py;}
inline double ThermV0Particle::GetDaughter1Pz() {return fDaughter1Pz;}

inline double ThermV0Particle::GetDaughter2Mass() {return fDaughter2Mass;}

inline double ThermV0Particle::GetDaughter2T() {return fDaughter2T;}
inline double ThermV0Particle::GetDaughter2X() {return fDaughter2X;}
inline double ThermV0Particle::GetDaughter2Y() {return fDaughter2Y;}
inline double ThermV0Particle::GetDaughter2Z() {return fDaughter2Z;}

inline double ThermV0Particle::GetDaughter2E() {return fDaughter2E;}
inline double ThermV0Particle::GetDaughter2Px() {return fDaughter2Px;}
inline double ThermV0Particle::GetDaughter2Py() {return fDaughter2Py;}
inline double ThermV0Particle::GetDaughter2Pz() {return fDaughter2Pz;}

//----------

inline void ThermV0Particle::SetDaughter1PID(int aPID) {fDaughter1PID = aPID;}
inline void ThermV0Particle::SetDaughter2PID(int aPID) {fDaughter2PID = aPID;}



#endif
