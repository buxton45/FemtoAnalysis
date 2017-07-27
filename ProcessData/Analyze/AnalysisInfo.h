/* AnalysisInfo.h */

#ifndef ANALYSISINFO_H
#define ANALYSISINFO_H

#include "Types.h"

class AnalysisInfo {

public:
  AnalysisInfo(AnalysisType aAnalysisType);
  virtual ~AnalysisInfo();


  //Inline functions
  AnalysisType AnalysisType();
  AnalysisType ConjAnalysisType();

  vector<ParticleType> ParticleTypes();
  vector<ParticlePDGType> ParticlePDGTypes();

  DaughterPairType DaughterPairType();

  CoulombType CoulombType();
  double BohrRadius();


private:
  AnalysisType fAnalysisType;
  AnalysisType fConjAnalysisType;

  vector<ParticleType> fParticleTypes;
  vector<ParticlePDGType> fParticlePDGTypes;

  DaughterPairType fDaughterPairType;

  CoulombType fCoulombType;
  double fBohrRadius;


#ifdef __ROOT__
  ClassDef(AnalysisInfo, 1)
#endif
};

inline AnalysisType AnalysisInfo::AnalysisType() {return fAnalysisType;}
inline AnalysisType AnalysisInfo::ConjAnalysisType() {return fConjAnalysisType;}

inline vector<ParticleType> AnalysisInfo::ParticleTypes() {return fParticleTypes;}
inline vector<ParticlePDGType> AnalysisInfo::ParticlePDGTypes() {return fParticlePDGTypes;}

inline DaughterPairType AnalysisInfo::DaughterPairType() {return fDaughterPairType;}

inline CoulombType AnalysisInfo::CoulombType() {return fCoulombType;}
inline double AnalysisInfo::BohrRadius() {return fBohrRadius;}

#endif
