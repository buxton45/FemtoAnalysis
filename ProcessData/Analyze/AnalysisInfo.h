/* AnalysisInfo.h */

#ifndef ANALYSISINFO_H
#define ANALYSISINFO_H

#include <iostream>
#include <cassert>

#include "Types.h"

using std::cout;
using std::endl;

class AnalysisInfo {

public:
  AnalysisInfo(AnalysisType aAnalysisType);
  virtual ~AnalysisInfo();

  ParticlePDGType GetParticlePDGType(ParticleType aParticleType);
  void SetDaughterPairType();
  void SetCoulombType();
  void SetBohrRadius();
  void SetIsResidual();

  //Inline functions
  AnalysisType GetAnalysisType();
  AnalysisType GetConjAnalysisType();
  bool IsResidual();

  vector<ParticleType> GetParticleTypes();
  vector<ParticlePDGType> GetParticlePDGTypes();

  vector<DaughterPairType> GetDaughterPairTypes();

  CoulombType GetCoulombType();
  double GetBohrRadius();


private:
  AnalysisType fAnalysisType;
  AnalysisType fConjAnalysisType;
  bool fIsResidual;

  vector<ParticleType> fParticleTypes;
  vector<ParticlePDGType> fParticlePDGTypes;

  vector<DaughterPairType> fDaughterPairTypes;  //if residual analysis, this will be empty

  CoulombType fCoulombType;
  double fBohrRadius;


#ifdef __ROOT__
  ClassDef(AnalysisInfo, 1)
#endif
};

inline AnalysisType AnalysisInfo::GetAnalysisType() {return fAnalysisType;}
inline AnalysisType AnalysisInfo::GetConjAnalysisType() {return fConjAnalysisType;}
inline bool AnalysisInfo::IsResidual() {return fIsResidual;}

inline vector<ParticleType> AnalysisInfo::GetParticleTypes() {return fParticleTypes;}
inline vector<ParticlePDGType> AnalysisInfo::GetParticlePDGTypes() {return fParticlePDGTypes;}

inline vector<DaughterPairType> AnalysisInfo::GetDaughterPairTypes() {return fDaughterPairTypes;}

inline CoulombType AnalysisInfo::GetCoulombType() {return fCoulombType;}
inline double AnalysisInfo::GetBohrRadius() {return fBohrRadius;}

#endif
