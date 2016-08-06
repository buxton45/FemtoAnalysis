///////////////////////////////////////////////////////////////////////////
// ThermEventsCollection:                                                //
///////////////////////////////////////////////////////////////////////////

#ifndef THERMEVENTSCOLLECTION
#define THERMEVENTSCOLLECTION

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>

#include "TSystemDirectory.h"
#include "TSystemFile.h"

#include "ThermEvent.h"
class ThermEvent;


class ThermEventsCollection {

public:
  ThermEventsCollection();
  virtual ~ThermEventsCollection();

  void WriteRow(ostream &aOutput, vector<double> &aRow);
  vector<double> PackageV0ParticleForWriting(ThermV0Particle &aV0);
  vector<double> PackageParticleForWriting(ThermParticle &aParticle);
  void WriteThermEventV0s(ostream &aOutput, ParticlePDGType aParticleType, ThermEvent* aThermEvent);
  void WriteThermEventParticles(ostream &aOutput, ParticlePDGType aParticleType, ThermEvent* aThermEvent);
  void WriteAllEventsParticlesOfType(TString aOutputName, ParticlePDGType aParticleType);
  void WriteAllEvents(TString aOutputNameBase);

  void ExtractEventsFromTxtFile(TString aFileName, ParticlePDGType aPDGType);
  void ExtractEventsFromAllTxtFiles(TString aFileLocationBase);

  void ExtractEventsFromRootFile(TString aFileLocation);
  void ExtractFromAllRootFiles(const char *aDirName);

private:
  vector<TString> fFileNameCollection;
  vector<ThermEvent*> fEventsCollection;



#ifdef __ROOT__
  ClassDef(ThermEventsCollection, 1)
#endif
};


//inline stuff


#endif




