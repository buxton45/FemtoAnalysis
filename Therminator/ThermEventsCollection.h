///////////////////////////////////////////////////////////////////////////
// ThermEventsCollection:                                                //
///////////////////////////////////////////////////////////////////////////

#ifndef THERMEVENTSCOLLECTION
#define THERMEVENTSCOLLECTION

#include "ThermEvent.h"
class ThermEvent;


class ThermEventsCollection {

public:
  ThermEventsCollection();
  virtual ~ThermEventsCollection();

  bool IsParticleOfInterest(ParticleCoor* tParticle);
  void ExtractEventsFromRootFile(TString aFileLocation);

private:
  vector<ThermEvent*> fEventsCollection;



#ifdef __ROOT__
  ClassDef(ThermEventsCollection, 1)
#endif
};


//inline stuff


#endif




