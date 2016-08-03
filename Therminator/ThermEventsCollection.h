///////////////////////////////////////////////////////////////////////////
// ThermEventsCollection:                                                //
///////////////////////////////////////////////////////////////////////////

#ifndef THERMEVENTSCOLLECTION
#define THERMEVENTSCOLLECTION

#include "TSystemDirectory.h"
#include "TSystemFile.h"

#include "ThermEvent.h"
class ThermEvent;


class ThermEventsCollection {

public:
  ThermEventsCollection();
  virtual ~ThermEventsCollection();

  void ExtractEventsFromRootFile(TString aFileLocation);
  void ExtractFromAllFiles(const char *aDirName);

private:
  vector<TString> fFileNameCollection;
  vector<ThermEvent*> fEventsCollection;



#ifdef __ROOT__
  ClassDef(ThermEventsCollection, 1)
#endif
};


//inline stuff


#endif




