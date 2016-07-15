//NOTE: THIS PROGRAM HAS ESSENTIALLY BEEN REPLACED BY MYANALYSISCONSTRUCTOR!!!

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// AliFemtoVertexMultAnalysis - Femtoscopic analysis which mixes event    //
// with respect to the z position of the primary vertex and event total   //
// multiplicity                                                           //
//                                                                        //
////////////////////////////////////////////////////////////////////////////
//  myAliFemtoVertexMultAnalysis - adds member function GetOutputList     //
//  which will grab my purity histograms from the single particle cuts    // 
//  and add them to the standard output list of AliFemtoSimpleAnalysis    //
////////////////////////////////////////////////////////////////////////////

#ifndef MYALIFEMTOVERTEXMULTANALYSIS_H
#define MYALIFEMTOVERTEXMULTANALYSIS_H

#include "AliFemtoSimpleAnalysis.h"        // base analysis class
#include "myAliFemtoV0TrackCut.h"

class myAliFemtoVertexMultAnalysis : public AliFemtoSimpleAnalysis {

public:

  myAliFemtoVertexMultAnalysis(char* name, unsigned int binsVertex=10, double minVertex=-100., double maxVertex=+100., unsigned int binsMult=10, double minMult=-1.e9, double maxMult=+1.e9);
  myAliFemtoVertexMultAnalysis(const myAliFemtoVertexMultAnalysis& TheOriginalAnalysis);  // copy constructor
  myAliFemtoVertexMultAnalysis& operator=(const myAliFemtoVertexMultAnalysis& TheOriginalAnalysis);    
  virtual void ProcessEvent(const AliFemtoEvent* ProcessThisEvent);
  virtual ~myAliFemtoVertexMultAnalysis();
  virtual AliFemtoString Report();       //! returns reports of all cuts applied and correlation functions being done
  virtual unsigned int OverflowVertexZ() const { return fOverFlowVertexZ;}
  virtual unsigned int UnderflowVertexZ() const { return fUnderFlowVertexZ;}
  virtual unsigned int OverflowMult() const { return fOverFlowMult;}
  virtual unsigned int UnderflowMult() const { return fUnderFlowMult;}
//----Personal
  virtual TList* GetOutputList();
//-----------
protected:
  double fVertexZ[2];                 /* min/max z-vertex position allowed to be processed */
  unsigned int fVertexZBins;          /* number of VERTEX mixing bins in z-vertex in EventMixing Buffer */
  unsigned int fOverFlowVertexZ;      /* number of events encountered which had too large z-vertex */
  unsigned int fUnderFlowVertexZ;     /* number of events encountered which had too small z-vertex */
  double fMult[2];                    /* min/max multiplicity allowed for event to be processed */
  unsigned int fMultBins;             /* number of MULTIPLICITY mixing bins in z-vertex in EventMixing Buffer */
  unsigned int fOverFlowMult;         /* number of events encountered which had too large multiplicity */
  unsigned int fUnderFlowMult;        /* number of events encountered which had too small multiplicity */
  char* fOutputName;		      /* name given to output directory for specific analysis*/
  
#ifdef __ROOT__
  ClassDef(myAliFemtoVertexMultAnalysis, 1)
#endif
    
};

#endif
