///////////////////////////////////////////////////////////////////////////
// Types:                                                                //
//                                                                       //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef TYPES_H
#define TYPES_H

//#include "TString.h"

#include <vector>
#include <complex>
using std::vector;


  //-------enum types-----------------------------------------
  enum PartnerAnalysisType {kcLamK0=0, kcLamcKch=1, kcXicKch=2, kcLamcLam=3, kcLamcPi=4};

  enum AnalysisRunType {kTrain=0, kTrainSys=1, kGrid=2};
  enum AnalysisType {kLamK0=0, kALamK0=1, kLamKchP=2, kALamKchP=3, kLamKchM=4, kALamKchM=5, kXiKchP=6, kAXiKchP=7, kXiKchM=8, kAXiKchM=9, kLamLam=10, kALamALam=11, kLamALam=12, kLamPiP=13, kALamPiP=14, kLamPiM=15, kALamPiM=16};
  enum BFieldType {kFemtoPlus=0, kFemtoMinus=1, kBp1=2, kBp2=3, kBm1=4, kBm2=5, kBm3=6};
  enum CentralityType {k0010=0, k1030=1, k3050=2, kMB=3};
  enum ParticleType {kLam=0, kALam=1, kK0=2, kKchP=3, kKchM=4, kXi=5, kAXi=6, kPiP=7, kPiM=8, kProton=9, kAntiProton=10};

  enum DaughterPairType {kPosPos = 0, kPosNeg = 1, kNegPos = 2, kNegNeg = 3, kTrackPos = 4, kTrackNeg = 5, kTrackTrack = 6, kV0Track=7, kMaxNDaughterPairTypes = 8};

  enum AxisType {kXaxis=0, kYaxis=1};

  enum ParameterType {kLambda=0, kRadius=1, kRef0=2, kImf0=3, kd0=4, kRef02=5, kImf02=6, kd02=7, kNorm=8};

  enum KStarTrueVsRecType {kSame=0, kRotSame=1, kMixed=2, kRotMixed=3};

  enum FitType {kChi2PML=0, kChi2=1};
  enum FitGeneratorType {kPair=0, kConjPair=1, kPairwConj=2};
  enum NonFlatBgdFitType {kLinear=0, kQuadratic=1, kGaussian=2};

  enum InterpType {kGTilde=0, kHyperGeo1F1=1, kScattLen=2};
  enum InterpAxisType {kKaxis=0, kRaxis=1, kThetaaxis=2, kReF0axis=3, kImF0axis=4, kD0axis=5};

  enum GeneralFitterType {kLedEq=0, kLedViaSimPairs=1, kCoulomb=2};

  enum CoulombType {kAttractive=0, kRepulsive=1, kNeutral=2};

  enum ResidualType {kSig0KchP=0, kASig0KchP=1, kSig0KchM=2, kASig0KchM=3, kXi0KchP=4, kAXi0KchP=5, kXi0KchM=6, kAXi0KchM=7, kXiCKchP=8, kAXiCKchP=9, kXiCKchM=10, kAXiCKchM=11, kOmegaKchP=12, kAOmegaKchP=13, kOmegaKchM=14, kAOmegaKchM=15}; 

//----------------------------------------------------------
  //TODO Don't forget, when adding new particle, add int value to cPDGValues array
  enum ParticlePDGType {kPDGProt   = 2212,  kPDGAntiProt = -2212, 
		        kPDGPiP    = 211,   kPDGPiM      = -211, 
                        kPDGK0     = 311,
                        kPDGKchP   = 321,   kPDGKchM     = -321,
		        kPDGLam    = 3122,  kPDGALam     = -3122,
		        kPDGSigma  = 3212,  kPDGASigma   = -3212,
		        kPDGXiC    = 3312,  kPDGAXiC     = -3312,
		        kPDGXi0    = 3322,  kPDGAXi0     = -3322,
		        kPDGOmega  = 3334,  kPDGAOmega   = -3334,
                        kPDGNull      = 0                          };
  extern const int cPDGValues[18];
  //----------------------------------------------------------


  //-------Struct definitions---------------------------------
  struct BinInfo
  {
    int nBins;
    double minVal, maxVal, binWidth;
  };

  struct BinInfoKStar
  {
    int nBinsK;
    double minK, maxK, binWidthK;
    int nPairsPerBin[100] = {};  //TODO for now, make this bigger than I'll ever need
    int binOffset[100] = {};
  };

  struct BinInfoSamplePairs
  {
    int nAnalyses;
    int nBinsK;
    int nPairsPerBin;
    double minK, maxK, binWidthK;
    int nElementsPerPair;
  };

  //------------------
  struct BinInfoGTilde
  {
    int nBinsK, nBinsR;
    double binWidthK, binWidthR;
    double minK, maxK, minR, maxR;
    double minInterpK, maxInterpK, minInterpR, maxInterpR;
  };

//------------------
  struct BinInfoHyperGeo1F1
  {
    int nBinsK, nBinsR, nBinsTheta;
    double binWidthK, binWidthR, binWidthTheta;
    double minK, maxK, minR, maxR, minTheta, maxTheta;
    double minInterpK, maxInterpK, minInterpR, maxInterpR, minInterpTheta, maxInterpTheta;
  };

//------------------
  struct BinInfoScattLen
  {
    int nBinsReF0, nBinsImF0, nBinsD0, nBinsK;
    double binWidthReF0, binWidthImF0, binWidthD0, binWidthK;
    double minReF0, maxReF0, minImF0, maxImF0, minD0, maxD0, minK, maxK;
    double minInterpReF0, maxInterpReF0, minInterpImF0, maxInterpImF0, minInterpD0, maxInterpD0, minInterpK, maxInterpK;
  };

//------------------
  struct SystematicsFileInfo
  {
    const char* resultsDate;
    const char* dirNameModifierBase1;
    const char* dirNameModifierBase2;
    vector<double> modifierValues1;
    vector<double> modifierValues2;
    bool allCentralities;
  };

  //----------------------------------------------------------

  //--------------typedefs------------------------------------------------
  typedef vector<double> td1dVec;
  typedef vector<vector<double> > td2dVec;
  typedef vector<vector<vector<double> > > td3dVec;
  typedef vector<vector<vector<vector<double> > > > td4dVec;
  //----------------------------------------------------------------------


  //-------------------------------------------------
  extern const char* const cAnalysisBaseTags[17];
  extern const char* const cAnalysisRootTags[17];
  extern const char* const cBFieldTags[7];
  extern const char* const cCentralityTags[4];
  extern const char* const cPrettyCentralityTags[4];
  extern const char* const cParticleTags[11];
  extern const char* const cRootParticleTags[11];
  extern const char* const cKStarTrueVsRecTypeTags[4];

  extern const char* cKStarCfBaseTagNum;
  extern const char* cKStarCfBaseTagDen;

  extern const char* cKStarCfMCTrueBaseTagNum;
  extern const char* cKStarCfMCTrueBaseTagDen;

  extern const char* cKStar2dCfBaseTagNum;
  extern const char* cKStar2dCfBaseTagDen;

    //
  extern const char* cModelKStarCfNumTrueBaseTag;
  extern const char* cModelKStarCfDenBaseTag;

  extern const char* cModelKStarCfNumTrueIdealBaseTag;
  extern const char* cModelKStarCfDenIdealBaseTag;

  extern const char* cModelKStarCfNumTrueUnitWeightsBaseTag;
  extern const char* cModelKStarCfNumTrueIdealUnitWeightsBaseTag;

  extern const char* cModelKStarCfNumFakeBaseTag;
  extern const char* cModelKStarCfNumFakeIdealBaseTag;

//  extern const char* cModelKStarTrueVsRecBaseTag;
  extern const char* cModelKStarTrueVsRecSameBaseTag;
  extern const char* cModelKStarTrueVsRecRotSameBaseTag;
  extern const char* cModelKStarTrueVsRecMixedBaseTag;
  extern const char* cModelKStarTrueVsRecRotMixedBaseTag;

    //

  extern const char* cLambdaPurityTag;
  extern const char* cAntiLambdaPurityTag;
  extern const char* cK0ShortPurityTag;
  extern const char* cXiPurityTag;
  extern const char* cAXiPurityTag;

  
  extern const char* const cDaughterPairTags[7];

  extern const char* const cAvgSepCfBaseTagsNum[7];
  extern const char* const cAvgSepCfBaseTagsDen[7];

  extern const char* const cSepCfsBaseTagsNum[7];
  extern const char* const cSepCfsBaseTagsDen[7];

  extern const char* const cAvgSepCfCowboysAndSailorsBaseTagsNum[7];
  extern const char* const cAvgSepCfCowboysAndSailorsBaseTagsDen[7];
  //-------------------------------------------------

  //--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**//
  //--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**//

  extern const char* const cParameterNames[9];

  //-------------------------------------------------
  extern const double cLamK0_0010StartValues[6];
  extern const double cLamK0_1030StartValues[6];
  extern const double cLamK0_3050StartValues[6];
  extern const double *cLamK0StartValues[3];

  extern const double cALamK0_0010StartValues[6];
  extern const double cALamK0_1030StartValues[6];
  extern const double cALamK0_3050StartValues[6];
  extern const double *cALamK0StartValues[3];

  //-----

  extern const double cLamKchP_0010StartValues[6];
  extern const double cLamKchP_1030StartValues[6];
  extern const double cLamKchP_3050StartValues[6];
  extern const double *cLamKchPStartValues[3];

  extern const double cALamKchM_0010StartValues[6];
  extern const double cALamKchM_1030StartValues[6];
  extern const double cALamKchM_3050StartValues[6];
  extern const double *cALamKchMStartValues[3];

  extern const double cLamKchM_0010StartValues[6];
  extern const double cLamKchM_1030StartValues[6];
  extern const double cLamKchM_3050StartValues[6];
  extern const double *cLamKchMStartValues[3];

  extern const double cALamKchP_0010StartValues[6];
  extern const double cALamKchP_1030StartValues[6];
  extern const double cALamKchP_3050StartValues[6];
  extern const double *cALamKchPStartValues[3];

  //-----

  extern const double cXiKchP_0010StartValues[6];
  extern const double cXiKchP_1030StartValues[6];
  extern const double cXiKchP_3050StartValues[6];
  extern const double *cXiKchPStartValues[3];

  extern const double cAXiKchP_0010StartValues[6];
  extern const double cAXiKchP_1030StartValues[6];
  extern const double cAXiKchP_3050StartValues[6];
  extern const double *cAXiKchPStartValues[3];

  extern const double cXiKchM_0010StartValues[6];
  extern const double cXiKchM_1030StartValues[6];
  extern const double cXiKchM_3050StartValues[6];
  extern const double *cXiKchMStartValues[3];

  extern const double cAXiKchM_0010StartValues[6];
  extern const double cAXiKchM_1030StartValues[6];
  extern const double cAXiKchM_3050StartValues[6];
  extern const double *cAXiKchMStartValues[3];

  //-----

  extern const double **cStartValues[10];  // = fStartValues[6][3][6]
                                          //first index = AnalysisType
                                          //second index = CentralityType
                                          //third index = Parameter value

  //-----
  extern const double hbarc;
  extern const double gBohrRadiusXiK;
  extern const double gBohrRadiusOmegaK;

  static const std::complex<double> ImI (0.,1.);


#endif

