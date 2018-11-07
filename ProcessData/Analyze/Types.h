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
#include <cassert>
#include <iostream>
using std::vector;
using namespace std;

  //-------enum types-----------------------------------------
  enum PartnerAnalysisType {kcLamK0=0, kcLamcKch=1, kcXicKch=2, kcLamcLam=3, kcLamcPi=4};

  enum AnalysisRunType {kTrain=0, kTrainSys=1, kGrid=2};

  enum AnalysisType {
kLamK0=0, kALamK0=1, 
kLamKchP=2, kALamKchM=3, kLamKchM=4, kALamKchP=5, 
kXiKchP=6, kAXiKchM=7, kXiKchM=8, kAXiKchP=9, 
kXiK0=10, kAXiK0=11, 
kLamLam=12, kALamALam=13, kLamALam=14, 
kLamPiP=15, kALamPiM=16, kLamPiM=17, kALamPiP=18, 

//----- Residual Types -----
kResSig0KchP=19, kResASig0KchM=20, kResSig0KchM=21, kResASig0KchP=22, 
kResXi0KchP=23, kResAXi0KchM=24, kResXi0KchM=25, kResAXi0KchP=26, 
kResXiCKchP=27, kResAXiCKchM=28, kResXiCKchM=29, kResAXiCKchP=30, 
kResOmegaKchP=31, kResAOmegaKchM=32, kResOmegaKchM=33, kResAOmegaKchP=34, 
kResSigStPKchP=35, kResASigStMKchM=36, kResSigStPKchM=37, kResASigStMKchP=38, 
kResSigStMKchP=39, kResASigStPKchM=40, kResSigStMKchM=41, kResASigStPKchP=42,
kResSigSt0KchP=43, kResASigSt0KchM=44, kResSigSt0KchM=45, kResASigSt0KchP=46,

kResLamKSt0=47, kResALamAKSt0=48, kResLamAKSt0=49, kResALamKSt0=50,
kResSig0KSt0=51, kResASig0AKSt0=52, kResSig0AKSt0=53, kResASig0KSt0=54, 
kResXi0KSt0=55, kResAXi0AKSt0=56, kResXi0AKSt0=57, kResAXi0KSt0=58, 
kResXiCKSt0=59, kResAXiCAKSt0=60, kResXiCAKSt0=61, kResAXiCKSt0=62,

kResSig0K0=63, kResASig0K0=64, 
kResXi0K0=65, kResAXi0K0=66, 
kResXiCK0=67, kResAXiCK0=68, 
kResOmegaK0=69, kResAOmegaK0=70, 
kResSigStPK0=71, kResASigStMK0=72, 
kResSigStMK0=73, kResASigStPK0=74,
kResSigSt0K0=75, kResASigSt0K0=76,

kResLamKSt0ToLamK0=77, kResALamKSt0ToALamK0=78,
kResSig0KSt0ToLamK0=79, kResASig0KSt0ToALamK0=80, 
kResXi0KSt0ToLamK0=81, kResAXi0KSt0ToALamK0=82, 
kResXiCKSt0ToLamK0=83, kResAXiCKSt0ToALamK0=84,
kKchPKchP=85, kK0K0=86
};

  enum BFieldType {kFemtoPlus=0, kFemtoMinus=1, kBp1=2, kBp2=3, kBm1=4, kBm2=5, kBm3=6};
  enum CentralityType {k0010=0, k1030=1, k3050=2, kMB=3};

  enum DaughterPairType {kPosPos = 0, kPosNeg = 1, kNegPos = 2, kNegNeg = 3, kTrackPos = 4, kTrackNeg = 5, kTrackTrack = 6, kTrackBac=7, kBacPos = 8, kBacNeg = 9, kMaxNDaughterPairTypes = 10};

  enum AxisType {kXaxis=0, kYaxis=1};

  enum ParameterType {kLambda=0, kRadius=1, kRef0=2, kImf0=3, kd0=4, kRef02=5, kImf02=6, kd02=7, kNorm=8, kBgd=9};
  enum ParamValueType {kValue=0, kStatErr=1, kSystErr=2};

  enum KStarTrueVsRecType {kSame=0, kRotSame=1, kMixed=2, kRotMixed=3};

  enum FitType {kChi2PML=0, kChi2=1};
  enum FitGeneratorType {kPair=0, kConjPair=1, kPairwConj=2};
  enum NonFlatBgdFitType {kLinear=0, kQuadratic=1, kGaussian=2, kPolynomial=3, kDivideByTherm=4};
  enum ThermEventsType {kMe=0, kAdam=1, kMeAndAdam=2};

  enum IncludeResidualsType {kIncludeNoResiduals=0, kInclude10Residuals=1, kInclude3Residuals=2};
  enum ChargedResidualsType {kUseXiDataForAll=0, kUseXiDataAndCoulombOnlyInterp=1, kUseCoulombOnlyInterpForAll=2};
  enum ResPrimMaxDecayType {k0fm=0, k4fm=1, k5fm=2, k6fm=3, k10fm=4, k100fm=5};

  enum InterpType {kGTilde=0, kHyperGeo1F1=1, kScattLen=2};
  enum InterpAxisType {kKaxis=0, kRaxis=1, kThetaaxis=2, kReF0axis=3, kImF0axis=4, kD0axis=5};

  enum GeneralFitterType {kLedEq=0, kLedViaSimPairs=1, kCoulomb=2};

  enum CoulombType {kAttractive=0, kRepulsive=1, kNeutral=2};
/*
  enum ResidualType {
kSig0KchP=0, kASig0KchM=1, kSig0KchM=2, kASig0KchP=3, 
kXi0KchP=4, kAXi0KchM=5, kXi0KchM=6, kAXi0KchP=7, 
kXiCKchP=8, kAXiCKchM=9, kXiCKchM=10, kAXiCKchP=11, 
kOmegaKchP=12, kAOmegaKchM=13, kOmegaKchM=14, kAOmegaKchP=15, 
kSigStPKchP=16, kASigStMKchM=17, kSigStPKchM=18, kASigStMKchP=19, 
kSigStMKchP=20, kASigStPKchM=21, kSigStMKchM=22, kASigStPKchP=23,
kSigSt0KchP=24, kASigSt0KchM=25, kSigSt0KchM=26, kASigSt0KchP=27,
kLamKSt0=28, kALamAKSt0=29, kLamAKSt0=30, kALamKSt0=31,
kSig0KSt0=32, kASig0AKSt0=33, kSig0AKSt0=34, kASig0KSt0=35, 
kXi0KSt0=36, kAXi0AKSt0=37, kXi0AKSt0=38, kAXi0KSt0=39, 
kXiCKSt0=40, kAXiCAKSt0=41, kXiCAKSt0=42, kAXiCKSt0=43
}; 
*/
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

			kPDGSigStP = 3224,  kPDGASigStM  = -3224,
			kPDGSigStM = 3114,  kPDGASigStP  = -3114,
			kPDGSigSt0 = 3214,  kPDGASigSt0  = -3214,
			kPDGKSt0   = 313,   kPDGAKSt0    = -313,
                        kPDGNull      = 0                          };
  extern const int cPDGValues[26];
  extern const char* const cPDGRootNames[26];
  extern const char* const GetPDGRootName(ParticlePDGType aType);
  extern vector<ParticlePDGType> GetResidualDaughtersAndMothers(AnalysisType aResidualType);

  enum ParticleType {kProton     = 0,  kAntiProton     = 1,
                     kPiP        = 2,  kPiM            = 3,
                     kK0         = 4,
                     kKchP       = 5,  kKchM           = 6,
                     kLam        = 7,  kALam           = 8,
                     kSig0       = 9,  kASig0          = 10,
                     kXi         = 11, kAXi            = 12,
                     kXi0        = 13, kAXi0           = 14,
                     kOmega      = 15, kAOmega         = 16,

                     kSigStP     = 17, kASigStM        = 18,
                     kSigStM     = 19, kASigStP        = 20,
                     kSigSt0     = 21, kASigSt0        = 22,
                     kKSt0       = 23, kAKSt0          = 24};  
  extern const char* const cParticleTags[25];
  extern const char* const cRootParticleTags[25]; 
  //----------------------------------------------------------


  //-------Struct definitions---------------------------------
  struct TransformInfo
  {
    ParticlePDGType particleType1;
    ParticlePDGType parentType1;

    ParticlePDGType particleType2;
    ParticlePDGType parentType2;

    TransformInfo(ParticlePDGType aParticleType1, ParticlePDGType aParticleType2, 
                  ParticlePDGType aParentType1  , ParticlePDGType aParentType2  )
    {
      particleType1 = aParticleType1;
      parentType1   = aParentType1;

      particleType2 = aParticleType2;
      parentType2   = aParentType2;

    }
  };


  struct tmpAnalysisInfo
  {
    AnalysisType analysisType;
    CentralityType centralityType;
  };

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
    vector<double> modifierValues1;

    const char* dirNameModifierBase2;
    vector<double> modifierValues2;

    bool allCentralities;

    SystematicsFileInfo(const char* aResultDate, 
                        const char* aDirNameModifierBase1, const vector<double> &aModifierValues1, 
                        const char* aDirNameModifierBase2, const vector<double> &aModifierValues2, 
                        bool aAllCentralities)
    {
      resultsDate = aResultDate;

      dirNameModifierBase1 = aDirNameModifierBase1;
      modifierValues1 = aModifierValues1;

      dirNameModifierBase2 = aDirNameModifierBase2;
      modifierValues2 = aModifierValues2;

      allCentralities = aAllCentralities;
    }
  };

//------------------
  struct ParamOwnerInfo
  {
    AnalysisType analysisType;
    CentralityType centralityType;
    BFieldType bFieldType;

    ParamOwnerInfo(AnalysisType aAnType, CentralityType aCentType, BFieldType aBFieldType)
    {
      analysisType   = aAnType;
      centralityType = aCentType;
      bFieldType     = aBFieldType;
    }

    ParamOwnerInfo()
    {
      analysisType   = kLamK0;
      centralityType = k0010;
      bFieldType     = kFemtoPlus;
    }
  };

  //----------------------------------------------------------

  //--------------typedefs------------------------------------------------
  typedef vector<double> td1dVec;
  typedef vector<vector<double> > td2dVec;
  typedef vector<vector<vector<double> > > td3dVec;
  typedef vector<vector<vector<vector<double> > > > td4dVec;
  //----------------------------------------------------------------------


  //-------------------------------------------------
  extern const char* const cAnalysisBaseTags[87];
  extern const char* const cAnalysisRootTags[87];
  extern const double cAnalysisLambdaFactors[85];
  extern const double cAnalysisLambdaFactors2[85];  //NOTE if I want to use this, switch name with cAnalysisLambdaFactors
  extern const double cAnalysisLambdaFactorsOLD[85];  //NOTE if I want to use this, switch name with cAnalysisLambdaFactors
  extern const double cmTFactorsFromTherminator[85];
//  extern const char* const cResidualBaseTags[44];
//  extern const char* const cResidualRootTags[44];

  extern const char* const cBFieldTags[7];
  extern const char* const cCentralityTags[4];
  extern const char* const cPrettyCentralityTags[4];
  extern const char* const cKStarTrueVsRecTypeTags[4];

  extern const char* const cNonFlatBgdFitTypeTags[5];
  extern const char* const cThermEventsTypeTags[3];

  extern const char* const cIncludeResidualsTypeTags[3];
  extern const char* const cChargedResidualsTypeTags[3];
  extern const char* const cResPrimMaxDecayTypeTags[6];

  extern const char* cKStarCfBaseTagNum;
  extern const char* cKStarCfBaseTagDen;
  extern const char* cKStarCfBaseTagNumRotatePar2;

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

  
  extern const char* const cDaughterPairTags[10];

  extern const char* const cAvgSepCfBaseTagsNum[8];
  extern const char* const cAvgSepCfBaseTagsDen[8];

  extern const char* const cSepCfsBaseTagsNum[8];
  extern const char* const cSepCfsBaseTagsDen[8];

  extern const char* const cAvgSepCfCowboysAndSailorsBaseTagsNum[8];
  extern const char* const cAvgSepCfCowboysAndSailorsBaseTagsDen[8];
  //-------------------------------------------------

  //--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**//
  //--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**//

  extern const char* const cParameterNames[10];

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

  extern const double **cStartValues[10];  // = fStartValues[10][3][6]
                                          //first index = AnalysisType
                                          //second index = CentralityType
                                          //third index = Parameter value


  //-----
  extern const double hbarc;
  extern const double gBohrRadiusXiK;
  extern const double gBohrRadiusOmegaK;
  extern const double gBohrRadiusSigStPK;
  extern const double gBohrRadiusSigStMK;

  static const std::complex<double> ImI (0.,1.);


#endif

