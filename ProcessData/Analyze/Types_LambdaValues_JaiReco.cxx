///////////////////////////////////////////////////////////////////////////
// Types_LambdaValues.cxx:                                               //
///////////////////////////////////////////////////////////////////////////


#include "Types_LambdaValues.h"
#include <cassert>

//NOTE:  Changing MaxDecayLengthPrimary only changes the primary lambda values
//       (and the "other" lambda values).
//       i.e. for a given number of residuals included, only the first 6 elements
//       of the arrays will differ


//------------ WHEN INCLUDING NO RESIDUALS ------------------------------------//
const double cAnalysisLambdaFactors_NoRes_MaxPrimDecay0[85] = {
/*kLamK0=*/   1.000,   /*kALamK0=*/   1.000,
/*kLamKchP=*/ 1.000,   /*kALamKchM=*/ 1.000,  /*kLamKchM=*/ 1.000, /*kALamKchP=*/ 1.000,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.000, /*kResASig0KchM=*/   0.000, /*kResSig0KchM=*/   0.000, /*kResASig0KchP=*/  0.000,
/*kResXi0KchP=*/    0.000, /*kResAXi0KchM=*/    0.000, /*kResXi0KchM=*/    0.000, /*kResAXi0KchP=*/   0.000,
/*kResXiCKchP=*/    0.000, /*kResAXiCKchM=*/    0.000, /*kResXiCKchM=*/    0.000, /*kResAXiCKchP=*/   0.000,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.000, /*kResASigStMKchM=*/ 0.000, /*kResSigStPKchM=*/ 0.000, /*kResASigStMKchP=*/ 0.000,
/*kResSigStMKchP=*/ 0.000, /*kResASigStPKchM=*/ 0.000, /*kResSigStMKchM=*/ 0.000, /*kResASigStPKchP=*/ 0.000,
/*kResSigSt0KchP=*/ 0.000, /*kResASigSt0KchM=*/ 0.000, /*kResSigSt0KchM=*/ 0.000, /*kResASigSt0KchP=*/ 0.000,

/*kResLamKSt0=*/  0.000, /*kResALamAKSt0=*/  0.000, /*kResLamAKSt0=*/  0.000, /*kResALamKSt0=*/  0.000,
/*kResSig0KSt0=*/ 0.000, /*kResASig0AKSt0=*/ 0.000, /*kResSig0AKSt0=*/ 0.000, /*kResASig0KSt0=*/ 0.000,
/*kResXi0KSt0=*/  0.000, /*kResAXi0AKSt0=*/  0.000, /*kResXi0AKSt0=*/  0.000, /*kResAXi0KSt0=*/  0.000,
/*kResXiCKSt0=*/  0.000, /*kResAXiCAKSt0=*/  0.000, /*kResXiCAKSt0=*/  0.000, /*kResAXiCKSt0=*/  0.000,

/*kResSig0K0=*/   0.000, /*kResASig0K0=*/   0.000,
/*kResXi0K0=*/    0.000, /*kResAXi0K0=*/    0.000,
/*kResXiCK0=*/    0.000, /*kResAXiCK0=*/    0.000,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.000, /*kResASigStMK0=*/ 0.000,
/*kResSigStMK0=*/ 0.000, /*kResASigStPK0=*/ 0.000,
/*kResSigSt0K0=*/ 0.000, /*kResASigSt0K0=*/ 0.000,

/*kResLamKSt0ToLamK0=*/  0.000, /*kResALamAKSt0ToALamK0=*/  0.000,
/*kResSig0KSt0ToLamK0=*/ 0.000, /*kResASig0AKSt0ToALamK0=*/ 0.000,
/*kResXi0KSt0ToLamK0=*/  0.000, /*kResAXi0AKSt0ToALamK0=*/  0.000,
/*kResXiCKSt0ToLamK0=*/  0.000, /*kResAXiCAKSt0ToALamK0=*/  0.000
};

const double cAnalysisLambdaFactors_NoRes_MaxPrimDecay4[85] = {
0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
0.,0.,0.,0.,0.
};

const double cAnalysisLambdaFactors_NoRes_MaxPrimDecay5[85] = {
0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
0.,0.,0.,0.,0.
};

const double cAnalysisLambdaFactors_NoRes_MaxPrimDecay6[85] = {
0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
0.,0.,0.,0.,0.
};

const double cAnalysisLambdaFactors_NoRes_MaxPrimDecay10[85] = {
0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
0.,0.,0.,0.,0.
};

const double cAnalysisLambdaFactors_NoRes_MaxPrimDecay100[85] = {
0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
0.,0.,0.,0.,0.
};


const double *cAnalysisLambdaFactors_NoRes[6] = {cAnalysisLambdaFactors_NoRes_MaxPrimDecay0, 
                                                 cAnalysisLambdaFactors_NoRes_MaxPrimDecay4,
                                                 cAnalysisLambdaFactors_NoRes_MaxPrimDecay5,
                                                 cAnalysisLambdaFactors_NoRes_MaxPrimDecay6,
                                                 cAnalysisLambdaFactors_NoRes_MaxPrimDecay10,
                                                 cAnalysisLambdaFactors_NoRes_MaxPrimDecay100};

//----------------------------------------------------------------------------------------------------------------
//****************************************************************************************************************
//----------------------------------------------------------------------------------------------------------------


//------------ WHEN INCLUDING ALL 10 RESIDUALS and MaxDecayLengthPrimary=0 ------------------------------------//
const double cAnalysisLambdaFactors_10Res_MaxPrimDecay0[85] = {
/*kLamK0=*/   0.126,   /*kALamK0=*/   0.128,
/*kLamKchP=*/ 0.117,   /*kALamKchM=*/ 0.118,  /*kLamKchM=*/ 0.116, /*kALamKchP=*/ 0.119,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.105, /*kResASig0KchM=*/   0.105, /*kResSig0KchM=*/   0.104, /*kResASig0KchP=*/  0.106,
/*kResXi0KchP=*/    0.047, /*kResAXi0KchM=*/    0.041, /*kResXi0KchM=*/    0.047, /*kResAXi0KchP=*/   0.042,
/*kResXiCKchP=*/    0.064, /*kResAXiCKchM=*/    0.058, /*kResXiCKchM=*/    0.063, /*kResAXiCKchP=*/   0.059,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.049, /*kResASigStMKchM=*/ 0.047, /*kResSigStPKchM=*/ 0.048, /*kResASigStMKchP=*/ 0.048,
/*kResSigStMKchP=*/ 0.044, /*kResASigStPKchM=*/ 0.046, /*kResSigStMKchM=*/ 0.044, /*kResASigStPKchP=*/ 0.046,
/*kResSigSt0KchP=*/ 0.044, /*kResASigSt0KchM=*/ 0.041, /*kResSigSt0KchM=*/ 0.044, /*kResASigSt0KchP=*/ 0.042,

/*kResLamKSt0=*/  0.041, /*kResALamAKSt0=*/  0.042, /*kResLamAKSt0=*/  0.041, /*kResALamKSt0=*/  0.042,
/*kResSig0KSt0=*/ 0.037, /*kResASig0AKSt0=*/ 0.037, /*kResSig0AKSt0=*/ 0.037, /*kResASig0KSt0=*/ 0.037,
/*kResXi0KSt0=*/  0.017, /*kResAXi0AKSt0=*/  0.015, /*kResXi0AKSt0=*/  0.017, /*kResAXi0KSt0=*/  0.015,
/*kResXiCKSt0=*/  0.022, /*kResAXiCAKSt0=*/  0.021, /*kResXiCAKSt0=*/  0.022, /*kResAXiCKSt0=*/  0.021,

/*kResSig0K0=*/   0.113, /*kResASig0K0=*/   0.114,
/*kResXi0K0=*/    0.051, /*kResAXi0K0=*/    0.045,
/*kResXiCK0=*/    0.069, /*kResAXiCK0=*/    0.063,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.052, /*kResASigStMK0=*/ 0.051,
/*kResSigStMK0=*/ 0.047, /*kResASigStPK0=*/ 0.050,
/*kResSigSt0K0=*/ 0.047, /*kResASigSt0K0=*/ 0.045,

/*kResLamKSt0ToLamK0=*/  0.020, /*kResALamAKSt0ToALamK0=*/  0.020,
/*kResSig0KSt0ToLamK0=*/ 0.017, /*kResASig0AKSt0ToALamK0=*/ 0.018,
/*kResXi0KSt0ToLamK0=*/  0.008, /*kResAXi0AKSt0ToALamK0=*/  0.007,
/*kResXiCKSt0ToLamK0=*/  0.011, /*kResAXiCAKSt0ToALamK0=*/  0.010
};


//------------ WHEN INCLUDING ALL 10 RESIDUALS and MaxDecayLengthPrimary=4 ------------------------------------//
const double cAnalysisLambdaFactors_10Res_MaxPrimDecay4[85] = {
/*kLamK0=*/   0.173,   /*kALamK0=*/   0.173,
/*kLamKchP=*/ 0.163,   /*kALamKchM=*/ 0.163,  /*kLamKchM=*/ 0.162, /*kALamKchP=*/ 0.163,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.105, /*kResASig0KchM=*/   0.105, /*kResSig0KchM=*/   0.104, /*kResASig0KchP=*/  0.106,
/*kResXi0KchP=*/    0.047, /*kResAXi0KchM=*/    0.042, /*kResXi0KchM=*/    0.047, /*kResAXi0KchP=*/   0.042,
/*kResXiCKchP=*/    0.064, /*kResAXiCKchM=*/    0.058, /*kResXiCKchM=*/    0.063, /*kResAXiCKchP=*/   0.059,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.049, /*kResASigStMKchM=*/ 0.047, /*kResSigStPKchM=*/ 0.048, /*kResASigStMKchP=*/ 0.048,
/*kResSigStMKchP=*/ 0.044, /*kResASigStPKchM=*/ 0.046, /*kResSigStMKchM=*/ 0.044, /*kResASigStPKchP=*/ 0.047,
/*kResSigSt0KchP=*/ 0.044, /*kResASigSt0KchM=*/ 0.041, /*kResSigSt0KchM=*/ 0.044, /*kResASigSt0KchP=*/ 0.042,

/*kResLamKSt0=*/  0.041, /*kResALamAKSt0=*/  0.042, /*kResLamAKSt0=*/  0.041, /*kResALamKSt0=*/  0.042,
/*kResSig0KSt0=*/ 0.037, /*kResASig0AKSt0=*/ 0.037, /*kResSig0AKSt0=*/ 0.037, /*kResASig0KSt0=*/ 0.037,
/*kResXi0KSt0=*/  0.017, /*kResAXi0AKSt0=*/  0.015, /*kResXi0AKSt0=*/  0.017, /*kResAXi0KSt0=*/  0.015,
/*kResXiCKSt0=*/  0.022, /*kResAXiCAKSt0=*/  0.021, /*kResXiCAKSt0=*/  0.022, /*kResAXiCKSt0=*/  0.021,

/*kResSig0K0=*/   0.113, /*kResASig0K0=*/   0.114,
/*kResXi0K0=*/    0.051, /*kResAXi0K0=*/    0.045,
/*kResXiCK0=*/    0.069, /*kResAXiCK0=*/    0.063,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.052, /*kResASigStMK0=*/ 0.051,
/*kResSigStMK0=*/ 0.047, /*kResASigStPK0=*/ 0.050,
/*kResSigSt0K0=*/ 0.047, /*kResASigSt0K0=*/ 0.045,

/*kResLamKSt0ToLamK0=*/  0.020, /*kResALamAKSt0ToALamK0=*/  0.020,
/*kResSig0KSt0ToLamK0=*/ 0.017, /*kResASig0AKSt0ToALamK0=*/ 0.018,
/*kResXi0KSt0ToLamK0=*/  0.008, /*kResAXi0AKSt0ToALamK0=*/  0.007,
/*kResXiCKSt0ToLamK0=*/  0.011, /*kResAXiCAKSt0ToALamK0=*/  0.010
};

//------------ WHEN INCLUDING ALL 10 RESIDUALS and MaxDecayLengthPrimary=5 ------------------------------------//
const double cAnalysisLambdaFactors_10Res_MaxPrimDecay5[85] = {
/*kLamK0=*/   0.227,   /*kALamK0=*/   0.228,
/*kLamKchP=*/ 0.198,   /*kALamKchM=*/ 0.199,  /*kLamKchM=*/ 0.198, /*kALamKchP=*/ 0.199,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.105, /*kResASig0KchM=*/   0.105, /*kResSig0KchM=*/   0.104, /*kResASig0KchP=*/  0.106,
/*kResXi0KchP=*/    0.047, /*kResAXi0KchM=*/    0.042, /*kResXi0KchM=*/    0.047, /*kResAXi0KchP=*/   0.042,
/*kResXiCKchP=*/    0.064, /*kResAXiCKchM=*/    0.058, /*kResXiCKchM=*/    0.063, /*kResAXiCKchP=*/   0.059,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.049, /*kResASigStMKchM=*/ 0.047, /*kResSigStPKchM=*/ 0.048, /*kResASigStMKchP=*/ 0.048,
/*kResSigStMKchP=*/ 0.044, /*kResASigStPKchM=*/ 0.046, /*kResSigStMKchM=*/ 0.044, /*kResASigStPKchP=*/ 0.047,
/*kResSigSt0KchP=*/ 0.044, /*kResASigSt0KchM=*/ 0.041, /*kResSigSt0KchM=*/ 0.044, /*kResASigSt0KchP=*/ 0.042,

/*kResLamKSt0=*/  0.041, /*kResALamAKSt0=*/  0.042, /*kResLamAKSt0=*/  0.041, /*kResALamKSt0=*/  0.042,
/*kResSig0KSt0=*/ 0.037, /*kResASig0AKSt0=*/ 0.037, /*kResSig0AKSt0=*/ 0.037, /*kResASig0KSt0=*/ 0.037,
/*kResXi0KSt0=*/  0.017, /*kResAXi0AKSt0=*/  0.015, /*kResXi0AKSt0=*/  0.017, /*kResAXi0KSt0=*/  0.015,
/*kResXiCKSt0=*/  0.022, /*kResAXiCAKSt0=*/  0.021, /*kResXiCAKSt0=*/  0.022, /*kResAXiCKSt0=*/  0.021,

/*kResSig0K0=*/   0.113, /*kResASig0K0=*/   0.114,
/*kResXi0K0=*/    0.051, /*kResAXi0K0=*/    0.045,
/*kResXiCK0=*/    0.069, /*kResAXiCK0=*/    0.063,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.052, /*kResASigStMK0=*/ 0.051,
/*kResSigStMK0=*/ 0.047, /*kResASigStPK0=*/ 0.050,
/*kResSigSt0K0=*/ 0.047, /*kResASigSt0K0=*/ 0.045,

/*kResLamKSt0ToLamK0=*/  0.020, /*kResALamAKSt0ToALamK0=*/  0.020,
/*kResSig0KSt0ToLamK0=*/ 0.017, /*kResASig0AKSt0ToALamK0=*/ 0.018,
/*kResXi0KSt0ToLamK0=*/  0.008, /*kResAXi0AKSt0ToALamK0=*/  0.007,
/*kResXiCKSt0ToLamK0=*/  0.011, /*kResAXiCAKSt0ToALamK0=*/  0.010
};


//------------ WHEN INCLUDING ALL 10 RESIDUALS and MaxDecayLengthPrimary=6 ------------------------------------//
const double cAnalysisLambdaFactors_10Res_MaxPrimDecay6[85] = {
/*kLamK0=*/   0.316,   /*kALamK0=*/   0.319,
/*kLamKchP=*/ 0.291,   /*kALamKchM=*/ 0.294,  /*kLamKchM=*/ 0.292, /*kALamKchP=*/ 0.294,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.105, /*kResASig0KchM=*/   0.105, /*kResSig0KchM=*/   0.104, /*kResASig0KchP=*/  0.106,
/*kResXi0KchP=*/    0.047, /*kResAXi0KchM=*/    0.042, /*kResXi0KchM=*/    0.047, /*kResAXi0KchP=*/   0.042,
/*kResXiCKchP=*/    0.064, /*kResAXiCKchM=*/    0.058, /*kResXiCKchM=*/    0.063, /*kResAXiCKchP=*/   0.059,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.049, /*kResASigStMKchM=*/ 0.047, /*kResSigStPKchM=*/ 0.048, /*kResASigStMKchP=*/ 0.048,
/*kResSigStMKchP=*/ 0.044, /*kResASigStPKchM=*/ 0.046, /*kResSigStMKchM=*/ 0.044, /*kResASigStPKchP=*/ 0.047,
/*kResSigSt0KchP=*/ 0.044, /*kResASigSt0KchM=*/ 0.041, /*kResSigSt0KchM=*/ 0.044, /*kResASigSt0KchP=*/ 0.042,

/*kResLamKSt0=*/  0.041, /*kResALamAKSt0=*/  0.042, /*kResLamAKSt0=*/  0.041, /*kResALamKSt0=*/  0.042,
/*kResSig0KSt0=*/ 0.037, /*kResASig0AKSt0=*/ 0.037, /*kResSig0AKSt0=*/ 0.037, /*kResASig0KSt0=*/ 0.037,
/*kResXi0KSt0=*/  0.017, /*kResAXi0AKSt0=*/  0.015, /*kResXi0AKSt0=*/  0.017, /*kResAXi0KSt0=*/  0.015,
/*kResXiCKSt0=*/  0.022, /*kResAXiCAKSt0=*/  0.021, /*kResXiCAKSt0=*/  0.022, /*kResAXiCKSt0=*/  0.021,

/*kResSig0K0=*/   0.113, /*kResASig0K0=*/   0.114,
/*kResXi0K0=*/    0.051, /*kResAXi0K0=*/    0.045,
/*kResXiCK0=*/    0.069, /*kResAXiCK0=*/    0.063,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.052, /*kResASigStMK0=*/ 0.051,
/*kResSigStMK0=*/ 0.047, /*kResASigStPK0=*/ 0.050,
/*kResSigSt0K0=*/ 0.047, /*kResASigSt0K0=*/ 0.045,

/*kResLamKSt0ToLamK0=*/  0.020, /*kResALamAKSt0ToALamK0=*/  0.020,
/*kResSig0KSt0ToLamK0=*/ 0.017, /*kResASig0AKSt0ToALamK0=*/ 0.018,
/*kResXi0KSt0ToLamK0=*/  0.008, /*kResAXi0AKSt0ToALamK0=*/  0.007,
/*kResXiCKSt0ToLamK0=*/  0.011, /*kResAXiCAKSt0ToALamK0=*/  0.010
};

//------------ WHEN INCLUDING ALL 10 RESIDUALS and MaxDecayLengthPrimary=10 ------------------------------------//
const double cAnalysisLambdaFactors_10Res_MaxPrimDecay10[85] = {
/*kLamK0=*/   0.342,   /*kALamK0=*/   0.344,
/*kLamKchP=*/ 0.317,   /*kALamKchM=*/ 0.319,  /*kLamKchM=*/ 0.317, /*kALamKchP=*/ 0.318,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.105, /*kResASig0KchM=*/   0.105, /*kResSig0KchM=*/   0.104, /*kResASig0KchP=*/  0.106,
/*kResXi0KchP=*/    0.047, /*kResAXi0KchM=*/    0.042, /*kResXi0KchM=*/    0.047, /*kResAXi0KchP=*/   0.042,
/*kResXiCKchP=*/    0.064, /*kResAXiCKchM=*/    0.058, /*kResXiCKchM=*/    0.063, /*kResAXiCKchP=*/   0.059,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.049, /*kResASigStMKchM=*/ 0.047, /*kResSigStPKchM=*/ 0.048, /*kResASigStMKchP=*/ 0.048,
/*kResSigStMKchP=*/ 0.044, /*kResASigStPKchM=*/ 0.046, /*kResSigStMKchM=*/ 0.044, /*kResASigStPKchP=*/ 0.047,
/*kResSigSt0KchP=*/ 0.044, /*kResASigSt0KchM=*/ 0.041, /*kResSigSt0KchM=*/ 0.044, /*kResASigSt0KchP=*/ 0.042,

/*kResLamKSt0=*/  0.041, /*kResALamAKSt0=*/  0.042, /*kResLamAKSt0=*/  0.041, /*kResALamKSt0=*/  0.042,
/*kResSig0KSt0=*/ 0.037, /*kResASig0AKSt0=*/ 0.037, /*kResSig0AKSt0=*/ 0.037, /*kResASig0KSt0=*/ 0.037,
/*kResXi0KSt0=*/  0.017, /*kResAXi0AKSt0=*/  0.015, /*kResXi0AKSt0=*/  0.017, /*kResAXi0KSt0=*/  0.015,
/*kResXiCKSt0=*/  0.022, /*kResAXiCAKSt0=*/  0.021, /*kResXiCAKSt0=*/  0.022, /*kResAXiCKSt0=*/  0.021,

/*kResSig0K0=*/   0.113, /*kResASig0K0=*/   0.114,
/*kResXi0K0=*/    0.051, /*kResAXi0K0=*/    0.045,
/*kResXiCK0=*/    0.069, /*kResAXiCK0=*/    0.063,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.052, /*kResASigStMK0=*/ 0.051,
/*kResSigStMK0=*/ 0.047, /*kResASigStPK0=*/ 0.050,
/*kResSigSt0K0=*/ 0.047, /*kResASigSt0K0=*/ 0.045,

/*kResLamKSt0ToLamK0=*/  0.020, /*kResALamAKSt0ToALamK0=*/  0.020,
/*kResSig0KSt0ToLamK0=*/ 0.017, /*kResASig0AKSt0ToALamK0=*/ 0.018,
/*kResXi0KSt0ToLamK0=*/  0.008, /*kResAXi0AKSt0ToALamK0=*/  0.007,
/*kResXiCKSt0ToLamK0=*/  0.011, /*kResAXiCAKSt0ToALamK0=*/  0.010
};


//------------ WHEN INCLUDING ALL 10 RESIDUALS and MaxDecayLengthPrimary=100 ------------------------------------//
const double cAnalysisLambdaFactors_10Res_MaxPrimDecay100[85] = {
/*kLamK0=*/   0.368,   /*kALamK0=*/   0.370,
/*kLamKchP=*/ 0.358,   /*kALamKchM=*/ 0.360,  /*kLamKchM=*/ 0.358, /*kALamKchP=*/ 0.360,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.105, /*kResASig0KchM=*/   0.105, /*kResSig0KchM=*/   0.104, /*kResASig0KchP=*/  0.106,
/*kResXi0KchP=*/    0.047, /*kResAXi0KchM=*/    0.042, /*kResXi0KchM=*/    0.047, /*kResAXi0KchP=*/   0.042,
/*kResXiCKchP=*/    0.064, /*kResAXiCKchM=*/    0.058, /*kResXiCKchM=*/    0.063, /*kResAXiCKchP=*/   0.059,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.049, /*kResASigStMKchM=*/ 0.047, /*kResSigStPKchM=*/ 0.048, /*kResASigStMKchP=*/ 0.048,
/*kResSigStMKchP=*/ 0.044, /*kResASigStPKchM=*/ 0.046, /*kResSigStMKchM=*/ 0.044, /*kResASigStPKchP=*/ 0.047,
/*kResSigSt0KchP=*/ 0.044, /*kResASigSt0KchM=*/ 0.041, /*kResSigSt0KchM=*/ 0.044, /*kResASigSt0KchP=*/ 0.042,

/*kResLamKSt0=*/  0.041, /*kResALamAKSt0=*/  0.042, /*kResLamAKSt0=*/  0.041, /*kResALamKSt0=*/  0.042,
/*kResSig0KSt0=*/ 0.037, /*kResASig0AKSt0=*/ 0.037, /*kResSig0AKSt0=*/ 0.037, /*kResASig0KSt0=*/ 0.037,
/*kResXi0KSt0=*/  0.017, /*kResAXi0AKSt0=*/  0.015, /*kResXi0AKSt0=*/  0.017, /*kResAXi0KSt0=*/  0.015,
/*kResXiCKSt0=*/  0.022, /*kResAXiCAKSt0=*/  0.021, /*kResXiCAKSt0=*/  0.022, /*kResAXiCKSt0=*/  0.021,

/*kResSig0K0=*/   0.113, /*kResASig0K0=*/   0.114,
/*kResXi0K0=*/    0.051, /*kResAXi0K0=*/    0.045,
/*kResXiCK0=*/    0.069, /*kResAXiCK0=*/    0.063,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.052, /*kResASigStMK0=*/ 0.051,
/*kResSigStMK0=*/ 0.047, /*kResASigStPK0=*/ 0.050,
/*kResSigSt0K0=*/ 0.047, /*kResASigSt0K0=*/ 0.045,

/*kResLamKSt0ToLamK0=*/  0.020, /*kResALamAKSt0ToALamK0=*/  0.020,
/*kResSig0KSt0ToLamK0=*/ 0.017, /*kResASig0AKSt0ToALamK0=*/ 0.018,
/*kResXi0KSt0ToLamK0=*/  0.008, /*kResAXi0AKSt0ToALamK0=*/  0.007,
/*kResXiCKSt0ToLamK0=*/  0.011, /*kResAXiCAKSt0ToALamK0=*/  0.010
};


const double *cAnalysisLambdaFactors_10Res[6] = {cAnalysisLambdaFactors_10Res_MaxPrimDecay0, 
                                                 cAnalysisLambdaFactors_10Res_MaxPrimDecay4, 
                                                 cAnalysisLambdaFactors_10Res_MaxPrimDecay5, 
                                                 cAnalysisLambdaFactors_10Res_MaxPrimDecay6, 
                                                 cAnalysisLambdaFactors_10Res_MaxPrimDecay10, 
                                                 cAnalysisLambdaFactors_10Res_MaxPrimDecay100};

//----------------------------------------------------------------------------------------------------------------
//****************************************************************************************************************
//----------------------------------------------------------------------------------------------------------------



//------------ WHEN INCLUDING 3 RESIDUALS and MaxDecayLengthPrimary=0 ------------------------------------//
const double cAnalysisLambdaFactors_3Res_MaxPrimDecay0[85] = {
/*kLamK0=*/   0.125,   /*kALamK0=*/   0.125,
/*kLamKchP=*/ 0.115,   /*kALamKchM=*/ 0.114,  /*kLamKchM=*/ 0.114, /*kALamKchP=*/ 0.116,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.103, /*kResASig0KchM=*/   0.102, /*kResSig0KchM=*/   0.102, /*kResASig0KchP=*/  0.102,
/*kResXi0KchP=*/    0.046, /*kResAXi0KchM=*/    0.040, /*kResXi0KchM=*/    0.046, /*kResAXi0KchP=*/   0.041,
/*kResXiCKchP=*/    0.063, /*kResAXiCKchM=*/    0.056, /*kResXiCKchM=*/    0.062, /*kResAXiCKchP=*/   0.057,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.000, /*kResASigStMKchM=*/ 0.000, /*kResSigStPKchM=*/ 0.000, /*kResASigStMKchP=*/ 0.000,
/*kResSigStMKchP=*/ 0.000, /*kResASigStPKchM=*/ 0.000, /*kResSigStMKchM=*/ 0.000, /*kResASigStPKchP=*/ 0.000,
/*kResSigSt0KchP=*/ 0.000, /*kResASigSt0KchM=*/ 0.000, /*kResSigSt0KchM=*/ 0.000, /*kResASigSt0KchP=*/ 0.000,

/*kResLamKSt0=*/  0.000, /*kResALamAKSt0=*/  0.000, /*kResLamAKSt0=*/  0.000, /*kResALamKSt0=*/  0.000,
/*kResSig0KSt0=*/ 0.000, /*kResASig0AKSt0=*/ 0.000, /*kResSig0AKSt0=*/ 0.000, /*kResASig0KSt0=*/ 0.000,
/*kResXi0KSt0=*/  0.000, /*kResAXi0AKSt0=*/  0.000, /*kResXi0AKSt0=*/  0.000, /*kResAXi0KSt0=*/  0.000,
/*kResXiCKSt0=*/  0.000, /*kResAXiCAKSt0=*/  0.000, /*kResXiCAKSt0=*/  0.000, /*kResAXiCKSt0=*/  0.000,

/*kResSig0K0=*/   0.112, /*kResASig0K0=*/   0.111,
/*kResXi0K0=*/    0.050, /*kResAXi0K0=*/    0.044,
/*kResXiCK0=*/    0.068, /*kResAXiCK0=*/    0.062,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.000, /*kResASigStMK0=*/ 0.000,
/*kResSigStMK0=*/ 0.000, /*kResASigStPK0=*/ 0.000,
/*kResSigSt0K0=*/ 0.000, /*kResASigSt0K0=*/ 0.000,

/*kResLamKSt0ToLamK0=*/  0.000, /*kResALamAKSt0ToALamK0=*/  0.000,
/*kResSig0KSt0ToLamK0=*/ 0.000, /*kResASig0AKSt0ToALamK0=*/ 0.000,
/*kResXi0KSt0ToLamK0=*/  0.000, /*kResAXi0AKSt0ToALamK0=*/  0.000,
/*kResXiCKSt0ToLamK0=*/  0.000, /*kResAXiCAKSt0ToALamK0=*/  0.000
};


//------------ WHEN INCLUDING 3 RESIDUALS and MaxDecayLengthPrimary=4 ------------------------------------//
const double cAnalysisLambdaFactors_3Res_MaxPrimDecay4[85] = {
/*kLamK0=*/   0.172,   /*kALamK0=*/   0.170,
/*kLamKchP=*/ 0.160,   /*kALamKchM=*/ 0.157,  /*kLamKchM=*/ 0.160, /*kALamKchP=*/ 0.158,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.103, /*kResASig0KchM=*/   0.102, /*kResSig0KchM=*/   0.102, /*kResASig0KchP=*/  0.102,
/*kResXi0KchP=*/    0.046, /*kResAXi0KchM=*/    0.040, /*kResXi0KchM=*/    0.046, /*kResAXi0KchP=*/   0.041,
/*kResXiCKchP=*/    0.063, /*kResAXiCKchM=*/    0.056, /*kResXiCKchM=*/    0.062, /*kResAXiCKchP=*/   0.057,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.000, /*kResASigStMKchM=*/ 0.000, /*kResSigStPKchM=*/ 0.000, /*kResASigStMKchP=*/ 0.000,
/*kResSigStMKchP=*/ 0.000, /*kResASigStPKchM=*/ 0.000, /*kResSigStMKchM=*/ 0.000, /*kResASigStPKchP=*/ 0.000,
/*kResSigSt0KchP=*/ 0.000, /*kResASigSt0KchM=*/ 0.000, /*kResSigSt0KchM=*/ 0.000, /*kResASigSt0KchP=*/ 0.000,

/*kResLamKSt0=*/  0.000, /*kResALamAKSt0=*/  0.000, /*kResLamAKSt0=*/  0.000, /*kResALamKSt0=*/  0.000,
/*kResSig0KSt0=*/ 0.000, /*kResASig0AKSt0=*/ 0.000, /*kResSig0AKSt0=*/ 0.000, /*kResASig0KSt0=*/ 0.000,
/*kResXi0KSt0=*/  0.000, /*kResAXi0AKSt0=*/  0.000, /*kResXi0AKSt0=*/  0.000, /*kResAXi0KSt0=*/  0.000,
/*kResXiCKSt0=*/  0.000, /*kResAXiCAKSt0=*/  0.000, /*kResXiCAKSt0=*/  0.000, /*kResAXiCKSt0=*/  0.000,

/*kResSig0K0=*/   0.112, /*kResASig0K0=*/   0.111,
/*kResXi0K0=*/    0.050, /*kResAXi0K0=*/    0.044,
/*kResXiCK0=*/    0.068, /*kResAXiCK0=*/    0.062,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.000, /*kResASigStMK0=*/ 0.000,
/*kResSigStMK0=*/ 0.000, /*kResASigStPK0=*/ 0.000,
/*kResSigSt0K0=*/ 0.000, /*kResASigSt0K0=*/ 0.000,

/*kResLamKSt0ToLamK0=*/  0.000, /*kResALamAKSt0ToALamK0=*/  0.000,
/*kResSig0KSt0ToLamK0=*/ 0.000, /*kResASig0AKSt0ToALamK0=*/ 0.000,
/*kResXi0KSt0ToLamK0=*/  0.000, /*kResAXi0AKSt0ToALamK0=*/  0.000,
/*kResXiCKSt0ToLamK0=*/  0.000, /*kResAXiCAKSt0ToALamK0=*/  0.000
};

//------------ WHEN INCLUDING 3 RESIDUALS and MaxDecayLengthPrimary=5 ------------------------------------//
const double cAnalysisLambdaFactors_3Res_MaxPrimDecay5[85] = {
/*kLamK0=*/   0.245,   /*kALamK0=*/   0.244,
/*kLamKchP=*/ 0.236,   /*kALamKchM=*/ 0.234,  /*kLamKchM=*/ 0.236, /*kALamKchP=*/ 0.234,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.103, /*kResASig0KchM=*/   0.102, /*kResSig0KchM=*/   0.102, /*kResASig0KchP=*/  0.102,
/*kResXi0KchP=*/    0.046, /*kResAXi0KchM=*/    0.040, /*kResXi0KchM=*/    0.046, /*kResAXi0KchP=*/   0.041,
/*kResXiCKchP=*/    0.063, /*kResAXiCKchM=*/    0.056, /*kResXiCKchM=*/    0.062, /*kResAXiCKchP=*/   0.057,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.000, /*kResASigStMKchM=*/ 0.000, /*kResSigStPKchM=*/ 0.000, /*kResASigStMKchP=*/ 0.000,
/*kResSigStMKchP=*/ 0.000, /*kResASigStPKchM=*/ 0.000, /*kResSigStMKchM=*/ 0.000, /*kResASigStPKchP=*/ 0.000,
/*kResSigSt0KchP=*/ 0.000, /*kResASigSt0KchM=*/ 0.000, /*kResSigSt0KchM=*/ 0.000, /*kResASigSt0KchP=*/ 0.000,

/*kResLamKSt0=*/  0.000, /*kResALamAKSt0=*/  0.000, /*kResLamAKSt0=*/  0.000, /*kResALamKSt0=*/  0.000,
/*kResSig0KSt0=*/ 0.000, /*kResASig0AKSt0=*/ 0.000, /*kResSig0AKSt0=*/ 0.000, /*kResASig0KSt0=*/ 0.000,
/*kResXi0KSt0=*/  0.000, /*kResAXi0AKSt0=*/  0.000, /*kResXi0AKSt0=*/  0.000, /*kResAXi0KSt0=*/  0.000,
/*kResXiCKSt0=*/  0.000, /*kResAXiCAKSt0=*/  0.000, /*kResXiCAKSt0=*/  0.000, /*kResAXiCKSt0=*/  0.000,

/*kResSig0K0=*/   0.112, /*kResASig0K0=*/   0.111,
/*kResXi0K0=*/    0.050, /*kResAXi0K0=*/    0.044,
/*kResXiCK0=*/    0.068, /*kResAXiCK0=*/    0.062,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.000, /*kResASigStMK0=*/ 0.000,
/*kResSigStMK0=*/ 0.000, /*kResASigStPK0=*/ 0.000,
/*kResSigSt0K0=*/ 0.000, /*kResASigSt0K0=*/ 0.000,

/*kResLamKSt0ToLamK0=*/  0.000, /*kResALamAKSt0ToALamK0=*/  0.000,
/*kResSig0KSt0ToLamK0=*/ 0.000, /*kResASig0AKSt0ToALamK0=*/ 0.000,
/*kResXi0KSt0ToLamK0=*/  0.000, /*kResAXi0AKSt0ToALamK0=*/  0.000,
/*kResXiCKSt0ToLamK0=*/  0.000, /*kResAXiCAKSt0ToALamK0=*/  0.000
};


//------------ WHEN INCLUDING 3 RESIDUALS and MaxDecayLengthPrimary=6 ------------------------------------//
const double cAnalysisLambdaFactors_3Res_MaxPrimDecay6[85] = {
/*kLamK0=*/   0.480,   /*kALamK0=*/   0.483,
/*kLamKchP=*/ 0.463,   /*kALamKchM=*/ 0.463,  /*kLamKchM=*/ 0.462, /*kALamKchP=*/ 0.464,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.103, /*kResASig0KchM=*/   0.102, /*kResSig0KchM=*/   0.102, /*kResASig0KchP=*/  0.102,
/*kResXi0KchP=*/    0.046, /*kResAXi0KchM=*/    0.040, /*kResXi0KchM=*/    0.046, /*kResAXi0KchP=*/   0.041,
/*kResXiCKchP=*/    0.063, /*kResAXiCKchM=*/    0.056, /*kResXiCKchM=*/    0.062, /*kResAXiCKchP=*/   0.057,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.000, /*kResASigStMKchM=*/ 0.000, /*kResSigStPKchM=*/ 0.000, /*kResASigStMKchP=*/ 0.000,
/*kResSigStMKchP=*/ 0.000, /*kResASigStPKchM=*/ 0.000, /*kResSigStMKchM=*/ 0.000, /*kResASigStPKchP=*/ 0.000,
/*kResSigSt0KchP=*/ 0.000, /*kResASigSt0KchM=*/ 0.000, /*kResSigSt0KchM=*/ 0.000, /*kResASigSt0KchP=*/ 0.000,

/*kResLamKSt0=*/  0.000, /*kResALamAKSt0=*/  0.000, /*kResLamAKSt0=*/  0.000, /*kResALamKSt0=*/  0.000,
/*kResSig0KSt0=*/ 0.000, /*kResASig0AKSt0=*/ 0.000, /*kResSig0AKSt0=*/ 0.000, /*kResASig0KSt0=*/ 0.000,
/*kResXi0KSt0=*/  0.000, /*kResAXi0AKSt0=*/  0.000, /*kResXi0AKSt0=*/  0.000, /*kResAXi0KSt0=*/  0.000,
/*kResXiCKSt0=*/  0.000, /*kResAXiCAKSt0=*/  0.000, /*kResXiCAKSt0=*/  0.000, /*kResAXiCKSt0=*/  0.000,

/*kResSig0K0=*/   0.112, /*kResASig0K0=*/   0.111,
/*kResXi0K0=*/    0.050, /*kResAXi0K0=*/    0.044,
/*kResXiCK0=*/    0.068, /*kResAXiCK0=*/    0.062,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.000, /*kResASigStMK0=*/ 0.000,
/*kResSigStMK0=*/ 0.000, /*kResASigStPK0=*/ 0.000,
/*kResSigSt0K0=*/ 0.000, /*kResASigSt0K0=*/ 0.000,

/*kResLamKSt0ToLamK0=*/  0.000, /*kResALamAKSt0ToALamK0=*/  0.000,
/*kResSig0KSt0ToLamK0=*/ 0.000, /*kResASig0AKSt0ToALamK0=*/ 0.000,
/*kResXi0KSt0ToLamK0=*/  0.000, /*kResAXi0AKSt0ToALamK0=*/  0.000,
/*kResXiCKSt0ToLamK0=*/  0.000, /*kResAXiCAKSt0ToALamK0=*/  0.000
};

//------------ WHEN INCLUDING 3 RESIDUALS and MaxDecayLengthPrimary=10 ------------------------------------//
const double cAnalysisLambdaFactors_3Res_MaxPrimDecay10[85] = {
/*kLamK0=*/   0.505,   /*kALamK0=*/   0.507,
/*kLamKchP=*/ 0.488,   /*kALamKchM=*/ 0.488,  /*kLamKchM=*/ 0.487, /*kALamKchP=*/ 0.489,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.103, /*kResASig0KchM=*/   0.102, /*kResSig0KchM=*/   0.102, /*kResASig0KchP=*/  0.102,
/*kResXi0KchP=*/    0.046, /*kResAXi0KchM=*/    0.040, /*kResXi0KchM=*/    0.046, /*kResAXi0KchP=*/   0.041,
/*kResXiCKchP=*/    0.063, /*kResAXiCKchM=*/    0.056, /*kResXiCKchM=*/    0.062, /*kResAXiCKchP=*/   0.057,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.000, /*kResASigStMKchM=*/ 0.000, /*kResSigStPKchM=*/ 0.000, /*kResASigStMKchP=*/ 0.000,
/*kResSigStMKchP=*/ 0.000, /*kResASigStPKchM=*/ 0.000, /*kResSigStMKchM=*/ 0.000, /*kResASigStPKchP=*/ 0.000,
/*kResSigSt0KchP=*/ 0.000, /*kResASigSt0KchM=*/ 0.000, /*kResSigSt0KchM=*/ 0.000, /*kResASigSt0KchP=*/ 0.000,

/*kResLamKSt0=*/  0.000, /*kResALamAKSt0=*/  0.000, /*kResLamAKSt0=*/  0.000, /*kResALamKSt0=*/  0.000,
/*kResSig0KSt0=*/ 0.000, /*kResASig0AKSt0=*/ 0.000, /*kResSig0AKSt0=*/ 0.000, /*kResASig0KSt0=*/ 0.000,
/*kResXi0KSt0=*/  0.000, /*kResAXi0AKSt0=*/  0.000, /*kResXi0AKSt0=*/  0.000, /*kResAXi0KSt0=*/  0.000,
/*kResXiCKSt0=*/  0.000, /*kResAXiCAKSt0=*/  0.000, /*kResXiCAKSt0=*/  0.000, /*kResAXiCKSt0=*/  0.000,

/*kResSig0K0=*/   0.112, /*kResASig0K0=*/   0.111,
/*kResXi0K0=*/    0.050, /*kResAXi0K0=*/    0.044,
/*kResXiCK0=*/    0.068, /*kResAXiCK0=*/    0.062,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.000, /*kResASigStMK0=*/ 0.000,
/*kResSigStMK0=*/ 0.000, /*kResASigStPK0=*/ 0.000,
/*kResSigSt0K0=*/ 0.000, /*kResASigSt0K0=*/ 0.000,

/*kResLamKSt0ToLamK0=*/  0.000, /*kResALamAKSt0ToALamK0=*/  0.000,
/*kResSig0KSt0ToLamK0=*/ 0.000, /*kResASig0AKSt0ToALamK0=*/ 0.000,
/*kResXi0KSt0ToLamK0=*/  0.000, /*kResAXi0AKSt0ToALamK0=*/  0.000,
/*kResXiCKSt0ToLamK0=*/  0.000, /*kResAXiCAKSt0ToALamK0=*/  0.000
};


//------------ WHEN INCLUDING 3 RESIDUALS and MaxDecayLengthPrimary=100 ------------------------------------//
const double cAnalysisLambdaFactors_3Res_MaxPrimDecay100[85] = {
/*kLamK0=*/   0.531,   /*kALamK0=*/   0.534,
/*kLamKchP=*/ 0.528,   /*kALamKchM=*/ 0.529,  /*kLamKchM=*/ 0.528, /*kALamKchP=*/ 0.530,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.103, /*kResASig0KchM=*/   0.102, /*kResSig0KchM=*/   0.102, /*kResASig0KchP=*/  0.102,
/*kResXi0KchP=*/    0.046, /*kResAXi0KchM=*/    0.040, /*kResXi0KchM=*/    0.046, /*kResAXi0KchP=*/   0.041,
/*kResXiCKchP=*/    0.063, /*kResAXiCKchM=*/    0.056, /*kResXiCKchM=*/    0.062, /*kResAXiCKchP=*/   0.057,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.000, /*kResASigStMKchM=*/ 0.000, /*kResSigStPKchM=*/ 0.000, /*kResASigStMKchP=*/ 0.000,
/*kResSigStMKchP=*/ 0.000, /*kResASigStPKchM=*/ 0.000, /*kResSigStMKchM=*/ 0.000, /*kResASigStPKchP=*/ 0.000,
/*kResSigSt0KchP=*/ 0.000, /*kResASigSt0KchM=*/ 0.000, /*kResSigSt0KchM=*/ 0.000, /*kResASigSt0KchP=*/ 0.000,

/*kResLamKSt0=*/  0.000, /*kResALamAKSt0=*/  0.000, /*kResLamAKSt0=*/  0.000, /*kResALamKSt0=*/  0.000,
/*kResSig0KSt0=*/ 0.000, /*kResASig0AKSt0=*/ 0.000, /*kResSig0AKSt0=*/ 0.000, /*kResASig0KSt0=*/ 0.000,
/*kResXi0KSt0=*/  0.000, /*kResAXi0AKSt0=*/  0.000, /*kResXi0AKSt0=*/  0.000, /*kResAXi0KSt0=*/  0.000,
/*kResXiCKSt0=*/  0.000, /*kResAXiCAKSt0=*/  0.000, /*kResXiCAKSt0=*/  0.000, /*kResAXiCKSt0=*/  0.000,

/*kResSig0K0=*/   0.112, /*kResASig0K0=*/   0.111,
/*kResXi0K0=*/    0.050, /*kResAXi0K0=*/    0.044,
/*kResXiCK0=*/    0.068, /*kResAXiCK0=*/    0.062,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.000, /*kResASigStMK0=*/ 0.000,
/*kResSigStMK0=*/ 0.000, /*kResASigStPK0=*/ 0.000,
/*kResSigSt0K0=*/ 0.000, /*kResASigSt0K0=*/ 0.000,

/*kResLamKSt0ToLamK0=*/  0.000, /*kResALamAKSt0ToALamK0=*/  0.000,
/*kResSig0KSt0ToLamK0=*/ 0.000, /*kResASig0AKSt0ToALamK0=*/ 0.000,
/*kResXi0KSt0ToLamK0=*/  0.000, /*kResAXi0AKSt0ToALamK0=*/  0.000,
/*kResXiCKSt0ToLamK0=*/  0.000, /*kResAXiCAKSt0ToALamK0=*/  0.000
};


const double *cAnalysisLambdaFactors_3Res[6] = {cAnalysisLambdaFactors_3Res_MaxPrimDecay0, 
                                                cAnalysisLambdaFactors_3Res_MaxPrimDecay4, 
                                                cAnalysisLambdaFactors_3Res_MaxPrimDecay5, 
                                                cAnalysisLambdaFactors_3Res_MaxPrimDecay6, 
                                                cAnalysisLambdaFactors_3Res_MaxPrimDecay10, 
                                                cAnalysisLambdaFactors_3Res_MaxPrimDecay100};


//----------------------------------------------------------------------------------------------------------------
//****************************************************************************************************************
//----------------------------------------------------------------------------------------------------------------

const double** cAnalysisLambdaFactorsArr[3] = {cAnalysisLambdaFactors_NoRes, cAnalysisLambdaFactors_10Res, cAnalysisLambdaFactors_3Res};

