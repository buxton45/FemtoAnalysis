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
/*kLamK0=*/   0.141,   /*kALamK0=*/   0.144,
/*kLamKchP=*/ 0.131,   /*kALamKchM=*/ 0.132,  /*kLamKchM=*/ 0.130, /*kALamKchP=*/ 0.133,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.117, /*kResASig0KchM=*/   0.115, /*kResSig0KchM=*/   0.116, /*kResASig0KchP=*/  0.116,
/*kResXi0KchP=*/    0.041, /*kResAXi0KchM=*/    0.037, /*kResXi0KchM=*/    0.040, /*kResAXi0KchP=*/   0.037,
/*kResXiCKchP=*/    0.052, /*kResAXiCKchM=*/    0.048, /*kResXiCKchM=*/    0.052, /*kResAXiCKchP=*/   0.048,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.054, /*kResASigStMKchM=*/ 0.051, /*kResSigStPKchM=*/ 0.054, /*kResASigStMKchP=*/ 0.052,
/*kResSigStMKchP=*/ 0.049, /*kResASigStPKchM=*/ 0.050, /*kResSigStMKchM=*/ 0.048, /*kResASigStPKchP=*/ 0.050,
/*kResSigSt0KchP=*/ 0.049, /*kResASigSt0KchM=*/ 0.045, /*kResSigSt0KchM=*/ 0.048, /*kResASigSt0KchP=*/ 0.045,

/*kResLamKSt0=*/  0.046, /*kResALamAKSt0=*/  0.047, /*kResLamAKSt0=*/  0.046, /*kResALamKSt0=*/  0.047,
/*kResSig0KSt0=*/ 0.041, /*kResASig0AKSt0=*/ 0.041, /*kResSig0AKSt0=*/ 0.041, /*kResASig0KSt0=*/ 0.041,
/*kResXi0KSt0=*/  0.014, /*kResAXi0AKSt0=*/  0.013, /*kResXi0AKSt0=*/  0.014, /*kResAXi0KSt0=*/  0.013,
/*kResXiCKSt0=*/  0.018, /*kResAXiCAKSt0=*/  0.017, /*kResXiCAKSt0=*/  0.019, /*kResAXiCKSt0=*/  0.017,

/*kResSig0K0=*/   0.126, /*kResASig0K0=*/   0.125,
/*kResXi0K0=*/    0.044, /*kResAXi0K0=*/    0.040,
/*kResXiCK0=*/    0.057, /*kResAXiCK0=*/    0.052,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.058, /*kResASigStMK0=*/ 0.056,
/*kResSigStMK0=*/ 0.053, /*kResASigStPK0=*/ 0.054,
/*kResSigSt0K0=*/ 0.053, /*kResASigSt0K0=*/ 0.049,

/*kResLamKSt0ToLamK0=*/  0.022, /*kResALamAKSt0ToALamK0=*/  0.022,
/*kResSig0KSt0ToLamK0=*/ 0.020, /*kResASig0AKSt0ToALamK0=*/ 0.019,
/*kResXi0KSt0ToLamK0=*/  0.007, /*kResAXi0AKSt0ToALamK0=*/  0.006,
/*kResXiCKSt0ToLamK0=*/  0.009, /*kResAXiCAKSt0ToALamK0=*/  0.008
};


//------------ WHEN INCLUDING ALL 10 RESIDUALS and MaxDecayLengthPrimary=4 ------------------------------------//
const double cAnalysisLambdaFactors_10Res_MaxPrimDecay4[85] = {
/*kLamK0=*/   0.192,   /*kALamK0=*/   0.193,
/*kLamKchP=*/ 0.180,   /*kALamKchM=*/ 0.180,  /*kLamKchM=*/ 0.179, /*kALamKchP=*/ 0.181,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.116, /*kResASig0KchM=*/   0.114, /*kResSig0KchM=*/   0.115, /*kResASig0KchP=*/  0.116,
/*kResXi0KchP=*/    0.040, /*kResAXi0KchM=*/    0.037, /*kResXi0KchM=*/    0.040, /*kResAXi0KchP=*/   0.037,
/*kResXiCKchP=*/    0.052, /*kResAXiCKchM=*/    0.047, /*kResXiCKchM=*/    0.052, /*kResAXiCKchP=*/   0.048,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.054, /*kResASigStMKchM=*/ 0.051, /*kResSigStPKchM=*/ 0.053, /*kResASigStMKchP=*/ 0.051,
/*kResSigStMKchP=*/ 0.048, /*kResASigStPKchM=*/ 0.050, /*kResSigStMKchM=*/ 0.048, /*kResASigStPKchP=*/ 0.050,
/*kResSigSt0KchP=*/ 0.048, /*kResASigSt0KchM=*/ 0.045, /*kResSigSt0KchM=*/ 0.048, /*kResASigSt0KchP=*/ 0.045,

/*kResLamKSt0=*/  0.046, /*kResALamAKSt0=*/  0.047, /*kResLamAKSt0=*/  0.046, /*kResALamKSt0=*/  0.047,
/*kResSig0KSt0=*/ 0.041, /*kResASig0AKSt0=*/ 0.041, /*kResSig0AKSt0=*/ 0.041, /*kResASig0KSt0=*/ 0.041,
/*kResXi0KSt0=*/  0.014, /*kResAXi0AKSt0=*/  0.013, /*kResXi0AKSt0=*/  0.014, /*kResAXi0KSt0=*/  0.013,
/*kResXiCKSt0=*/  0.018, /*kResAXiCAKSt0=*/  0.017, /*kResXiCAKSt0=*/  0.018, /*kResAXiCKSt0=*/  0.017,

/*kResSig0K0=*/   0.125, /*kResASig0K0=*/   0.124,
/*kResXi0K0=*/    0.043, /*kResAXi0K0=*/    0.040,
/*kResXiCK0=*/    0.056, /*kResAXiCK0=*/    0.052,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.058, /*kResASigStMK0=*/ 0.055,
/*kResSigStMK0=*/ 0.052, /*kResASigStPK0=*/ 0.054,
/*kResSigSt0K0=*/ 0.052, /*kResASigSt0K0=*/ 0.048,

/*kResLamKSt0ToLamK0=*/  0.022, /*kResALamAKSt0ToALamK0=*/  0.022,
/*kResSig0KSt0ToLamK0=*/ 0.019, /*kResASig0AKSt0ToALamK0=*/ 0.019,
/*kResXi0KSt0ToLamK0=*/  0.007, /*kResAXi0AKSt0ToALamK0=*/  0.006,
/*kResXiCKSt0ToLamK0=*/  0.009, /*kResAXiCAKSt0ToALamK0=*/  0.008
};

//------------ WHEN INCLUDING ALL 10 RESIDUALS and MaxDecayLengthPrimary=5 ------------------------------------//
const double cAnalysisLambdaFactors_10Res_MaxPrimDecay5[85] = {
/*kLamK0=*/   0.249,   /*kALamK0=*/   0.251,
/*kLamKchP=*/ 0.218,   /*kALamKchM=*/ 0.219,  /*kLamKchM=*/ 0.218, /*kALamKchP=*/ 0.219,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.115, /*kResASig0KchM=*/   0.114, /*kResSig0KchM=*/   0.114, /*kResASig0KchP=*/  0.115,
/*kResXi0KchP=*/    0.040, /*kResAXi0KchM=*/    0.037, /*kResXi0KchM=*/    0.040, /*kResAXi0KchP=*/   0.037,
/*kResXiCKchP=*/    0.052, /*kResAXiCKchM=*/    0.047, /*kResXiCKchM=*/    0.051, /*kResAXiCKchP=*/   0.048,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.053, /*kResASigStMKchM=*/ 0.051, /*kResSigStPKchM=*/ 0.053, /*kResASigStMKchP=*/ 0.051,
/*kResSigStMKchP=*/ 0.048, /*kResASigStPKchM=*/ 0.049, /*kResSigStMKchM=*/ 0.048, /*kResASigStPKchP=*/ 0.050,
/*kResSigSt0KchP=*/ 0.048, /*kResASigSt0KchM=*/ 0.044, /*kResSigSt0KchM=*/ 0.048, /*kResASigSt0KchP=*/ 0.045,

/*kResLamKSt0=*/  0.045, /*kResALamAKSt0=*/  0.047, /*kResLamAKSt0=*/  0.045, /*kResALamKSt0=*/  0.047,
/*kResSig0KSt0=*/ 0.041, /*kResASig0AKSt0=*/ 0.041, /*kResSig0AKSt0=*/ 0.041, /*kResASig0KSt0=*/ 0.041,
/*kResXi0KSt0=*/  0.014, /*kResAXi0AKSt0=*/  0.013, /*kResXi0AKSt0=*/  0.014, /*kResAXi0KSt0=*/  0.013,
/*kResXiCKSt0=*/  0.018, /*kResAXiCAKSt0=*/  0.017, /*kResXiCAKSt0=*/  0.018, /*kResAXiCKSt0=*/  0.017,

/*kResSig0K0=*/   0.124, /*kResASig0K0=*/   0.123,
/*kResXi0K0=*/    0.043, /*kResAXi0K0=*/    0.040,
/*kResXiCK0=*/    0.055, /*kResAXiCK0=*/    0.051,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.057, /*kResASigStMK0=*/ 0.055,
/*kResSigStMK0=*/ 0.052, /*kResASigStPK0=*/ 0.054,
/*kResSigSt0K0=*/ 0.052, /*kResASigSt0K0=*/ 0.048,

/*kResLamKSt0ToLamK0=*/  0.022, /*kResALamAKSt0ToALamK0=*/  0.022,
/*kResSig0KSt0ToLamK0=*/ 0.019, /*kResASig0AKSt0ToALamK0=*/ 0.019,
/*kResXi0KSt0ToLamK0=*/  0.007, /*kResAXi0AKSt0ToALamK0=*/  0.006,
/*kResXiCKSt0ToLamK0=*/  0.009, /*kResAXiCAKSt0ToALamK0=*/  0.008
};


//------------ WHEN INCLUDING ALL 10 RESIDUALS and MaxDecayLengthPrimary=6 ------------------------------------//
const double cAnalysisLambdaFactors_10Res_MaxPrimDecay6[85] = {
/*kLamK0=*/   0.341,   /*kALamK0=*/   0.345,
/*kLamKchP=*/ 0.314,   /*kALamKchM=*/ 0.318,  /*kLamKchM=*/ 0.315, /*kALamKchP=*/ 0.318,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.113, /*kResASig0KchM=*/   0.113, /*kResSig0KchM=*/   0.112, /*kResASig0KchP=*/  0.114,
/*kResXi0KchP=*/    0.039, /*kResAXi0KchM=*/    0.036, /*kResXi0KchM=*/    0.039, /*kResAXi0KchP=*/   0.037,
/*kResXiCKchP=*/    0.051, /*kResAXiCKchM=*/    0.047, /*kResXiCKchM=*/    0.050, /*kResAXiCKchP=*/   0.047,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.052, /*kResASigStMKchM=*/ 0.050, /*kResSigStPKchM=*/ 0.052, /*kResASigStMKchP=*/ 0.051,
/*kResSigStMKchP=*/ 0.047, /*kResASigStPKchM=*/ 0.049, /*kResSigStMKchM=*/ 0.047, /*kResASigStPKchP=*/ 0.049,
/*kResSigSt0KchP=*/ 0.047, /*kResASigSt0KchM=*/ 0.044, /*kResSigSt0KchM=*/ 0.047, /*kResASigSt0KchP=*/ 0.044,

/*kResLamKSt0=*/  0.045, /*kResALamAKSt0=*/  0.046, /*kResLamAKSt0=*/  0.045, /*kResALamKSt0=*/  0.046,
/*kResSig0KSt0=*/ 0.040, /*kResASig0AKSt0=*/ 0.040, /*kResSig0AKSt0=*/ 0.040, /*kResASig0KSt0=*/ 0.040,
/*kResXi0KSt0=*/  0.014, /*kResAXi0AKSt0=*/  0.013, /*kResXi0AKSt0=*/  0.014, /*kResAXi0KSt0=*/  0.013,
/*kResXiCKSt0=*/  0.018, /*kResAXiCAKSt0=*/  0.017, /*kResXiCAKSt0=*/  0.018, /*kResAXiCKSt0=*/  0.017,

/*kResSig0K0=*/   0.121, /*kResASig0K0=*/   0.122,
/*kResXi0K0=*/    0.042, /*kResAXi0K0=*/    0.039,
/*kResXiCK0=*/    0.055, /*kResAXiCK0=*/    0.051,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.056, /*kResASigStMK0=*/ 0.054,
/*kResSigStMK0=*/ 0.051, /*kResASigStPK0=*/ 0.053,
/*kResSigSt0K0=*/ 0.051, /*kResASigSt0K0=*/ 0.047,

/*kResLamKSt0ToLamK0=*/  0.021, /*kResALamAKSt0ToALamK0=*/  0.022,
/*kResSig0KSt0ToLamK0=*/ 0.019, /*kResASig0AKSt0ToALamK0=*/ 0.019,
/*kResXi0KSt0ToLamK0=*/  0.007, /*kResAXi0AKSt0ToALamK0=*/  0.006,
/*kResXiCKSt0ToLamK0=*/  0.008, /*kResAXiCAKSt0ToALamK0=*/  0.008
};

//------------ WHEN INCLUDING ALL 10 RESIDUALS and MaxDecayLengthPrimary=10 ------------------------------------//
const double cAnalysisLambdaFactors_10Res_MaxPrimDecay10[85] = {
/*kLamK0=*/   0.367,   /*kALamK0=*/   0.370,
/*kLamKchP=*/ 0.341,   /*kALamKchM=*/ 0.343,  /*kLamKchM=*/ 0.341, /*kALamKchP=*/ 0.343,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.113, /*kResASig0KchM=*/   0.112, /*kResSig0KchM=*/   0.111, /*kResASig0KchP=*/  0.113,
/*kResXi0KchP=*/    0.039, /*kResAXi0KchM=*/    0.036, /*kResXi0KchM=*/    0.039, /*kResAXi0KchP=*/   0.036,
/*kResXiCKchP=*/    0.050, /*kResAXiCKchM=*/    0.047, /*kResXiCKchM=*/    0.050, /*kResAXiCKchP=*/   0.047,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.052, /*kResASigStMKchM=*/ 0.050, /*kResSigStPKchM=*/ 0.051, /*kResASigStMKchP=*/ 0.050,
/*kResSigStMKchP=*/ 0.047, /*kResASigStPKchM=*/ 0.049, /*kResSigStMKchM=*/ 0.047, /*kResASigStPKchP=*/ 0.049,
/*kResSigSt0KchP=*/ 0.047, /*kResASigSt0KchM=*/ 0.044, /*kResSigSt0KchM=*/ 0.047, /*kResASigSt0KchP=*/ 0.044,

/*kResLamKSt0=*/  0.044, /*kResALamAKSt0=*/  0.046, /*kResLamAKSt0=*/  0.044, /*kResALamKSt0=*/  0.046,
/*kResSig0KSt0=*/ 0.040, /*kResASig0AKSt0=*/ 0.040, /*kResSig0AKSt0=*/ 0.040, /*kResASig0KSt0=*/ 0.040,
/*kResXi0KSt0=*/  0.014, /*kResAXi0AKSt0=*/  0.013, /*kResXi0AKSt0=*/  0.014, /*kResAXi0KSt0=*/  0.013,
/*kResXiCKSt0=*/  0.018, /*kResAXiCAKSt0=*/  0.017, /*kResXiCAKSt0=*/  0.018, /*kResAXiCKSt0=*/  0.017,

/*kResSig0K0=*/   0.121, /*kResASig0K0=*/   0.122,
/*kResXi0K0=*/    0.042, /*kResAXi0K0=*/    0.039,
/*kResXiCK0=*/    0.054, /*kResAXiCK0=*/    0.051,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.056, /*kResASigStMK0=*/ 0.054,
/*kResSigStMK0=*/ 0.050, /*kResASigStPK0=*/ 0.053,
/*kResSigSt0K0=*/ 0.051, /*kResASigSt0K0=*/ 0.047,

/*kResLamKSt0ToLamK0=*/  0.021, /*kResALamAKSt0ToALamK0=*/  0.022,
/*kResSig0KSt0ToLamK0=*/ 0.019, /*kResASig0AKSt0ToALamK0=*/ 0.019,
/*kResXi0KSt0ToLamK0=*/  0.006, /*kResAXi0AKSt0ToALamK0=*/  0.006,
/*kResXiCKSt0ToLamK0=*/  0.008, /*kResAXiCAKSt0ToALamK0=*/  0.008
};


//------------ WHEN INCLUDING ALL 10 RESIDUALS and MaxDecayLengthPrimary=100 ------------------------------------//
const double cAnalysisLambdaFactors_10Res_MaxPrimDecay100[85] = {
/*kLamK0=*/   0.393,   /*kALamK0=*/   0.396,
/*kLamKchP=*/ 0.381,   /*kALamKchM=*/ 0.385,  /*kLamKchM=*/ 0.382, /*kALamKchP=*/ 0.384,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.112, /*kResASig0KchM=*/   0.112, /*kResSig0KchM=*/   0.111, /*kResASig0KchP=*/  0.113,
/*kResXi0KchP=*/    0.039, /*kResAXi0KchM=*/    0.036, /*kResXi0KchM=*/    0.038, /*kResAXi0KchP=*/   0.036,
/*kResXiCKchP=*/    0.050, /*kResAXiCKchM=*/    0.046, /*kResXiCKchM=*/    0.050, /*kResAXiCKchP=*/   0.047,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.052, /*kResASigStMKchM=*/ 0.050, /*kResSigStPKchM=*/ 0.051, /*kResASigStMKchP=*/ 0.050,
/*kResSigStMKchP=*/ 0.047, /*kResASigStPKchM=*/ 0.048, /*kResSigStMKchM=*/ 0.046, /*kResASigStPKchP=*/ 0.049,
/*kResSigSt0KchP=*/ 0.047, /*kResASigSt0KchM=*/ 0.043, /*kResSigSt0KchM=*/ 0.046, /*kResASigSt0KchP=*/ 0.044,

/*kResLamKSt0=*/  0.044, /*kResALamAKSt0=*/  0.046, /*kResLamAKSt0=*/  0.044, /*kResALamKSt0=*/  0.046,
/*kResSig0KSt0=*/ 0.039, /*kResASig0AKSt0=*/ 0.040, /*kResSig0AKSt0=*/ 0.039, /*kResASig0KSt0=*/ 0.040,
/*kResXi0KSt0=*/  0.014, /*kResAXi0AKSt0=*/  0.013, /*kResXi0AKSt0=*/  0.014, /*kResAXi0KSt0=*/  0.013,
/*kResXiCKSt0=*/  0.018, /*kResAXiCAKSt0=*/  0.017, /*kResXiCAKSt0=*/  0.018, /*kResAXiCKSt0=*/  0.016,

/*kResSig0K0=*/   0.120, /*kResASig0K0=*/   0.121,
/*kResXi0K0=*/    0.042, /*kResAXi0K0=*/    0.039,
/*kResXiCK0=*/    0.054, /*kResAXiCK0=*/    0.050,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.056, /*kResASigStMK0=*/ 0.054,
/*kResSigStMK0=*/ 0.050, /*kResASigStPK0=*/ 0.053,
/*kResSigSt0K0=*/ 0.050, /*kResASigSt0K0=*/ 0.047,

/*kResLamKSt0ToLamK0=*/  0.021, /*kResALamAKSt0ToALamK0=*/  0.022,
/*kResSig0KSt0ToLamK0=*/ 0.019, /*kResASig0AKSt0ToALamK0=*/ 0.019,
/*kResXi0KSt0ToLamK0=*/  0.006, /*kResAXi0AKSt0ToALamK0=*/  0.006,
/*kResXiCKSt0ToLamK0=*/  0.008, /*kResAXiCAKSt0ToALamK0=*/  0.008
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
/*kLamK0=*/   0.145,   /*kALamK0=*/   0.145,
/*kLamKchP=*/ 0.134,   /*kALamKchM=*/ 0.133,  /*kLamKchM=*/ 0.133, /*kALamKchP=*/ 0.134,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.120, /*kResASig0KchM=*/   0.115, /*kResSig0KchM=*/   0.119, /*kResASig0KchP=*/  0.117,
/*kResXi0KchP=*/    0.042, /*kResAXi0KchM=*/    0.037, /*kResXi0KchM=*/    0.041, /*kResAXi0KchP=*/   0.037,
/*kResXiCKchP=*/    0.054, /*kResAXiCKchM=*/    0.048, /*kResXiCKchM=*/    0.053, /*kResAXiCKchP=*/   0.048,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.000, /*kResASigStMKchM=*/ 0.000, /*kResSigStPKchM=*/ 0.000, /*kResASigStMKchP=*/ 0.000,
/*kResSigStMKchP=*/ 0.000, /*kResASigStPKchM=*/ 0.000, /*kResSigStMKchM=*/ 0.000, /*kResASigStPKchP=*/ 0.000,
/*kResSigSt0KchP=*/ 0.000, /*kResASigSt0KchM=*/ 0.000, /*kResSigSt0KchM=*/ 0.000, /*kResASigSt0KchP=*/ 0.000,

/*kResLamKSt0=*/  0.000, /*kResALamAKSt0=*/  0.000, /*kResLamAKSt0=*/  0.000, /*kResALamKSt0=*/  0.000,
/*kResSig0KSt0=*/ 0.000, /*kResASig0AKSt0=*/ 0.000, /*kResSig0AKSt0=*/ 0.000, /*kResASig0KSt0=*/ 0.000,
/*kResXi0KSt0=*/  0.000, /*kResAXi0AKSt0=*/  0.000, /*kResXi0AKSt0=*/  0.000, /*kResAXi0KSt0=*/  0.000,
/*kResXiCKSt0=*/  0.000, /*kResAXiCAKSt0=*/  0.000, /*kResXiCAKSt0=*/  0.000, /*kResAXiCKSt0=*/  0.000,

/*kResSig0K0=*/   0.130, /*kResASig0K0=*/   0.126,
/*kResXi0K0=*/    0.045, /*kResAXi0K0=*/    0.041,
/*kResXiCK0=*/    0.058, /*kResAXiCK0=*/    0.052,
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
/*kLamK0=*/   0.197,   /*kALamK0=*/   0.194,
/*kLamKchP=*/ 0.185,   /*kALamKchM=*/ 0.181,  /*kLamKchM=*/ 0.184, /*kALamKchP=*/ 0.182,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.119, /*kResASig0KchM=*/   0.115, /*kResSig0KchM=*/   0.118, /*kResASig0KchP=*/  0.116,
/*kResXi0KchP=*/    0.041, /*kResAXi0KchM=*/    0.037, /*kResXi0KchM=*/    0.041, /*kResAXi0KchP=*/   0.037,
/*kResXiCKchP=*/    0.053, /*kResAXiCKchM=*/    0.048, /*kResXiCKchM=*/    0.053, /*kResAXiCKchP=*/   0.048,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.000, /*kResASigStMKchM=*/ 0.000, /*kResSigStPKchM=*/ 0.000, /*kResASigStMKchP=*/ 0.000,
/*kResSigStMKchP=*/ 0.000, /*kResASigStPKchM=*/ 0.000, /*kResSigStMKchM=*/ 0.000, /*kResASigStPKchP=*/ 0.000,
/*kResSigSt0KchP=*/ 0.000, /*kResASigSt0KchM=*/ 0.000, /*kResSigSt0KchM=*/ 0.000, /*kResASigSt0KchP=*/ 0.000,

/*kResLamKSt0=*/  0.000, /*kResALamAKSt0=*/  0.000, /*kResLamAKSt0=*/  0.000, /*kResALamKSt0=*/  0.000,
/*kResSig0KSt0=*/ 0.000, /*kResASig0AKSt0=*/ 0.000, /*kResSig0AKSt0=*/ 0.000, /*kResASig0KSt0=*/ 0.000,
/*kResXi0KSt0=*/  0.000, /*kResAXi0AKSt0=*/  0.000, /*kResXi0AKSt0=*/  0.000, /*kResAXi0KSt0=*/  0.000,
/*kResXiCKSt0=*/  0.000, /*kResAXiCAKSt0=*/  0.000, /*kResXiCAKSt0=*/  0.000, /*kResAXiCKSt0=*/  0.000,

/*kResSig0K0=*/   0.128, /*kResASig0K0=*/   0.125,
/*kResXi0K0=*/    0.044, /*kResAXi0K0=*/    0.040,
/*kResXiCK0=*/    0.058, /*kResAXiCK0=*/    0.052,
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
/*kLamK0=*/   0.277,   /*kALamK0=*/   0.275,
/*kLamKchP=*/ 0.268,   /*kALamKchM=*/ 0.265,  /*kLamKchM=*/ 0.268, /*kALamKchP=*/ 0.265,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.117, /*kResASig0KchM=*/   0.114, /*kResSig0KchM=*/   0.116, /*kResASig0KchP=*/  0.115,
/*kResXi0KchP=*/    0.041, /*kResAXi0KchM=*/    0.037, /*kResXi0KchM=*/    0.040, /*kResAXi0KchP=*/   0.037,
/*kResXiCKchP=*/    0.052, /*kResAXiCKchM=*/    0.047, /*kResXiCKchM=*/    0.052, /*kResAXiCKchP=*/   0.048,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.000, /*kResASigStMKchM=*/ 0.000, /*kResSigStPKchM=*/ 0.000, /*kResASigStMKchP=*/ 0.000,
/*kResSigStMKchP=*/ 0.000, /*kResASigStPKchM=*/ 0.000, /*kResSigStMKchM=*/ 0.000, /*kResASigStPKchP=*/ 0.000,
/*kResSigSt0KchP=*/ 0.000, /*kResASigSt0KchM=*/ 0.000, /*kResSigSt0KchM=*/ 0.000, /*kResASigSt0KchP=*/ 0.000,

/*kResLamKSt0=*/  0.000, /*kResALamAKSt0=*/  0.000, /*kResLamAKSt0=*/  0.000, /*kResALamKSt0=*/  0.000,
/*kResSig0KSt0=*/ 0.000, /*kResASig0AKSt0=*/ 0.000, /*kResSig0AKSt0=*/ 0.000, /*kResASig0KSt0=*/ 0.000,
/*kResXi0KSt0=*/  0.000, /*kResAXi0AKSt0=*/  0.000, /*kResXi0AKSt0=*/  0.000, /*kResAXi0KSt0=*/  0.000,
/*kResXiCKSt0=*/  0.000, /*kResAXiCAKSt0=*/  0.000, /*kResXiCAKSt0=*/  0.000, /*kResAXiCKSt0=*/  0.000,

/*kResSig0K0=*/   0.127, /*kResASig0K0=*/   0.124,
/*kResXi0K0=*/    0.044, /*kResAXi0K0=*/    0.040,
/*kResXiCK0=*/    0.057, /*kResAXiCK0=*/    0.052,
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
/*kLamK0=*/   0.518,   /*kALamK0=*/   0.520,
/*kLamKchP=*/ 0.502,   /*kALamKchM=*/ 0.502,  /*kLamKchM=*/ 0.501, /*kALamKchP=*/ 0.503,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.112, /*kResASig0KchM=*/   0.111, /*kResSig0KchM=*/   0.111, /*kResASig0KchP=*/  0.112,
/*kResXi0KchP=*/    0.039, /*kResAXi0KchM=*/    0.036, /*kResXi0KchM=*/    0.038, /*kResAXi0KchP=*/   0.036,
/*kResXiCKchP=*/    0.050, /*kResAXiCKchM=*/    0.046, /*kResXiCKchM=*/    0.050, /*kResAXiCKchP=*/   0.046,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.000, /*kResASigStMKchM=*/ 0.000, /*kResSigStPKchM=*/ 0.000, /*kResASigStMKchP=*/ 0.000,
/*kResSigStMKchP=*/ 0.000, /*kResASigStPKchM=*/ 0.000, /*kResSigStMKchM=*/ 0.000, /*kResASigStPKchP=*/ 0.000,
/*kResSigSt0KchP=*/ 0.000, /*kResASigSt0KchM=*/ 0.000, /*kResSigSt0KchM=*/ 0.000, /*kResASigSt0KchP=*/ 0.000,

/*kResLamKSt0=*/  0.000, /*kResALamAKSt0=*/  0.000, /*kResLamAKSt0=*/  0.000, /*kResALamKSt0=*/  0.000,
/*kResSig0KSt0=*/ 0.000, /*kResASig0AKSt0=*/ 0.000, /*kResSig0AKSt0=*/ 0.000, /*kResASig0KSt0=*/ 0.000,
/*kResXi0KSt0=*/  0.000, /*kResAXi0AKSt0=*/  0.000, /*kResXi0AKSt0=*/  0.000, /*kResAXi0KSt0=*/  0.000,
/*kResXiCKSt0=*/  0.000, /*kResAXiCAKSt0=*/  0.000, /*kResXiCAKSt0=*/  0.000, /*kResAXiCKSt0=*/  0.000,

/*kResSig0K0=*/   0.121, /*kResASig0K0=*/   0.121,
/*kResXi0K0=*/    0.042, /*kResAXi0K0=*/    0.039,
/*kResXiCK0=*/    0.054, /*kResAXiCK0=*/    0.050,
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
/*kLamK0=*/   0.543,   /*kALamK0=*/   0.544,
/*kLamKchP=*/ 0.527,   /*kALamKchM=*/ 0.526,  /*kLamKchM=*/ 0.526, /*kALamKchP=*/ 0.527,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.111, /*kResASig0KchM=*/   0.110, /*kResSig0KchM=*/   0.110, /*kResASig0KchP=*/  0.111,
/*kResXi0KchP=*/    0.039, /*kResAXi0KchM=*/    0.035, /*kResXi0KchM=*/    0.038, /*kResAXi0KchP=*/   0.036,
/*kResXiCKchP=*/    0.050, /*kResAXiCKchM=*/    0.046, /*kResXiCKchM=*/    0.050, /*kResAXiCKchP=*/   0.046,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.000, /*kResASigStMKchM=*/ 0.000, /*kResSigStPKchM=*/ 0.000, /*kResASigStMKchP=*/ 0.000,
/*kResSigStMKchP=*/ 0.000, /*kResASigStPKchM=*/ 0.000, /*kResSigStMKchM=*/ 0.000, /*kResASigStPKchP=*/ 0.000,
/*kResSigSt0KchP=*/ 0.000, /*kResASigSt0KchM=*/ 0.000, /*kResSigSt0KchM=*/ 0.000, /*kResASigSt0KchP=*/ 0.000,

/*kResLamKSt0=*/  0.000, /*kResALamAKSt0=*/  0.000, /*kResLamAKSt0=*/  0.000, /*kResALamKSt0=*/  0.000,
/*kResSig0KSt0=*/ 0.000, /*kResASig0AKSt0=*/ 0.000, /*kResSig0AKSt0=*/ 0.000, /*kResASig0KSt0=*/ 0.000,
/*kResXi0KSt0=*/  0.000, /*kResAXi0AKSt0=*/  0.000, /*kResXi0AKSt0=*/  0.000, /*kResAXi0KSt0=*/  0.000,
/*kResXiCKSt0=*/  0.000, /*kResAXiCAKSt0=*/  0.000, /*kResXiCAKSt0=*/  0.000, /*kResAXiCKSt0=*/  0.000,

/*kResSig0K0=*/   0.120, /*kResASig0K0=*/   0.120,
/*kResXi0K0=*/    0.042, /*kResAXi0K0=*/    0.039,
/*kResXiCK0=*/    0.054, /*kResAXiCK0=*/    0.050,
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
/*kLamK0=*/   0.568,   /*kALamK0=*/   0.570,
/*kLamKchP=*/ 0.565,   /*kALamKchM=*/ 0.566,  /*kLamKchM=*/ 0.565, /*kALamKchP=*/ 0.567,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.111, /*kResASig0KchM=*/   0.110, /*kResSig0KchM=*/   0.110, /*kResASig0KchP=*/  0.111,
/*kResXi0KchP=*/    0.038, /*kResAXi0KchM=*/    0.035, /*kResXi0KchM=*/    0.038, /*kResAXi0KchP=*/   0.036,
/*kResXiCKchP=*/    0.050, /*kResAXiCKchM=*/    0.045, /*kResXiCKchM=*/    0.049, /*kResAXiCKchP=*/   0.046,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.000, /*kResASigStMKchM=*/ 0.000, /*kResSigStPKchM=*/ 0.000, /*kResASigStMKchP=*/ 0.000,
/*kResSigStMKchP=*/ 0.000, /*kResASigStPKchM=*/ 0.000, /*kResSigStMKchM=*/ 0.000, /*kResASigStPKchP=*/ 0.000,
/*kResSigSt0KchP=*/ 0.000, /*kResASigSt0KchM=*/ 0.000, /*kResSigSt0KchM=*/ 0.000, /*kResASigSt0KchP=*/ 0.000,

/*kResLamKSt0=*/  0.000, /*kResALamAKSt0=*/  0.000, /*kResLamAKSt0=*/  0.000, /*kResALamKSt0=*/  0.000,
/*kResSig0KSt0=*/ 0.000, /*kResASig0AKSt0=*/ 0.000, /*kResSig0AKSt0=*/ 0.000, /*kResASig0KSt0=*/ 0.000,
/*kResXi0KSt0=*/  0.000, /*kResAXi0AKSt0=*/  0.000, /*kResXi0AKSt0=*/  0.000, /*kResAXi0KSt0=*/  0.000,
/*kResXiCKSt0=*/  0.000, /*kResAXiCAKSt0=*/  0.000, /*kResXiCAKSt0=*/  0.000, /*kResAXiCKSt0=*/  0.000,

/*kResSig0K0=*/   0.120, /*kResASig0K0=*/   0.120,
/*kResXi0K0=*/    0.041, /*kResAXi0K0=*/    0.039,
/*kResXiCK0=*/    0.054, /*kResAXiCK0=*/    0.050,
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

