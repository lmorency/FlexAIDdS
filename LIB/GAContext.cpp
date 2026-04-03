// GAContext.cpp — Out-of-line destructor and move-assignment
//
// The destructor must be out-of-line because it deletes tqcm, which
// is a forward-declared type. When Eigen is available, TurboQuant.h
// provides the full definition; otherwise tqcm is never populated.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

#include "GAContext.h"

#ifdef FLEXAIDS_HAS_EIGEN
#include "TurboQuant.h"  // provides full definition of QuantizedContactMatrix
#endif

GAContext::~GAContext() {
#ifdef FLEXAIDS_HAS_EIGEN
    delete tqcm;
#endif
    // Without Eigen, tqcm is always nullptr — no action needed
}

GAContext& GAContext::operator=(GAContext&& o) noexcept {
    if (this != &o) {
#ifdef FLEXAIDS_HAS_EIGEN
        delete tqcm;
#endif
        gen_id = o.gen_id;
        nrejected = o.nrejected;
        dispatch_logged = o.dispatch_logged;
        tqcm = o.tqcm;
        tqcm_ntypes = o.tqcm_ntypes;
        o.tqcm = nullptr;
    }
    return *this;
}
