// RUN: esic %s | esic | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar(%A: !esi.compound<true, 3, 4>) {
        // %0 = constant (true, 3, 10) : esi.compound
        %1 = "esi.cast_compound"(%A) : (!esi.compound<true, 3, 4>) ->  i1
        return
    }
}
