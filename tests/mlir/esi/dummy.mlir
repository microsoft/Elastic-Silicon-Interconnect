// RUN: esic %s | esic | FileCheck %s

// CHECK-LABEL: module
module {
    // CHECK-LABEL: func @bar(%{{.*}}: !esi.compound<true,3,4>)
    func @bar(%A: !esi.compound<true, 3, 4>) {
        // %0 = constant (true, 3, 10) : esi.compound
        %1 = "esi.cast_compound"(%A) : (!esi.compound<true, 3, 4>) ->  i1
        return
    }
}
