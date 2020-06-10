// RUN: esic %s | esic | FileCheck %s

!exStruct1 = type !esi.struct<{ui1,"int1"}, {f32, "float1"}>
!exStruct2 = type !esi.struct<{si2,"sint2"}, {f32, "float1"}>

// CHECK-LABEL: module
module {
    // CHECK-LABEL: func @list1(%{{.*}}: !exStruct1)
    func @struct1(%A: !exStruct1) {
        return
    }
}
