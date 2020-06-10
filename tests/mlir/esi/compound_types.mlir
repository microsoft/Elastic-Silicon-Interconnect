// RUN: esic %s | esic | FileCheck %s

!exStruct1 = type !esi.struct<
    {ui1,   "int1"},
    {f32,   "float1"}
>
!exStruct2 = type !esi.struct<{si2,"sint2"}, {f32, "float1"}>

!exUnion1 = type !esi.union<{!exStruct1,"struct1"}, {f32, "float1"}>

// CHECK-LABEL: module
module {
    // CHECK-LABEL: func @struct1(%{{.*}}: !esi.struct<{ui1,"int1"},{f32,"float1"}>)
    func @struct1(%A: !exStruct1) {
        return
    }

    // CHECK-LABEL: func @union1(%arg0: !esi.union<{!esi.struct<{ui1,"int1"},{f32,"float1"}>,"struct1"},{f32,"float1"}>)
    func @union1(%A: !exUnion1) {
        return
    }
}
