#ifndef __ESI_CAPNP_CONVERT__
#define __ESI_CAPNP_CONVERT__

#include <mlir/IR/Types.h>
#include <llvm/Support/Error.h>
#include <capnp/schema.h>
#include <capnp/schema-parser.h>
#include <vector>
#include <map>
#include <memory>

namespace esi {
namespace capnp {

    /// These are all the annotations which ESI knows about. Keep these in
    /// sync with the ESICoreAnnotations.capnp file.
    /// </summary>
    struct Annotations {

        template<int NUM>
        static bool __addToNameList() { return true; }
        #define Annotation(NAME, ANNOTATION_ID, NUM) \
            constexpr static uint64_t NAME = ANNOTATION_ID; \
            template<> \
            static bool __addToNameList<NUM>() { \
                __addToNameList<NUM+1>(); \
                idToName[ANNOTATION_ID] = #NAME; \
                return true; \
            }

    private:
        static std::map<uint64_t, std::string> idToName;

    public:
        Annotation(BITS,                    0xac112269228ad38c, 0);
        Annotation(INLINE,                  0x83f1b26b0188c1bb, 1);
        Annotation(ARRAY,                   0x93ce43d5fd6478ee, 2);
        Annotation(C_UNION,                 0xed2e4e8a596d00a5, 3);
        Annotation(FIXED_LIST,              0x8e0d4f6349687e9b, 4);
        Annotation(FIXED,                   0xb0aef92d8eed92a5, 5);
        Annotation(FIXED_POINT,             0x82adb6b7cba4ca97, 6);
        Annotation(FIXED_POINT_VALUE,       0x81eebdd3a9e24c9d, 7);
        Annotation(FLOAT,                   0xc06dd6e3ee4392de, 8);
        Annotation(FLOATING_POINT,          0xa9e717a24fd51f71, 9);
        Annotation(FLOATING_POINT_VALUE,    0xaf862f0ea103797c, 10);
        Annotation(OFFSET,                  0xcdbc3408a9217752, 11);
        Annotation(HWOFFSET,                0xf7afdfd9eb5a7d15, 12);
        #undef Annotation

        static bool contains(uint64_t id) {
            return idToName.find(id) != idToName.end();
        }
        static const std::string& nameof(uint64_t id) {
            auto nameIter = idToName.find(id);
            if (nameIter == idToName.end())
                return "";
            return nameIter->second;
        }
    };

    llvm::Error ConvertToESI(
        ::mlir::MLIRContext*,
        ::capnp::ParsedSchema& rootSchema,
        ::std::vector<mlir::Type>& outputTypes);
}
}
#endif
