#ifndef __ESI_COMPOUNDS_HPP__
#define __ESI_COMPOUNDS_HPP__

#include <type_traits>
#include "EsiInt.hpp"

template<bool Signed, int Exp, int Mant>
struct EsiFloatingPoint
{
    std::enable_if<Signed> { bool sign; }
    esi_int<Exp> exp;
    esi_int<Mant> mant;
};

#endif // __ESI_COMPOUNDS_HPP__
