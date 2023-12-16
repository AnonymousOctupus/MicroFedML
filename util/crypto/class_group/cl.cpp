// g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) -o classgroup$(python3-config --extension-suffix) cl.cpp /usr/lib/x86_64-linux-gnu/libcrypto.so /usr/lib/x86_64-linux-gnu/libgmpxx.so /usr/lib/x86_64-linux-gnu/libcrypto.so

#include "bicycl.hpp"
#include <chrono>
#include <gmp.h>
#include <stdlib.h>

const BICYCL::SecLevel seclevel = 128;

#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>


namespace py = pybind11;

class ClassGroup {
private:
    BICYCL::Mpz q = BICYCL::Mpz("108442577883873400988079448518214027771297265527489521437545791762477486455369");
    BICYCL::RandGen randgen;
    BICYCL::CL_HSMqk CL = BICYCL::CL_HSMqk (q, 1, seclevel, randgen);
public:
    ClassGroup(/* args */){
        
        // std::cout << "constructor called" << std::endl;

        // a not so random seed to initialize Bicycl PRNG
        randgen.set_seed ((long) 1); 

        // // Let us use a 256 bits prime q for the message space Z/qZ
        // // BICYCL::Mpz q ("108442577883873400988079448518214027771297265527489521437545791762477486455369"); 
        // q = BICYCL::Mpz("108442577883873400988079448518214027771297265527489521437545791762477486455369"); 

        // // ########## CL HSM setup for message q^k
        // // input : q, k=1, seclevel, randgen,

        // // BICYCL::CL_HSMqk CL (q, 1, seclevel, randgen);
        // CL = BICYCL::CL_HSMqk (q, 1, seclevel, randgen);
        
        // std::cout << CL << std::endl;

    }


    ~ClassGroup(){
        
    }

    BICYCL::QFI mul(BICYCL::QFI a1, BICYCL::QFI a2){
        BICYCL::QFI result;
        CL.Cl_Delta().nucomp(result, a1, a2);
        // std::cout << "result = " << result << std::endl;
        return result;
    }


    BICYCL::QFI div(BICYCL::QFI a1, BICYCL::QFI a2){
        BICYCL::QFI result;
        CL.Cl_Delta().nucompinv(result, a1, a2);
        // std::cout << "result = " << result << std::endl;
        return result;
    }


    BICYCL::QFI power(BICYCL::QFI a1, int e){
        std::string s = std::to_string(e);
        BICYCL::Mpz m (s);
        BICYCL::QFI result;
        CL.Cl_Delta().nupow(result, a1, m);
        // std::cout << "result = " << result << std::endl;
        return result;
    }


    BICYCL::QFI power_g(int e){
        std::string s = std::to_string(e);
        BICYCL::Mpz m (s);
        BICYCL::QFI result;
        CL.power_of_h(result, m);
        // result.eval(&m)
        // return result.eval();
        // std::cout << "result = " << result << std::endl;
        return result;
    }


    BICYCL::QFI power_f(int e){
        std::string s = std::to_string(e);
        BICYCL::Mpz m (s);
        BICYCL::QFI result = CL.power_of_f(m);
        // result.eval(&m)
        // return result.eval();
        // std::cout << "result = " << result << std::endl;
        return result;
    }

    

    BICYCL::Mpz log_f(BICYCL::QFI m){
        BICYCL::Mpz result = CL.dlog_in_F(m);
        // std::string s = std::to_string(result);
        // std::cout << "result = " << result << std::endl;
        return result;
    }


    std::string to_string_mpz(BICYCL::Mpz m){
        std::ostringstream stm;
        stm << m;
        return stm.str();
    }

    std::string to_string_qfi(BICYCL::QFI q){
        std::ostringstream stm;
        stm << q;
        return stm.str();
    }


};

int add(int i, int j) {
    return i + j;
}


PYBIND11_MODULE(classgroup, m) {
    m.doc() = "class group plugin"; // Optional module docstring
    m.def("add", &add, "A function that adds two numbers");
    // m.def("test", &test, "test funcs");
    py::class_<ClassGroup>(m, "ClassGroup")
        .def(py::init())
        .def("mul", &ClassGroup::mul)
        .def("div", &ClassGroup::div)
        .def("power", &ClassGroup::power)
        .def("power_g", &ClassGroup::power_g)
        .def("power_f", &ClassGroup::power_f)
        .def("log_f", &ClassGroup::log_f)
        .def("to_string_mpz", &ClassGroup::to_string_mpz)
        .def("to_string_qfi", &ClassGroup::to_string_qfi);

    py::class_<BICYCL::Mpz>(m, "Mpz")
        .def(py::init());
        // .def(py::init(const Mpz &))
        // .def("add", &BICYCL::Mpz::add);

    py::class_<BICYCL::QFI>(m, "QFI")
        .def(py::init());
}














// ======================================

int test(){
    BICYCL::RandGen randgen;
  // a not so random seed to initialize Bicycl PRNG
  randgen.set_seed ((long) 1); 

  // Let us use a 256 bits prime q for the message space Z/qZ
  BICYCL::Mpz q ("108442577883873400988079448518214027771297265527489521437545791762477486455369"); 

  // ########## CL HSM setup for message q^k
  // input : q, k=1, seclevel, randgen,

  BICYCL::CL_HSMqk CL (q, 1, seclevel, randgen);

  std::cout << CL << std::endl;

  // ########## Keygen
  
  BICYCL::CL_HSMqk::SecretKey sk = CL.keygen (randgen);
  BICYCL::CL_HSMqk::PublicKey pk = CL.keygen (sk);

  std::cout << "sk = " << sk << std::endl;
  std::cout << "pk = " << pk << std::endl;

  // ########## Encryption by hand
  // c4 = h^r , pk^r f^m where pk = h^sk

  BICYCL::QFI c4_1, c4_2, tmp;
//   BICYCL::Mpz m ("14412432141242141");

    // Generating a random integer m of bitlen bit length
  int bitlen = 30;
  int iterations = 1000000;
  int record_time = 0;

  for (int i = 0; i < iterations; i++){
    // int r = rand();
    // r = r & (1 << bitlen - 1);
    unsigned long long int r;
    r = rand();
    // r = r << 32;
    r += rand();
    std::string s = std::to_string(r);
    BICYCL::Mpz m (s);

    // f^m computed with a special function using a simple direct formula
    tmp = CL.power_of_f(m);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    BICYCL::Mpz result = CL.dlog_in_F(tmp);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    record_time = record_time + std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
 
    
  }
  std::cout << "iterations = " << iterations  << std::endl;
  std::cout << "Time difference = " << record_time << "[ns]" << std::endl;
}

namespace py = pybind11;

// PYBIND11_MODULE(example, m) {
//     m.doc() = "pybind11 example plugin"; // Optional module docstring
//     m.def("add", &add, "A function that adds two numbers");
//     m.def("test", &test, "test funcs");
// }
