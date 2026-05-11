# 1 "../ByteTrack/lapjv.cpp"
# 1 "<built-in>" 1
# 1 "<built-in>" 3
# 481 "<built-in>" 3
# 1 "<command line>" 1
# 1 "<built-in>" 2
# 1 "../ByteTrack/lapjv.cpp" 2
# 1 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\stdio.h" 1 3
# 101 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\stdio.h" 3
# 1 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\__config" 1 3
# 44 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\__config" 3
# 1 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\__configuration/abi.h" 1 3
# 18 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\__configuration/abi.h" 3
# 1 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\__configuration/compiler.h" 1 3
# 20 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\__configuration/compiler.h" 3
# 19 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\__configuration/abi.h" 2 3
# 1 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\__configuration/platform.h" 1 3
# 20 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\__configuration/platform.h" 3
# 20 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\__configuration/abi.h" 2 3
# 23 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\__configuration/abi.h" 3
# 45 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\__config" 2 3
# 1 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\__configuration/availability.h" 1 3
# 14 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\__configuration/availability.h" 3
# 1 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\__configuration/language.h" 1 3
# 20 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\__configuration/language.h" 3
# 15 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\__configuration/availability.h" 2 3
# 18 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\__configuration/availability.h" 3
# 46 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\__config" 2 3
# 51 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\__config" 3
# 102 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\stdio.h" 2 3
# 105 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\stdio.h" 3



# 1 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 1 3
# 41 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
      namespace std {

        extern "C" {
# 53 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
    typedef unsigned int size_t;
# 68 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
    typedef __builtin_va_list __va_list;
# 87 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
typedef struct __fpos_t_struct {
    unsigned long long int __pos;





    struct {
        unsigned int __state1, __state2;
    } __mbstate;
} fpos_t;
# 108 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
typedef struct __FILE FILE;
# 119 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
struct __FILE {
    union {
        long __FILE_alignment;



        char __FILE_size[84];

    } __FILE_opaque;
};
# 138 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern FILE __stdin, __stdout, __stderr;
extern FILE *__aeabi_stdin, *__aeabi_stdout, *__aeabi_stderr;
# 224 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int remove(const char * ) __attribute__((__nonnull__(1)));







extern __attribute__((__nothrow__)) int rename(const char * , const char * ) __attribute__((__nonnull__(1,2)));
# 243 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) FILE *tmpfile(void);






extern __attribute__((__nothrow__)) char *tmpnam(char * );
# 265 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int fclose(FILE * ) __attribute__((__nonnull__(1)));
# 275 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int fflush(FILE * );
# 285 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) FILE *fopen(const char * __restrict ,
                           const char * __restrict ) __attribute__((__nonnull__(1,2)));
# 329 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) FILE *freopen(const char * __restrict ,
                    const char * __restrict ,
                    FILE * __restrict ) __attribute__((__nonnull__(2,3)));
# 342 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) void setbuf(FILE * __restrict ,
                    char * __restrict ) __attribute__((__nonnull__(1)));






extern __attribute__((__nothrow__)) int setvbuf(FILE * __restrict ,
                   char * __restrict ,
                   int , size_t ) __attribute__((__nonnull__(1)));
# 370 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
#pragma __printf_args
extern __attribute__((__nothrow__)) int fprintf(FILE * __restrict ,
                    const char * __restrict , ...) __attribute__((__nonnull__(1,2)));
# 393 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
#pragma __printf_args
extern __attribute__((__nothrow__)) int _fprintf(FILE * __restrict ,
                     const char * __restrict , ...) __attribute__((__nonnull__(1,2)));





#pragma __printf_args
extern __attribute__((__nothrow__)) int printf(const char * __restrict , ...) __attribute__((__nonnull__(1)));






#pragma __printf_args
extern __attribute__((__nothrow__)) int _printf(const char * __restrict , ...) __attribute__((__nonnull__(1)));





#pragma __printf_args
extern __attribute__((__nothrow__)) int sprintf(char * __restrict , const char * __restrict , ...) __attribute__((__nonnull__(1,2)));








#pragma __printf_args
extern __attribute__((__nothrow__)) int _sprintf(char * __restrict , const char * __restrict , ...) __attribute__((__nonnull__(1,2)));






#pragma __printf_args
extern __attribute__((__nothrow__)) int __ARM_snprintf(char * __restrict , size_t ,
                     const char * __restrict , ...) __attribute__((__nonnull__(3)));


#pragma __printf_args
extern __attribute__((__nothrow__)) int snprintf(char * __restrict , size_t ,
                     const char * __restrict , ...) __attribute__((__nonnull__(3)));
# 460 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
#pragma __printf_args
extern __attribute__((__nothrow__)) int _snprintf(char * __restrict , size_t ,
                      const char * __restrict , ...) __attribute__((__nonnull__(3)));





#pragma __scanf_args
extern __attribute__((__nothrow__)) int fscanf(FILE * __restrict ,
                    const char * __restrict , ...) __attribute__((__nonnull__(1,2)));
# 503 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
#pragma __scanf_args
extern __attribute__((__nothrow__)) int _fscanf(FILE * __restrict ,
                     const char * __restrict , ...) __attribute__((__nonnull__(1,2)));





#pragma __scanf_args
extern __attribute__((__nothrow__)) int scanf(const char * __restrict , ...) __attribute__((__nonnull__(1)));








#pragma __scanf_args
extern __attribute__((__nothrow__)) int _scanf(const char * __restrict , ...) __attribute__((__nonnull__(1)));





#pragma __scanf_args
extern __attribute__((__nothrow__)) int sscanf(const char * __restrict ,
                    const char * __restrict , ...) __attribute__((__nonnull__(1,2)));
# 541 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
#pragma __scanf_args
extern __attribute__((__nothrow__)) int _sscanf(const char * __restrict ,
                     const char * __restrict , ...) __attribute__((__nonnull__(1,2)));







extern __attribute__((__nothrow__)) int vfscanf(FILE * __restrict , const char * __restrict , __va_list) __attribute__((__nonnull__(1,2)));
extern __attribute__((__nothrow__)) int vscanf(const char * __restrict , __va_list) __attribute__((__nonnull__(1)));
extern __attribute__((__nothrow__)) int vsscanf(const char * __restrict , const char * __restrict , __va_list) __attribute__((__nonnull__(1,2)));

extern __attribute__((__nothrow__)) int _vfscanf(FILE * __restrict , const char * __restrict , __va_list) __attribute__((__nonnull__(1,2)));
extern __attribute__((__nothrow__)) int _vscanf(const char * __restrict , __va_list) __attribute__((__nonnull__(1)));
extern __attribute__((__nothrow__)) int _vsscanf(const char * __restrict , const char * __restrict , __va_list) __attribute__((__nonnull__(1,2)));
extern __attribute__((__nothrow__)) int __ARM_vsscanf(const char * __restrict , const char * __restrict , __va_list) __attribute__((__nonnull__(1,2)));

extern __attribute__((__nothrow__)) int vprintf(const char * __restrict , __va_list ) __attribute__((__nonnull__(1)));







extern __attribute__((__nothrow__)) int _vprintf(const char * __restrict , __va_list ) __attribute__((__nonnull__(1)));





extern __attribute__((__nothrow__)) int vfprintf(FILE * __restrict ,
                    const char * __restrict , __va_list ) __attribute__((__nonnull__(1,2)));
# 584 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int vsprintf(char * __restrict ,
                     const char * __restrict , __va_list ) __attribute__((__nonnull__(1,2)));
# 594 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int __ARM_vsnprintf(char * __restrict , size_t ,
                     const char * __restrict , __va_list ) __attribute__((__nonnull__(3)));

extern __attribute__((__nothrow__)) int vsnprintf(char * __restrict , size_t ,
                     const char * __restrict , __va_list ) __attribute__((__nonnull__(3)));
# 609 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int _vsprintf(char * __restrict ,
                      const char * __restrict , __va_list ) __attribute__((__nonnull__(1,2)));





extern __attribute__((__nothrow__)) int _vfprintf(FILE * __restrict ,
                     const char * __restrict , __va_list ) __attribute__((__nonnull__(1,2)));





extern __attribute__((__nothrow__)) int _vsnprintf(char * __restrict , size_t ,
                      const char * __restrict , __va_list ) __attribute__((__nonnull__(3)));
# 635 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
#pragma __printf_args
extern __attribute__((__nothrow__)) int __ARM_asprintf(char ** , const char * __restrict , ...) __attribute__((__nonnull__(2)));
extern __attribute__((__nothrow__)) int __ARM_vasprintf(char ** , const char * __restrict , __va_list ) __attribute__((__nonnull__(2)));
# 649 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int fgetc(FILE * ) __attribute__((__nonnull__(1)));
# 659 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) char *fgets(char * __restrict , int ,
                    FILE * __restrict ) __attribute__((__nonnull__(1,3)));
# 673 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int fputc(int , FILE * ) __attribute__((__nonnull__(2)));
# 683 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int fputs(const char * __restrict , FILE * __restrict ) __attribute__((__nonnull__(1,2)));






extern __attribute__((__nothrow__)) int getc(FILE * ) __attribute__((__nonnull__(1)));
# 701 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
    inline int getchar() { return getc((&::std:: __stdin)); }
# 713 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) char *gets(char * ) __attribute__((__nonnull__(1)));
# 725 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int putc(int , FILE * ) __attribute__((__nonnull__(2)));
# 734 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
    inline int putchar(int __c) { return putc(__c, (&::std:: __stdout)); }
# 744 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int puts(const char * ) __attribute__((__nonnull__(1)));







extern __attribute__((__nothrow__)) int ungetc(int , FILE * ) __attribute__((__nonnull__(2)));
# 778 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) size_t fread(void * __restrict ,
                    size_t , size_t , FILE * __restrict ) __attribute__((__nonnull__(1,4)));
# 794 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) size_t __fread_bytes_avail(void * __restrict ,
                    size_t , FILE * __restrict ) __attribute__((__nonnull__(1,3)));
# 810 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) size_t fwrite(const void * __restrict ,
                    size_t , size_t , FILE * __restrict ) __attribute__((__nonnull__(1,4)));
# 822 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int fgetpos(FILE * __restrict , fpos_t * __restrict ) __attribute__((__nonnull__(1,2)));
# 833 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int fseek(FILE * , long int , int ) __attribute__((__nonnull__(1)));
# 850 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int fsetpos(FILE * __restrict , const fpos_t * __restrict ) __attribute__((__nonnull__(1,2)));
# 863 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) long int ftell(FILE * ) __attribute__((__nonnull__(1)));
# 877 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) void rewind(FILE * ) __attribute__((__nonnull__(1)));
# 886 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) void clearerr(FILE * ) __attribute__((__nonnull__(1)));







extern __attribute__((__nothrow__)) int feof(FILE * ) __attribute__((__nonnull__(1)));




extern __attribute__((__nothrow__)) int ferror(FILE * ) __attribute__((__nonnull__(1)));




extern __attribute__((__nothrow__)) void perror(const char * );
# 917 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int _fisatty(FILE * ) __attribute__((__nonnull__(1)));



extern __attribute__((__nothrow__)) void __use_no_semihosting_swi(void);
extern __attribute__((__nothrow__)) void __use_no_semihosting(void);
# 932 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
        }
      }
# 942 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
    using ::std::size_t;
    using ::std::fpos_t;
    using ::std::FILE;
    using ::std::remove;
    using ::std::rename;
    using ::std::tmpfile;
    using ::std::tmpnam;
    using ::std::fclose;
    using ::std::fflush;
    using ::std::fopen;
    using ::std::freopen;
    using ::std::setbuf;
    using ::std::setvbuf;
    using ::std::fprintf;
    using ::std::_fprintf;
    using ::std::printf;
    using ::std::_printf;
    using ::std::sprintf;
    using ::std::_sprintf;
    using ::std::snprintf;
    using ::std::vsnprintf;
    using ::std::vfscanf;
    using ::std::vscanf;
    using ::std::vsscanf;
    using ::std::_snprintf;
    using ::std::_vsnprintf;




    using ::std::__ARM_asprintf;
    using ::std::__ARM_vasprintf;
    using ::std::__ARM_vsnprintf;
    using ::std::__ARM_snprintf;
    using ::std::__ARM_vsscanf;
    using ::std::fscanf;
    using ::std::_fscanf;
    using ::std::scanf;
    using ::std::_scanf;
    using ::std::sscanf;
    using ::std::_sscanf;
    using ::std::_vfscanf;
    using ::std::_vscanf;
    using ::std::_vsscanf;
    using ::std::vprintf;
    using ::std::_vprintf;
    using ::std::vfprintf;
    using ::std::_vfprintf;
    using ::std::vsprintf;
    using ::std::_vsprintf;
    using ::std::fgetc;
    using ::std::fgets;
    using ::std::fputc;
    using ::std::fputs;
    using ::std::getc;
    using ::std::getchar;
    using ::std::gets;
    using ::std::putc;
    using ::std::putchar;
    using ::std::puts;
    using ::std::ungetc;
    using ::std::fread;
    using ::std::__fread_bytes_avail;
    using ::std::fwrite;
    using ::std::fgetpos;
    using ::std::fseek;
    using ::std::fsetpos;
    using ::std::ftell;
    using ::std::rewind;
    using ::std::clearerr;
    using ::std::feof;
    using ::std::ferror;
    using ::std::perror;
    using ::std::_fisatty;
    using ::std::__use_no_semihosting_swi;
    using ::std::__use_no_semihosting;
# 109 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\stdio.h" 2 3
# 2 "../ByteTrack/lapjv.cpp" 2
# 1 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\stdlib.h" 1 3
# 92 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\stdlib.h" 3



# 1 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 1 3
# 52 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
      namespace std {

          extern "C" {
# 64 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
    typedef unsigned int size_t;
# 96 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
typedef struct div_t { int quot, rem; } div_t;

typedef struct ldiv_t { long int quot, rem; } ldiv_t;


typedef struct lldiv_t { long long quot, rem; } lldiv_t;
# 139 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) int __aeabi_MB_CUR_MAX(void);
# 158 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) double atof(const char * ) __attribute__((__nonnull__(1)));





extern __attribute__((__nothrow__)) int atoi(const char * ) __attribute__((__nonnull__(1)));





extern __attribute__((__nothrow__)) long int atol(const char * ) __attribute__((__nonnull__(1)));






extern __attribute__((__nothrow__)) long long atoll(const char * ) __attribute__((__nonnull__(1)));







extern __attribute__((__nothrow__)) double strtod(const char * __restrict , char ** __restrict ) __attribute__((__nonnull__(1)));
# 206 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) float strtof(const char * __restrict , char ** __restrict ) __attribute__((__nonnull__(1)));
extern __attribute__((__nothrow__)) long double strtold(const char * __restrict , char ** __restrict ) __attribute__((__nonnull__(1)));




extern __attribute__((__nothrow__)) long int strtol(const char * __restrict ,
                        char ** __restrict , int ) __attribute__((__nonnull__(1)));
# 243 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) unsigned long int strtoul(const char * __restrict ,
                                       char ** __restrict , int ) __attribute__((__nonnull__(1)));
# 275 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) long long strtoll(const char * __restrict ,
                                  char ** __restrict , int )
                          __attribute__((__nonnull__(1)));






extern __attribute__((__nothrow__)) unsigned long long strtoull(const char * __restrict ,
                                            char ** __restrict , int )
                                   __attribute__((__nonnull__(1)));






extern __attribute__((__nothrow__)) int rand(void);
# 303 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) void srand(unsigned int );
# 313 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
struct _rand_state { int __x[57]; };
extern __attribute__((__nothrow__)) int _rand_r(struct _rand_state *);
extern __attribute__((__nothrow__)) void _srand_r(struct _rand_state *, unsigned int);
struct _ANSI_rand_state { int __x[1]; };
extern __attribute__((__nothrow__)) int _ANSI_rand_r(struct _ANSI_rand_state *);
extern __attribute__((__nothrow__)) void _ANSI_srand_r(struct _ANSI_rand_state *, unsigned int);





extern __attribute__((__nothrow__)) void *calloc(size_t , size_t );





extern __attribute__((__nothrow__)) void free(void * );







extern __attribute__((__nothrow__)) void *malloc(size_t );





extern __attribute__((__nothrow__)) void *realloc(void * , size_t );
# 374 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
typedef int (*__heapprt)(void *, char const *, ...);
extern __attribute__((__nothrow__)) void __heapstats(int (* )(void * ,
                                           char const * , ...),
                        void * ) __attribute__((__nonnull__(1)));
# 390 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) int __heapvalid(int (* )(void * ,
                                           char const * , ...),
                       void * , int ) __attribute__((__nonnull__(1)));
# 411 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) __attribute__((__noreturn__)) void abort(void);
# 422 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) int atexit(void (* )(void)) __attribute__((__nonnull__(1)));
# 444 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) __attribute__((__noreturn__)) void exit(int );
# 460 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) __attribute__((__noreturn__)) void _Exit(int );
# 471 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) char *getenv(const char * ) __attribute__((__nonnull__(1)));
# 484 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) int system(const char * );
# 497 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern void *bsearch(const void * , const void * ,
              size_t , size_t ,
              int (* )(const void *, const void *)) __attribute__((__nonnull__(1,2,5)));
# 532 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern void qsort(void * , size_t , size_t ,
           int (* )(const void *, const void *)) __attribute__((__nonnull__(1,4)));
# 560 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) __attribute__((__const__)) int abs(int );






extern __attribute__((__nothrow__)) __attribute__((__const__)) div_t div(int , int );
# 579 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) __attribute__((__const__)) long int labs(long int );






   extern "C++" inline __attribute__((__nothrow__)) __attribute__((__const__)) long abs(long int x) { return labs(x); }


extern __attribute__((__nothrow__)) __attribute__((__const__)) ldiv_t ldiv(long int , long int );
# 604 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
   extern "C++" inline __attribute__((__nothrow__)) __attribute__((__const__)) ldiv_t div(long int __numer, long int __denom) {
       return ldiv(__numer, __denom);
   }



extern __attribute__((__nothrow__)) __attribute__((__const__)) long long llabs(long long );






   extern "C++" inline __attribute__((__nothrow__)) __attribute__((__const__)) long long abs(long long x) { return llabs(x); }


extern __attribute__((__nothrow__)) __attribute__((__const__)) lldiv_t lldiv(long long , long long );
# 635 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
   extern "C++" inline __attribute__((__nothrow__)) __attribute__((__const__)) lldiv_t div(long long __numer, long long __denom) {
       return lldiv(__numer, __denom);
   }






typedef struct __sdiv32by16 { long quot, rem; } __sdiv32by16;
typedef struct __udiv32by16 { unsigned long quot, rem; } __udiv32by16;

typedef struct __sdiv64by32 { long rem, quot; } __sdiv64by32;

__attribute__((__value_in_regs__)) extern __attribute__((__nothrow__)) __attribute__((__const__)) __sdiv32by16 __rt_sdiv32by16(
     int ,
     short int );



__attribute__((__value_in_regs__)) extern __attribute__((__nothrow__)) __attribute__((__const__)) __udiv32by16 __rt_udiv32by16(
     unsigned int ,
     unsigned short );



__attribute__((__value_in_regs__)) extern __attribute__((__nothrow__)) __attribute__((__const__)) __sdiv64by32 __rt_sdiv64by32(
     int , unsigned int ,
     int );







extern __attribute__((__nothrow__)) unsigned int __fp_status(unsigned int , unsigned int );
# 705 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) int mblen(const char * , size_t );
# 720 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) int mbtowc(wchar_t * __restrict ,
                   const char * __restrict , size_t );
# 739 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) int wctomb(char * , wchar_t );
# 761 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) size_t mbstowcs(wchar_t * __restrict ,
                      const char * __restrict , size_t ) __attribute__((__nonnull__(2)));
# 779 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) size_t wcstombs(char * __restrict ,
                      const wchar_t * __restrict , size_t ) __attribute__((__nonnull__(2)));
# 798 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) void __use_realtime_heap(void);
extern __attribute__((__nothrow__)) void __use_realtime_division(void);
extern __attribute__((__nothrow__)) void __use_two_region_memory(void);
extern __attribute__((__nothrow__)) void __use_no_heap(void);
extern __attribute__((__nothrow__)) void __use_no_heap_region(void);

extern __attribute__((__nothrow__)) char const *__C_library_version_string(void);
extern __attribute__((__nothrow__)) int __C_library_version_number(void);


         }
      }
# 820 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
        using ::std::atoll;
        using ::std::lldiv_t;

      using ::std::div_t;
      using ::std::ldiv_t;
      using ::std::atof;
      using ::std::atoi;
      using ::std::atol;
      using ::std::strtod;

      using ::std::strtof;
      using ::std::strtold;

      using ::std::strtol;
      using ::std::strtoul;
      using ::std::strtoll;
      using ::std::strtoull;
      using ::std::rand;
      using ::std::srand;
      using ::std::_rand_state;
      using ::std::_rand_r;
      using ::std::_srand_r;
      using ::std::_ANSI_rand_state;
      using ::std::_ANSI_rand_r;
      using ::std::_ANSI_srand_r;
      using ::std::calloc;
      using ::std::free;
      using ::std::malloc;
      using ::std::realloc;



      using ::std::__heapprt;
      using ::std::__heapstats;
      using ::std::__heapvalid;
      using ::std::abort;
      using ::std::atexit;
      using ::std::exit;
      using ::std::_Exit;
      using ::std::getenv;
      using ::std::system;
      using ::std::bsearch;
      using ::std::qsort;
      using ::std::abs;
      using ::std::div;
      using ::std::labs;
      using ::std::ldiv;

        using ::std::llabs;
        using ::std::lldiv;

      using ::std::__sdiv32by16;
      using ::std::__udiv32by16;
      using ::std::__sdiv64by32;
      using ::std::__rt_sdiv32by16;
      using ::std::__rt_udiv32by16;
      using ::std::__rt_sdiv64by32;
      using ::std::__fp_status;
      using ::std::mblen;
      using ::std::mbtowc;
      using ::std::wctomb;
      using ::std::mbstowcs;
      using ::std::wcstombs;
      using ::std::__use_realtime_heap;
      using ::std::__use_realtime_division;
      using ::std::__use_two_region_memory;
      using ::std::__use_no_heap;
      using ::std::__use_no_heap_region;
      using ::std::__C_library_version_string;
      using ::std::__C_library_version_number;
      using ::std::size_t;
      using ::std::__aeabi_MB_CUR_MAX;
# 96 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\stdlib.h" 2 3



extern "C++" {
# 118 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\stdlib.h" 3
[[__nodiscard__]] inline __attribute__((__visibility__("hidden"))) __attribute__((__exclude_from_explicit_instantiation__)) __attribute__((__abi_tag__("nn200000"))) float abs(float __lcpp_x) noexcept {
  return __builtin_fabsf(__lcpp_x);
}

[[__nodiscard__]] inline __attribute__((__visibility__("hidden"))) __attribute__((__exclude_from_explicit_instantiation__)) __attribute__((__abi_tag__("nn200000"))) double abs(double __lcpp_x) noexcept {
  return __builtin_fabs(__lcpp_x);
}

[[__nodiscard__]] inline __attribute__((__visibility__("hidden"))) __attribute__((__exclude_from_explicit_instantiation__)) __attribute__((__abi_tag__("nn200000"))) long double abs(long double __lcpp_x) noexcept {
  return __builtin_fabsl(__lcpp_x);
}
# 149 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\stdlib.h" 3
}
# 3 "../ByteTrack/lapjv.cpp" 2
# 1 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\string.h" 1 3
# 58 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\string.h" 3



# 1 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 1 3
# 32 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
        namespace std {

        extern "C" {
# 44 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
    typedef unsigned int size_t;
# 58 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) void *memcpy(void * __restrict ,
                    const void * __restrict , size_t ) __attribute__((__nonnull__(1,2)));






extern __attribute__((__nothrow__)) void *memmove(void * ,
                    const void * , size_t ) __attribute__((__nonnull__(1,2)));
# 77 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) char *strcpy(char * __restrict , const char * __restrict ) __attribute__((__nonnull__(1,2)));






extern __attribute__((__nothrow__)) char *strncpy(char * __restrict , const char * __restrict , size_t ) __attribute__((__nonnull__(1,2)));
# 93 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) char *strcat(char * __restrict , const char * __restrict ) __attribute__((__nonnull__(1,2)));






extern __attribute__((__nothrow__)) char *strncat(char * __restrict , const char * __restrict , size_t ) __attribute__((__nonnull__(1,2)));
# 117 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) int memcmp(const void * , const void * , size_t ) __attribute__((__nonnull__(1,2)));







extern __attribute__((__nothrow__)) int strcmp(const char * , const char * ) __attribute__((__nonnull__(1,2)));






extern __attribute__((__nothrow__)) int strncmp(const char * , const char * , size_t ) __attribute__((__nonnull__(1,2)));
# 141 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) int strcasecmp(const char * , const char * ) __attribute__((__nonnull__(1,2)));







extern __attribute__((__nothrow__)) int strncasecmp(const char * , const char * , size_t ) __attribute__((__nonnull__(1,2)));
# 158 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) int strcoll(const char * , const char * ) __attribute__((__nonnull__(1,2)));
# 169 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) size_t strxfrm(char * __restrict , const char * __restrict , size_t ) __attribute__((__nonnull__(2)));
# 188 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) const void *memchr(const void * , int , size_t ) __attribute__((__nonnull__(1)));
extern "C++" void *memchr(void * __s, int __c, size_t __n) __attribute__((__nonnull__(1)));
extern "C++" inline void *memchr(void * __s, int __c, size_t __n)
    { return const_cast<void *>(memchr(const_cast<const void *>(__s), __c, __n)); }
# 204 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) const char *strchr(const char * , int ) __attribute__((__nonnull__(1)));
extern "C++" char *strchr(char * __s, int __c) __attribute__((__nonnull__(1)));
extern "C++" inline char *strchr(char * __s, int __c)
    { return const_cast<char *>(strchr(const_cast<const char *>(__s), __c)); }
# 218 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) size_t strcspn(const char * , const char * ) __attribute__((__nonnull__(1,2)));
# 227 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) const char *strpbrk(const char * , const char * ) __attribute__((__nonnull__(1,2)));
extern "C++" char *strpbrk(char * __s1, const char * __s2) __attribute__((__nonnull__(1,2)));
extern "C++" inline char *strpbrk(char * __s1, const char * __s2)
    { return const_cast<char *>(strpbrk(const_cast<const char *>(__s1), __s2)); }
# 242 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) const char *strrchr(const char * , int ) __attribute__((__nonnull__(1)));
extern "C++" char *strrchr(char * __s, int __c) __attribute__((__nonnull__(1)));
extern "C++" inline char *strrchr(char * __s, int __c)
    { return const_cast<char *>(strrchr(const_cast<const char *>(__s), __c)); }
# 257 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) size_t strspn(const char * , const char * ) __attribute__((__nonnull__(1,2)));







extern __attribute__((__nothrow__)) const char *strstr(const char * , const char * ) __attribute__((__nonnull__(1,2)));
extern "C++" char *strstr(char * __s1, const char * __s2) __attribute__((__nonnull__(1,2)));
extern "C++" inline char *strstr(char * __s1, const char * __s2)
    { return const_cast<char *>(strstr(const_cast<const char *>(__s1), __s2)); }
# 280 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) char *strtok(char * __restrict , const char * __restrict ) __attribute__((__nonnull__(2)));
extern __attribute__((__nothrow__)) char *_strtok_r(char * , const char * , char ** ) __attribute__((__nonnull__(2,3)));
# 321 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) void *memset(void * , int , size_t ) __attribute__((__nonnull__(1)));





extern __attribute__((__nothrow__)) char *strerror(int );







extern __attribute__((__nothrow__)) size_t strlen(const char * ) __attribute__((__nonnull__(1)));





extern __attribute__((__nothrow__)) size_t strnlen(const char * , size_t ) __attribute__((__nonnull__(1)));







extern __attribute__((__nothrow__)) size_t strlcpy(char * , const char * , size_t ) __attribute__((__nonnull__(1,2)));
# 369 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) size_t strlcat(char * , const char * , size_t ) __attribute__((__nonnull__(1,2)));
# 395 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) void _membitcpybl(void * , const void * , int , int , size_t ) __attribute__((__nonnull__(1,2)));
extern __attribute__((__nothrow__)) void _membitcpybb(void * , const void * , int , int , size_t ) __attribute__((__nonnull__(1,2)));
extern __attribute__((__nothrow__)) void _membitcpyhl(void * , const void * , int , int , size_t ) __attribute__((__nonnull__(1,2)));
extern __attribute__((__nothrow__)) void _membitcpyhb(void * , const void * , int , int , size_t ) __attribute__((__nonnull__(1,2)));
extern __attribute__((__nothrow__)) void _membitcpywl(void * , const void * , int , int , size_t ) __attribute__((__nonnull__(1,2)));
extern __attribute__((__nothrow__)) void _membitcpywb(void * , const void * , int , int , size_t ) __attribute__((__nonnull__(1,2)));
extern __attribute__((__nothrow__)) void _membitmovebl(void * , const void * , int , int , size_t ) __attribute__((__nonnull__(1,2)));
extern __attribute__((__nothrow__)) void _membitmovebb(void * , const void * , int , int , size_t ) __attribute__((__nonnull__(1,2)));
extern __attribute__((__nothrow__)) void _membitmovehl(void * , const void * , int , int , size_t ) __attribute__((__nonnull__(1,2)));
extern __attribute__((__nothrow__)) void _membitmovehb(void * , const void * , int , int , size_t ) __attribute__((__nonnull__(1,2)));
extern __attribute__((__nothrow__)) void _membitmovewl(void * , const void * , int , int , size_t ) __attribute__((__nonnull__(1,2)));
extern __attribute__((__nothrow__)) void _membitmovewb(void * , const void * , int , int , size_t ) __attribute__((__nonnull__(1,2)));
# 457 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
         }
      }





      using ::std::size_t;
      using ::std::memcpy;
      using ::std::memmove;
      using ::std::strcpy;
      using ::std::strncpy;
      using ::std::strcat;
      using ::std::strncat;
      using ::std::memcmp;
      using ::std::strcmp;
      using ::std::strncmp;
      using ::std::strcasecmp;
      using ::std::strncasecmp;
      using ::std::strcoll;
      using ::std::strxfrm;
      using ::std::memchr;
      using ::std::strchr;
      using ::std::strcspn;
      using ::std::strpbrk;
      using ::std::strrchr;
      using ::std::strspn;
      using ::std::strstr;
      using ::std::strtok;



      using ::std::_strtok_r;
      using ::std::memset;
      using ::std::strerror;
      using ::std::strlen;
      using ::std::strnlen;
      using ::std::strlcpy;
      using ::std::strlcat;
      using ::std::_membitcpybl;
      using ::std::_membitcpybb;
      using ::std::_membitcpyhl;
      using ::std::_membitcpyhb;
      using ::std::_membitcpywl;
      using ::std::_membitcpywb;
      using ::std::_membitmovebl;
      using ::std::_membitmovebb;
      using ::std::_membitmovehl;
      using ::std::_membitmovehb;
      using ::std::_membitmovewl;
      using ::std::_membitmovewb;
# 62 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\string.h" 2 3
# 72 "C:\\Users\\antia\\AppData\\Local\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\libcxx\\string.h" 3
extern "C++" {
inline __attribute__((__visibility__("hidden"))) __attribute__((__exclude_from_explicit_instantiation__)) __attribute__((__abi_tag__("nn200000"))) __attribute__((__enable_if__(true, ""))) const char* strchr(const char* __s, int __c) {
  return __builtin_strchr(__s, __c);
}
inline __attribute__((__visibility__("hidden"))) __attribute__((__exclude_from_explicit_instantiation__)) __attribute__((__abi_tag__("nn200000"))) __attribute__((__enable_if__(true, ""))) char* strchr(char* __s, int __c) {
  return __builtin_strchr(__s, __c);
}

inline __attribute__((__visibility__("hidden"))) __attribute__((__exclude_from_explicit_instantiation__)) __attribute__((__abi_tag__("nn200000"))) __attribute__((__enable_if__(true, ""))) const char* strpbrk(const char* __s1, const char* __s2) {
  return __builtin_strpbrk(__s1, __s2);
}
inline __attribute__((__visibility__("hidden"))) __attribute__((__exclude_from_explicit_instantiation__)) __attribute__((__abi_tag__("nn200000"))) __attribute__((__enable_if__(true, ""))) char* strpbrk(char* __s1, const char* __s2) {
  return __builtin_strpbrk(__s1, __s2);
}

inline __attribute__((__visibility__("hidden"))) __attribute__((__exclude_from_explicit_instantiation__)) __attribute__((__abi_tag__("nn200000"))) __attribute__((__enable_if__(true, ""))) const char* strrchr(const char* __s, int __c) {
  return __builtin_strrchr(__s, __c);
}
inline __attribute__((__visibility__("hidden"))) __attribute__((__exclude_from_explicit_instantiation__)) __attribute__((__abi_tag__("nn200000"))) __attribute__((__enable_if__(true, ""))) char* strrchr(char* __s, int __c) {
  return __builtin_strrchr(__s, __c);
}

inline __attribute__((__visibility__("hidden"))) __attribute__((__exclude_from_explicit_instantiation__)) __attribute__((__abi_tag__("nn200000"))) __attribute__((__enable_if__(true, ""))) const void* memchr(const void* __s, int __c, size_t __n) {
  return __builtin_memchr(__s, __c, __n);
}
inline __attribute__((__visibility__("hidden"))) __attribute__((__exclude_from_explicit_instantiation__)) __attribute__((__abi_tag__("nn200000"))) __attribute__((__enable_if__(true, ""))) void* memchr(void* __s, int __c, size_t __n) {
  return __builtin_memchr(__s, __c, __n);
}

inline __attribute__((__visibility__("hidden"))) __attribute__((__exclude_from_explicit_instantiation__)) __attribute__((__abi_tag__("nn200000"))) __attribute__((__enable_if__(true, ""))) const char* strstr(const char* __s1, const char* __s2) {
  return __builtin_strstr(__s1, __s2);
}
inline __attribute__((__visibility__("hidden"))) __attribute__((__exclude_from_explicit_instantiation__)) __attribute__((__abi_tag__("nn200000"))) __attribute__((__enable_if__(true, ""))) char* strstr(char* __s1, const char* __s2) {
  return __builtin_strstr(__s1, __s2);
}
}
# 4 "../ByteTrack/lapjv.cpp" 2

# 1 "../ByteTrack/include\\lapjv.h" 1
# 53 "../ByteTrack/include\\lapjv.h"
typedef signed int int_t;
typedef unsigned int uint_t;
typedef double cost_t;
typedef char boolean;
typedef enum fp_t { FP_1 = 1, FP_2 = 2, FP_DYNAMIC = 3 } fp_t;

extern int_t lapjv_internal(
 const uint_t n, cost_t *cost[],
 int_t *x, int_t *y);
# 6 "../ByteTrack/lapjv.cpp" 2



int_t _ccrrt_dense(const uint_t n, cost_t *cost[],
 int_t *free_rows, int_t *x, int_t *y, cost_t *v)
{
 int_t n_free_rows;
 boolean *unique;

 for (uint_t i = 0; i < n; i++) {
  x[i] = -1;
  v[i] = 1000000;
  y[i] = 0;
 }
 for (uint_t i = 0; i < n; i++) {
  for (uint_t j = 0; j < n; j++) {
   const cost_t c = cost[i][j];
   if (c < v[j]) {
    v[j] = c;
    y[j] = i;
   }
                                                                          ;
  }
 }
                       ;
                        ;
 if ((unique = (boolean *)malloc(sizeof(boolean) * (n))) == 0) { return -1; };
 memset(unique, 1, n);
 {
  int_t j = n;
  do {
   j--;
   const int_t i = y[j];
   if (x[i] < 0) {
    x[i] = j;
   }
   else {
    unique[i] = 0;
    y[j] = -1;
   }
  } while (j > 0);
 }
 n_free_rows = 0;
 for (uint_t i = 0; i < n; i++) {
  if (x[i] < 0) {
   free_rows[n_free_rows++] = i;
  }
  else if (unique[i]) {
   const int_t j = x[i];
   cost_t min = 1000000;
   for (uint_t j2 = 0; j2 < n; j2++) {
    if (j2 == (uint_t)j) {
     continue;
    }
    const cost_t c = cost[i][j2] - v[j2];
    if (c < min) {
     min = c;
    }
   }
                                            ;
   v[j] -= min;
  }
 }
 if (unique != 0) { free(unique); unique = 0; };
 return n_free_rows;
}




int_t _carr_dense(
 const uint_t n, cost_t *cost[],
 const uint_t n_free_rows,
 int_t *free_rows, int_t *x, int_t *y, cost_t *v)
{
 uint_t current = 0;
 int_t new_free_rows = 0;
 uint_t rr_cnt = 0;
                        ;
                        ;
                       ;
                                          ;
 while (current < n_free_rows) {
  int_t i0;
  int_t j1, j2;
  cost_t v1, v2, v1_new;
  boolean v1_lowers;

  rr_cnt++;
                                                       ;
  const int_t free_i = free_rows[current++];
  j1 = 0;
  v1 = cost[free_i][0] - v[0];
  j2 = -1;
  v2 = 1000000;
  for (uint_t j = 1; j < n; j++) {
                                              ;
   const cost_t c = cost[free_i][j] - v[j];
   if (c < v2) {
    if (c >= v1) {
     v2 = c;
     j2 = j;
    }
    else {
     v2 = v1;
     v1 = c;
     j2 = j1;
     j1 = j;
    }
   }
  }
  i0 = y[j1];
  v1_new = v[j1] - (v2 - v1);
  v1_lowers = v1_new < v[j1];
                                                                                                                 ;
  if (rr_cnt < current * n) {
   if (v1_lowers) {
    v[j1] = v1_new;
   }
   else if (i0 >= 0 && j2 >= 0) {
    j1 = j2;
    i0 = y[j2];
   }
   if (i0 >= 0) {
    if (v1_lowers) {
     free_rows[--current] = i0;
    }
    else {
     free_rows[new_free_rows++] = i0;
    }
   }
  }
  else {
                                                                                   ;
   if (i0 >= 0) {
    free_rows[new_free_rows++] = i0;
   }
  }
  x[free_i] = j1;
  y[j1] = free_i;
 }
 return new_free_rows;
}




uint_t _find_dense(const uint_t n, uint_t lo, cost_t *d, int_t *cols, int_t *y)
{
 uint_t hi = lo + 1;
 cost_t mind = d[cols[lo]];
 for (uint_t k = hi; k < n; k++) {
  int_t j = cols[k];
  if (d[j] <= mind) {
   if (d[j] < mind) {
    hi = lo;
    mind = d[j];
   }
   cols[k] = cols[hi];
   cols[hi++] = j;
  }
 }
 return hi;
}




int_t _scan_dense(const uint_t n, cost_t *cost[],
 uint_t *plo, uint_t*phi,
 cost_t *d, int_t *cols, int_t *pred,
 int_t *y, cost_t *v)
{
 uint_t lo = *plo;
 uint_t hi = *phi;
 cost_t h, cred_ij;

 while (lo != hi) {
  int_t j = cols[lo++];
  const int_t i = y[j];
  const cost_t mind = d[j];
  h = cost[i][j] - v[j] - mind;
                                     ;

  for (uint_t k = hi; k < n; k++) {
   j = cols[k];
   cred_ij = cost[i][j] - v[j] - h;
   if (cred_ij < d[j]) {
    d[j] = cred_ij;
    pred[j] = i;
    if (cred_ij == mind) {
     if (y[j] < 0) {
      return j;
     }
     cols[k] = cols[hi];
     cols[hi++] = j;
    }
   }
  }
 }
 *plo = lo;
 *phi = hi;
 return -1;
}
# 218 "../ByteTrack/lapjv.cpp"
int_t find_path_dense(
 const uint_t n, cost_t *cost[],
 const int_t start_i,
 int_t *y, cost_t *v,
 int_t *pred)
{
 uint_t lo = 0, hi = 0;
 int_t final_j = -1;
 uint_t n_ready = 0;
 int_t *cols;
 cost_t *d;

 if ((cols = (int_t *)malloc(sizeof(int_t) * (n))) == 0) { return -1; };
 if ((d = (cost_t *)malloc(sizeof(cost_t) * (n))) == 0) { return -1; };

 for (uint_t i = 0; i < n; i++) {
  cols[i] = i;
  pred[i] = start_i;
  d[i] = cost[start_i][i] - v[i];
 }
                       ;
 while (final_j == -1) {

  if (lo == hi) {
                                     ;
   n_ready = lo;
   hi = _find_dense(n, lo, d, cols, y);
                                   ;
                             ;
   for (uint_t k = lo; k < hi; k++) {
    const int_t j = cols[k];
    if (y[j] < 0) {
     final_j = j;
    }
   }
  }
  if (final_j == -1) {
                                     ;
   final_j = _scan_dense(
    n, cost, &lo, &hi, d, cols, pred, y, v);
                         ;
                             ;
                             ;
  }
 }

                                      ;
                           ;
 {
  const cost_t mind = d[cols[lo]];
  for (uint_t k = 0; k < n_ready; k++) {
   const int_t j = cols[k];
   v[j] += d[j] - mind;
  }
 }

 if (cols != 0) { free(cols); cols = 0; };
 if (d != 0) { free(d); d = 0; };

 return final_j;
}




int_t _ca_dense(
 const uint_t n, cost_t *cost[],
 const uint_t n_free_rows,
 int_t *free_rows, int_t *x, int_t *y, cost_t *v)
{
 int_t *pred;

 if ((pred = (int_t *)malloc(sizeof(int_t) * (n))) == 0) { return -1; };

 for (int_t *pfree_i = free_rows; pfree_i < free_rows + n_free_rows; pfree_i++) {
  int_t i = -1, j;
  uint_t k = 0;

                                            ;
  j = find_path_dense(n, cost, *pfree_i, y, v, pred);
                ;
               ;
  while (i != *pfree_i) {
                            ;
                             ;
   i = pred[j];
                                         ;
   y[j] = i;
                          ;
   { int_t _temp_index = j; j = x[i]; x[i] = _temp_index; };
   k++;
   if (k >= n) {
                 ;
   }
  }
 }
 if (pred != 0) { free(pred); pred = 0; };
 return 0;
}




int lapjv_internal(
 const uint_t n, cost_t *cost[],
 int_t *x, int_t *y)
{
 int ret;
 int_t *free_rows;
 cost_t *v;

 if ((free_rows = (int_t *)malloc(sizeof(int_t) * (n))) == 0) { return -1; };
 if ((v = (cost_t *)malloc(sizeof(cost_t) * (n))) == 0) { return -1; };
 ret = _ccrrt_dense(n, cost, free_rows, x, y, v);
 int i = 0;
 while (ret > 0 && i < 2) {
  ret = _carr_dense(n, cost, ret, free_rows, x, y, v);
  i++;
 }
 if (ret > 0) {
  ret = _ca_dense(n, cost, ret, free_rows, x, y, v);
 }
 if (v != 0) { free(v); v = 0; };
 if (free_rows != 0) { free(free_rows); free_rows = 0; };
 return ret;
}
