# 整数环取模模板
适合各种算法竞赛的全局取模场景使用。

代码：

```cpp
#include<limits>
#include<cstdint>
#include<iostream>

/* A Modulo-safe Template Designed for Competitive Programming. */
/* Notes:
 *  0. Use C++ 20 or newer.
 *  1. Modulus `mod` must be less than Sqrt(MAX_OF_NUMTYPE), or it will overflow in multiplication.
 *  2. Modulus must be a prime.
 *  3. Use -O2 or higher optimization level.
 */
constexpr bool is_prime(uint64_t n)
{
    if (n <= 1) return false;
    for (uint64_t i = 2; i <= (n / i); ++i)
        if (n % i == 0) return false;
    return true;
}
template<typename NumType, uint64_t MOD>
class Ring
{
    static_assert(std::is_unsigned_v<NumType>, "NumType must be unsigned");
    static_assert(MOD > 1, "MOD must be greater than 1");
    static_assert(is_prime(MOD), "MOD must be a prime");
    static_assert(MOD <= std::numeric_limits<NumType>::max() / MOD, "MOD is too large (MOD^2 exceeds NumType max)");
    NumType num;
    private:
        Ring inv() const { return Ring(pow_mod(num, MOD - 2)); }
        static NumType pow_mod(NumType a, NumType b) {
            NumType res = 1;
            a %= MOD;
            while (b) {
                if (b & 1) res = static_cast<NumType>((1ULL * res * a) % MOD);
                a = static_cast<NumType>((1ULL * a * a) % MOD);
                b >>= 1;
            }
            return res;
        }
    public:
        Ring() { num = 0; }
        Ring(NumType x) : num(x % MOD) { }
        Ring operator+(const Ring<NumType, MOD>& b) const
        { 
            //Assert num and b.num are in [0, MOD)
            NumType res = num + b.num;
            if(res >= MOD) return Ring(res - MOD);
            else return Ring(res);
        }
        Ring operator-(const Ring& b) const  
        {
            //Assert num and b.num are in [0, MOD)
            if(num < b.num) return Ring(num + MOD - b.num);
            else return Ring(num - b.num);
        }
        Ring operator*(const Ring& b) const
        {
            if constexpr (MOD >= std::numeric_limits<uint32_t>::max())
                return Ring(static_cast<NumType>((static_cast<__uint128_t>(num) * b.num) % MOD));
            else
            {
                uint64_t product = 1ULL * num * b.num;
                static constexpr uint64_t mu = (static_cast<uint64_t>(1) << 63) / MOD;
                uint64_t q = (product * mu) >> 63;
                NumType res = static_cast<NumType>(product - q * MOD);
                if (res >= MOD) res -= MOD;
                return Ring(res);
            }
        }
        Ring operator/(const Ring& b) const 
        {
            if constexpr (is_prime(MOD)) return *this * b.inv();
            else static_assert(!(sizeof(NumType)), "Require a prime modulus.");
        }
        Ring operator%(const Ring& b) const { return Ring(num % b.num); } // b.num < MOD is ensured. This operator only cuts down size.
        Ring operator^(const Ring& exp) const { return Ring(pow_mod(num, exp.num)); }
        void operator+=(const Ring& b)
        {
            NumType res = num + b.num;
            if(res >= MOD) num = res - MOD;
            else num = res;
        }
        void operator-=(const Ring& b)
        {
            if(num < b.num) num = num + MOD - b.num;
            else num -= b.num;
        }
        auto operator<=>(const Ring& b) const { return num <=> b.num; }
        template<typename T, uint64_t M>
        friend std::istream& operator>>(std::istream& in, Ring<T, M>& a);
        template<typename T, uint64_t M>
        friend std::ostream& operator<<(std::ostream& out, const Ring<T, M>& a);
};
template<typename NumType, uint64_t MOD>
std::istream& operator>>(std::istream& in, Ring<NumType, MOD>& a)
{
    NumType tmp;
    in >> tmp;
    a.num = tmp % MOD;
    return in;
}
template<typename NumType, uint64_t MOD>
std::ostream& operator<<(std::ostream& out, const Ring<NumType, MOD>& a)
{
    out << a.num;
    return out;
}
using u64 = unsigned long long;
using Z = Ring<u64, 998244353>;

int main()
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    Z x, y;
    std::cin >> x >> y;
    std::cout << "x + y : " << x+y << '\n';
    std::cout << "x - y : " << x-y << '\n';
    std::cout << "x * y : " << x*y << '\n';
    std::cout << "x * inv y : " << x/y << '\n';
    std::cout << "x ^ y : " << (x^y) << '\n';
    return 0;
}
```