# 整数环取模模板
适合各种算法竞赛的全局取模场景使用。

代码：

```cpp
#include<cstdint>
#include<iostream>

/* A Modulo-safe Template Designed for Competitive Programming. */
/* Notes:
 *  0. Use C++ 17 or newer.
 *  1. Modulus `mod` must be less than Sqrt(MAX_OF_NUMTYPE), or it will overflow in multiplication.
 *  2. All inputs and outputs will NOT be moduloed for higher I/O efficiency.
 *  3. Modulus must be a prime.
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
        Ring(NumType x) : num(x) { }
        Ring operator+(const Ring& b) const { return Ring(static_cast<NumType>((num + b.num) % MOD)); }
        Ring operator-(const Ring& b) const  { return Ring(static_cast<NumType>((num + MOD - b.num) % MOD)); }
        Ring operator*(const Ring& b) const  { return Ring(static_cast<NumType>((1ULL * b.num * num) % MOD)); }
        Ring operator/(const Ring& b) const 
        {
            if constexpr (is_prime(MOD)) return *this * b.inv();
            else static_assert(false, "Require a prime modulus.");
        }
        Ring operator%(const Ring& b) const { return Ring((num % b.num) % MOD); } // cut down size.
        Ring operator^(const Ring& exp) const { return Ring(pow_mod(num, exp.num)); }
        template<typename T, uint64_t M>
        friend std::istream& operator>>(std::istream& in, Ring<T, M>& a);
        template<typename T, uint64_t M>
        friend std::ostream& operator<<(std::ostream& out, const Ring<T, M>& a);
};
template<typename NumType, uint64_t MOD>
std::istream& operator>>(std::istream& in, Ring<NumType, MOD>& a)
{
    in >> a.num;
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
    Z a, b;
    std::cin>>a>>b;
    std::cout<<a+b;
    return 0;
}
```