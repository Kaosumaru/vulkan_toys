#ifndef MTL_FUNCTION_TRAITS
#define MTL_FUNCTION_TRAITS

#include <tuple>

namespace mtl
{
    // functor
    template<class F>
    struct function_traits
    {
    private:
        using call_type = function_traits<decltype(&F::operator())>;
    public:
        using return_type = typename call_type::return_type;
        using decayed_signature = typename call_type::decayed_signature;

        static constexpr std::size_t arity = call_type::arity - 1;

        template <std::size_t N>
        struct argument
        {
            static_assert(N < arity, "error: invalid parameter index.");
            using type = typename call_type::template argument<N + 1>::type;
        };
    };

    // function pointer
    template<class R, class... Args>
    struct function_traits<R(*)(Args...)> : public function_traits<R(Args...)>
    {};

    template<class R, class... Args>
    struct function_traits<R(Args...)>
    {
        using return_type = R;
        using decayed_signature = R(Args...);

        static constexpr std::size_t arity = sizeof...(Args);

        template <std::size_t N>
        struct argument
        {
            static_assert(N < arity, "error: invalid parameter index.");
            using type = typename std::tuple_element<N, std::tuple<Args...>>::type;
        };
    };

    // member function pointer
    template<class C, class R, class... Args>
    struct function_traits<R(C::*)(Args...)> : public function_traits<R(C&, Args...)>
    {
        using decayed_signature = R(Args...);
    };

    // const member function pointer
    template<class C, class R, class... Args>
    struct function_traits<R(C::*)(Args...) const> : public function_traits<R(C&, Args...)>
    {
        using decayed_signature = R(Args...);
    };

    // member object pointer
    template<class C, class R>
    struct function_traits<R(C::*)> : public function_traits<R(C&)>
    {};



    template<class F>
    struct function_traits<F&> : public function_traits<F>
    {};

    template<class F>
    struct function_traits<F&&> : public function_traits<F>
    {};
}

#endif
