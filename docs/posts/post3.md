# C++ Shenanigans
If you find this article useful, leave a thumbs up or a comment [below](#comments) !

_Last updated: {{ git_revision_date_localized }}_.  
<!--- ![Visits](https://hitscounter.dev/api/hit?url=https%3A%2F%2Frbourgeois33.github.io%2Fposts%2Fpost3%2F&label=Visits) --->

## Why this blog-post?
**November 2025:** After years of using C++ without *really* trying to understand it (shame), I have finally decided to take it seriously. No more ChatGPT'ing the error messages and forgetting immediately about the fixes. No more copy-pasting code samples here and there and trying to fix them blind until it finally compiles. Moreover, I very much relate to [Lorenzo Miaggi](https://mlwithouttears.com/about/) *"Blogging is one of the most effective ways I learn. If I don't write, I don't really understand"*. This is why this post exists: It's a disorganized list of C++ shenanigans that I discovered along the way, working in [TRUST](https://cea-trust-platform.github.io/).

## The shenanigans

### When is `.template` required? And why?

Let's say that you want to write a templated function,
```c++
template<int N>
void foo(){
    std::cout<< N <<std::endl;
}
```
calling it in main is easy:
```c++
foo<10>(); //Okay in main
```
This will print `10` ([see it on gobolt](https://godbolt.org/z/sK1djxj1Y)). The compiler reads our file from top to bottom, first stumbles upon the definition of `foo`. Then, in `main`, it sees `foo<10>` and instantiates the corresponding version of foo that will be called at run-time:

```cpp
template<>
void foo<10>(){
    std::cout<< 10 <<std::endl;
}
```

But things can get a little more complicated calling template functions that are members of template classes. Let's first consider a simple template class:
```cpp
template<int M>
class A
{
public:
    A(){};

    template<int N>
    void foo(){
        std::cout << N <<std::endl;
    }
};
```

Our class `A` is templated by `int M` (which has no use) and has a template member function `foo` that simply prints its template `int` argument, `N`. Calling `foo` in `main` is perfectly fine ([see it on godbolt](https://godbolt.org/z/9qxc8v883)):

```cpp
A<42> a;
a.foo<10>(); //Okay in main!
```

But things start to go south when trying to call `foo` as a member function of an object whose type is not a fully explicit reference to `A`, e.g. `A<42>`. This for instance:

```cpp
// [Our class A]

// ...

template<typename object_t> 
void create_a_object_and_call_foo(){
    object_t object;
    object.foo<10>(); //Ambiguous! 
}

// ...

int main(){
    create_a_object_and_call_foo<A<42>>();
    return 0;
}
```
does not compile! ([see it on compiler explorer](https://godbolt.org/z/carf9Gh4q)). Indeed, when the compiler reaches the line `object.foo<10>();`, it has no idea what `object_t` is and that we plan it to be `A<42>` lower in the file. As a result, it has no idea what `object.foo` is. In particular, **the compiler cannot tell if foo is a template function or not**, leading to an ambiguous meaning:

- If `object.foo` is a template function with an int parameter, this line means: *call `object`'s `foo` method with template argument `10`*. This is what we have in mind.
- If `object.foo` is not a template function, the line means: *compute `object.foo`, compare it to `10`, and compare this bool result to `()`* which is wrong syntactically. 

As a result, we need to hint the compiler that `object.foo` is indeed a template function with `.template`:

```cpp
template<typename object_t> 
void create_a_object_and_call_foo(){
    object_t object;
    object.template foo<10>(); //not ambiguous!
}
```
This now compiles! ([see it on compiler explorer](https://godbolt.org/z/KcahPPj3x)). 

Since `object_t` could be anything, `object.foo` could be anything too! This can also get trickier. For example, even if we tell the compiler that `object_t` is in fact an instance of `A`:

```cpp
template<int M> 
void create_a_A_and_call_foo(){
    A<M> a;
    a.foo<10>(); //Ambiguous!
}
```
it still does not compile, for the exact same reason! `A<M>` is not a fully explicit reference to an instance of `A`. The compiler cannot make the link between `A<M>` and our class. Indeed, what if we instantiated a special case of `A`, let's say `A<76>` for which `A<76>::foo` is not template? We have an ambiguous statement. As a result, `.template` can be necessary in many seemingly different contexts ([see a few on compiler explorer and fix them yourself](https://godbolt.org/z/7zjYKv3df)) that all boil down to the same underlying rule:

**You need to add `.template` to all calls to template member functions of a template class in a context where the class is not fully explicited.**

### What does virtual means?

Let's consider the following code:
```cpp
#include <iostream>

struct Person {
    void what_am_I() {
        std::cout << "I'm a person!" << std::endl;
    }
};

struct Student : Person {
    void what_am_I() {
        std::cout << "I'm a person and also a student!" << std::endl;
    }
};

void what_is(Person &p) {
    p.what_am_I();
}

int main() {
    Person p;
    Student s;
    what_is(p);
    what_is(s);
}
```
We create a struct named `Person` and a derived class `Student`. Each struct has a member function `what_am_I`. A `Person` that is not a `Student` simply says `"I'm a person!"`. A good `Student` knows about item 32 of [[S. Meyer, 2005]](#1) and says `"I'm a person and also a student!"`. Then, we write a function `what_is` that takes any `Person` as an input and asks what it is. It is valid to give a `Student` to that function: a `Student` is a `Person` so it will be downcasted by reference. This is precisely what we do in `main`. Try and guess, what will this program output? [See it on compiler explorer](https://godbolt.org/z/1n19cjdhW).

Too much suspense. This program prints:
```bash
I'm a person!
I'm a person!
```
which is not wrong, but not necessarily what we intended. In `what_is`, `what_am_I` is statically linked to `Person::what_am_I`. We could change this by writing a special `what_is` for `Student`s, but that would be duplicated code. We could also template `what_is`, but this option is not always available in complex code bases. Moreover, if we reference-downcast the `Student` to a `Person` in another context, and call `what_is` on it, the `Person` version will be selected again. Instead let's change the behavior of our function with the `virtual` keyword:

```cpp
#include <iostream>

struct Person {
    virtual void what_am_I() {
        std::cout << "I'm a person!" << std::endl;
    }
};

struct Student : Person {
    void what_am_I() override {
        std::cout << "I'm a person and also a student!" << std::endl;
    }
};

void what_is(Person &p) {
    p.what_am_I();
}

int main() {
    Person p;
    Student s;
    what_is(p);
    what_is(s);
}
```
This program prints 
```bash
I'm a person!
I'm a person and also a student!
```
[See it on compiler explorer](https://godbolt.org/z/ffbr5r1M4). The `virtual` keyword in front of the base class `Person`'s `what_am_I` definition means that it will provide an interface (function name `what_am_I`, return type `void` and input types `None`) as well as a default implementation to all its derived classes. `Student` is free to override this default behavior. It provide its own implementation of `what_am_I` with the keyword `override`. Therefore all calls to a `Person`'s `what_am_I` will link at runtime with its deepest version applicable of `what_am_I`. In particular, even if a `Student` is downcasted to a `Person` by reference like in `what_is`, `p.what_am_I` will resolve to `Student::what_am_I`.

**Note:** The version with `virtual` is not more or less correct than the original one, it just means something different.

**Understand the `virtual` to finely control member function calls in object hierarchies.**


See Item 34 of [[S. Meyer, 2005]](#1) for more on virtual functions. In Item 36, the author tells us to **never** redifine an inherited non-virtual function. While I strongly, agree with this advice, you, like me, may have to work in a code-base where this rule is already violated.


## References

<a id="1">[S. Meyer, 2005]</a> Effective C++: 55 Specific Ways to Improve Your Programs and Designs. [PDF](https://dl.e-bookfa.ir/freebooks/Effective%20C++,%203rd%20Edition%20by%20Scott%20Meyers%20%28e-bookfa.ir%29.pdf). (Thanks Adrien for sharing !)


## Special thanks

Thanks to my colleagues of the LCAN team for always taking the time to answer my dumb C++ questions.

## Comments 
<script src="https://giscus.app/client.js"
        data-repo="rbourgeois33/rbourgeois33.github.io"
        data-repo-id="R_kgDOPmDw5g"
        data-category="General"
        data-category-id="DIC_kwDOPmDw5s4CvWWl"
        data-mapping="pathname"
        data-strict="0"
        data-reactions-enabled="1"
        data-emit-metadata="0"
        data-input-position="top"
        data-theme="preferred_color_scheme"
        data-lang="en"
        crossorigin="anonymous"
        async>
</script>