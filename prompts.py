"""Prompts for GPU bug categorization"""

BUG_CATEGORIZATION_PROMPT = """
please categorize a issue in the three following aspects:

1. is this really a bug? 
  1.a no
  1.b sort of no, working as designed, but design is problematic, unexpected, unconventional or surprising 
  1.c sort of yes, but it is because the user misinterpret the api 
  1.d really yes, definitely a bug 
  1.e: Documentation bug - the code works as intended but documentation is incorrect/misleading
2. from the perspective of the framework users, what exactly is the problem? 
  2.a compilation and build system 
  2.b type system and shape inference bugs 
  2.c accelerator and backend fragmentation
  2.d distribution and synchronization bugs 
  2.e numerical precision issues
  2.f performance regression, unreasonable duration of execution time 
  2.g resource exhaustion (using too much memory, but also file handles, threads, etc.)
  2.h api / integration issues (interacting with other frameworks or libraries)
  2.i determinism / reproducibility issues - same code produces different results when executed multiple times 
  2.j other
  2.k not a bug, not applicable
3. from the perspective the framework developers, how is the problem is being addressed? 
  3.a architectural refactoring that changes the design and structure of the system. 
  3.b adding checks, initializations, bounds validation, error handling, or edge case and special value fix, like empty tensor, NaN floating point values and etc 
  3.c Mathematical correctness fixes
  3.d specialized algorithm selection, including selecting dedicated kernel on input size, data properties and  memory access patterns (like sparse or dense), hardware-specific algorithm variants (GPU vs CPU), datatype and precision specific implementations (float32 vs float16, int8 vs fp8), optimizing,  parallelization and vectorization optimization 
  3.e other algorithm optimization  
  3.f addressing race conditions and distributed execution issues
  3.g reconfigure the environment: this often involves closed source toolchains like cuda library, gpu driver and also hardware implementation. by selecting a micro-arch specific library, update driver version or workaround the flaws of the hardware,  
  3.h won’t fix, due to man hour constraint or issue priority 
  3.i won’t fix, or proposed with workarounds that not really fixed the root cause, as this problem is fundamentally not fixable, due to that this problem is caused by closed source software or hardware 
  3.j won’t fix, due to other reasons
  3.k not a bug, not applicable
4. is this bug specific to some accelerator platform, or is it universal? 
sometimes the issue enquirer may only have access to a single accelerator platform and this information could be be available. it is fine to say `don’t know` 
  4.a yes, accelerator vendor-specific (NVIDIA only, AMD only, Intel only, etc)
  4.b yes, accelerator architecture-specific (for gpus: Ampere, Turing, etc, or specific compute capabilities. for cpus: x86, ARM, specific instruction sets. )
  4.c yes, OS-specific (Windows, Linux, macOS)
  4.d driver version-specific
  4.e other environmental specific configurations 
  4.f no
  4.g don’t know
  4.h not a bug, not applicable 
5. User expertise required to encounter:
  5.a Beginner - Users following tutorials, using pre-built models, or implementing standard workflows with high-level APIs.
  5.b Intermediate - Users building custom architectures, implementing research papers, or adapting models for specific use cases.
  5.c Advanced - Users extending framework capabilities, optimizing performance, or working with framework internals.
  5.d Not applicable, or not known 
6. Confidence in categorization:
   6.a High - Clear evidence
   6.b Medium - Some uncertainty
   6.c Low - Significant ambiguity

Note: If multiple categories apply, choose the most specific one.
For example, if it's both a "performance regression" and "GPU-specific", 
prioritize the root cause category.

When analyzing the issue, consider:
- Error messages and stack traces
- Hardware/software environment mentioned
- Steps to reproduce
- Discussion in comments
- Pull request or fix details (if linked)

Elaborate your choice and sumarize your conlusion in the last line. 
Expected last line format: "1.d, 2.f, 3.c, 4.a, 5.b, 6.a"
Only include the letter codes, with comma seperated
put all of your explanations before the last line. 

The url to the issue is given below, please fetch the webpage: 
"""