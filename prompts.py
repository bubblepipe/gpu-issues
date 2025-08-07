"""Prompts for GPU bug categorization"""

BUG_CATEGORIZATION_PROMPT = """
please categorize the issue in the three following aspects:

1. is this really a bug? 
  1.a no
  1.b sort of no, working as designed, but design is problematic, unexpected, unconventional or surprising 
  1.c sort of yes, but it is because the user misinterpret the api 
  1.d really yes, definitly a bug 
2. from the perspective of the framework users, what exactly is the problem? 
  2.a Compilation and build system 
  2.b type system and shape inference bugs 
  2.c accelerator and backend fragmentation
  2.d distribution and synchronization bugs 
  2.e numerical precision issues
  2.f performance regression, unreasonable duration of execution time 
  2.g resource exhaustion (using too much memory, but also file handles, threads, etc.)
  2.h api / integration issues (interacting with other frameworks or libraries)
  2.i other
  2.j not a bug, not applicable
3. from the perspective the framework developers, how is the problem is being addressed? 
  3.a architectural refactoring that changes the design and structure of the system. 
  3.b adding checks, bounds validation, and error handling
  3.c algorithm optimization  
  3.d addressing race conditions and distributed execution issues
  3.e reconfigure the environment: this often involves closed source toolchains like cuda library, gpu 3.f driver and also hardware implementation. by selecting a micro-arch specific library, update driver 3.g version or workaround the flaws of the hardware,  
  3.h won’t fix, due to man hour constraint or issue priority 
  3.j won’t fix, or proposed with workarounds that not really fixed the root cause, as this problem is foundamentally not fixable, due to that this problem is caused by closed source software or hardware 
  3.k not a bug, not applicable
4. is this bug specific to some accelerator platform, or is it universal? 
sometimes the issue enquirer may only have access to a single accelerator platform and this information could be be available. it is fine to say `don’t know` 
  4.a yes
  4.b no
  4.c don’t know
  4.d not a bug, not applicable 
please reply with only the code representing your option, with comma splitting in between, like `1.a, 2.f, 3.c`. I don't need any further explanation. 
The url to the issue is: 
"""