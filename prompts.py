"""Prompts for GPU bug categorization"""

BUG_CATEGORIZATION_PROMPT = """
please categorize the issue in the three following aspects:

1.  bug type: if it is a bug, it is a source code issue, or caused by stuff at a lower level, like GPU drivers, CUDA toolchain, hardware implementation? the answer to this question should be in one of the following: 
    1.a `not a bug`
    1.b `source code issue`
    1.c `low-level software stack, like GPU driver, CUDA toolchain, or even hardware bugs`
    1.d `user called the wrong api`
    1.e `other`
2. 
    2.a `program not able to compile`
    2.b `crashes during runtime`
    2.c `produces wrong result` (comparing to other gpu or cpu)
    2.d `unexpected runtime duration`
    2.e `unexpected amount of consumed memory`
    2.f `other`
    2.g `not a bug`
3. is the bug universal across all backends, or does it only cause problem within some specific architecture? please only answer 3.b when you have strong evidence. 
    3.a `no, universal`
    3.b `yes, some backend specifically`
    3.c `not applicable`
    3.d `dont know`
please reply with only the code representing your option, with comma splitting in between, like `1.a, 2.f, 3.c`. I don't need any further explanation. 
The url to the issue is: 
"""