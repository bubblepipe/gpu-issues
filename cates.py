"""Bug categorization enums and lookup tables"""

from enum import Enum


class BugType(Enum):
    NOT_A_BUG = "1.a not a bug"
    SOURCE_CODE_ISSUE = "1.b source code issue"
    LOW_LEVEL_SOFTWARE_STACK = "1.c low-level software stack, like GPU driver, CUDA toolchain, or even hardware bugs"
    WRONG_API_USAGE = "1.d user called the wrong api"
    OTHER = "1.e other"


class BugSymptom(Enum):
    NOT_ABLE_TO_COMPILE = "2.a not able to compile"
    CRASHES_DURING_RUNTIME = "2.b crashes during runtime"
    PRODUCES_WRONG_RESULT = "2.c produces wrong result"
    UNEXPECTED_RUNTIME_DURATION = "2.d unexpected runtime duration"
    UNEXPECTED_MEMORY_USAGE = "2.e unexpected amount of consumed memory"
    OTHER = "2.f other"
    NOT_A_BUG = "2.g not a bug"


class BugHeterogeneity(Enum):
    UNIVERSAL = "3.a no, universal"
    BACKEND_SPECIFIC = "3.b yes, some backend specifically"
    NOT_APPLICABLE = "3.c not applicable"
    DONT_KNOW = "3.d dont know"


# Legacy alias for backward compatibility
BugUniversality = BugHeterogeneity


BUG_TYPE_LOOKUP = {
    "1.a": BugType.NOT_A_BUG,
    "1.b": BugType.SOURCE_CODE_ISSUE,
    "1.c": BugType.LOW_LEVEL_SOFTWARE_STACK,
    "1.d": BugType.WRONG_API_USAGE,
    "1.e": BugType.OTHER,
}


BUG_SYMPTOM_LOOKUP = {
    "2.a": BugSymptom.NOT_ABLE_TO_COMPILE,
    "2.b": BugSymptom.CRASHES_DURING_RUNTIME,
    "2.c": BugSymptom.PRODUCES_WRONG_RESULT,
    "2.d": BugSymptom.UNEXPECTED_RUNTIME_DURATION,
    "2.e": BugSymptom.UNEXPECTED_MEMORY_USAGE,
    "2.f": BugSymptom.OTHER,
    "2.g": BugSymptom.NOT_A_BUG,
}


BUG_HETEROGENEITY_LOOKUP = {
    "3.a": BugHeterogeneity.UNIVERSAL,
    "3.b": BugHeterogeneity.BACKEND_SPECIFIC,
    "3.c": BugHeterogeneity.NOT_APPLICABLE,
    "3.d": BugHeterogeneity.DONT_KNOW,
}