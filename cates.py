"""Bug categorization enums and lookup tables"""

from enum import Enum


# Question 1: Bug Classification
class IsReallyBug(Enum):
    NOT_A_BUG = "1.a Not a bug"
    DESIGN_ISSUE = "1.b Design issue"
    USER_ERROR = "1.c User error"
    CONFIRMED_BUG = "1.d Confirmed bug"
    DOCUMENTATION_BUG = "1.e Documentation bug"
    DONT_KNOW = "1.f Don't know"


# Question 2: User-Visible Symptoms  
class UserPerspective(Enum):
    COMPILATION_FAILURE = "2.a Compilation failure"
    RUNTIME_CRASH = "2.b Runtime crash"
    INCORRECT_RESULTS = "2.c Incorrect results"
    PERFORMANCE_DEGRADATION = "2.d Performance degradation"
    MEMORY_ISSUES = "2.e Memory issues"
    API_CONFUSION = "2.f API/Interface confusion"
    NOT_APPLICABLE = "2.g Not applicable"
    DONT_KNOW = "2.h Don't know"
    OTHER = "2.i Other"


# Question 3: Root Cause Analysis
class DeveloperPerspective(Enum):
    DESIGN_FLAW = "3.a Design flaw"
    MISSING_SAFEGUARDS = "3.b Missing safeguards"
    CONCURRENCY_ISSUE = "3.c Concurrency issue"
    CONFIGURATION_ERROR = "3.d Configuration error"
    UNKNOWN = "3.e Unknown"
    NOT_APPLICABLE = "3.f Not applicable"
    OTHER = "3.g Other"


# Question 4: Resolution Status
class AcceleratorSpecific(Enum):
    FIXED = "4.a Fixed with code changes"
    WORKAROUND = "4.b Workaround provided"
    NOT_FIXED = "4.c Not fixed"
    NOT_APPLICABLE = "4.d Not applicable"
    DONT_KNOW = "4.e Don't know"


# Question 5: Platform Specificity
class UserExpertise(Enum):
    ENVIRONMENT_SPECIFIC = "5.a Environment-specific"
    UNIVERSAL = "5.b Universal"
    INSUFFICIENT_DATA = "5.c Insufficient data"
    NOT_APPLICABLE = "5.d Not applicable"

# Lookup dictionaries
IS_REALLY_BUG_LOOKUP = {
    "1.a": IsReallyBug.NOT_A_BUG,
    "1.b": IsReallyBug.DESIGN_ISSUE,
    "1.c": IsReallyBug.USER_ERROR,
    "1.d": IsReallyBug.CONFIRMED_BUG,
    "1.e": IsReallyBug.DOCUMENTATION_BUG,
    "1.f": IsReallyBug.DONT_KNOW,
}

USER_PERSPECTIVE_LOOKUP = {
    "2.a": UserPerspective.COMPILATION_FAILURE,
    "2.b": UserPerspective.RUNTIME_CRASH,
    "2.c": UserPerspective.INCORRECT_RESULTS,
    "2.d": UserPerspective.PERFORMANCE_DEGRADATION,
    "2.e": UserPerspective.MEMORY_ISSUES,
    "2.f": UserPerspective.API_CONFUSION,
    "2.g": UserPerspective.NOT_APPLICABLE,
    "2.h": UserPerspective.DONT_KNOW,
    "2.i": UserPerspective.OTHER,
}

DEVELOPER_PERSPECTIVE_LOOKUP = {
    "3.a": DeveloperPerspective.DESIGN_FLAW,
    "3.b": DeveloperPerspective.MISSING_SAFEGUARDS,
    "3.c": DeveloperPerspective.CONCURRENCY_ISSUE,
    "3.d": DeveloperPerspective.CONFIGURATION_ERROR,
    "3.e": DeveloperPerspective.UNKNOWN,
    "3.f": DeveloperPerspective.NOT_APPLICABLE,
    "3.g": DeveloperPerspective.OTHER,
}

ACCELERATOR_SPECIFIC_LOOKUP = {
    "4.a": AcceleratorSpecific.FIXED,
    "4.b": AcceleratorSpecific.WORKAROUND,
    "4.c": AcceleratorSpecific.NOT_FIXED,
    "4.d": AcceleratorSpecific.NOT_APPLICABLE,
    "4.e": AcceleratorSpecific.DONT_KNOW,
}

USER_EXPERTISE_LOOKUP = {
    "5.a": UserExpertise.ENVIRONMENT_SPECIFIC,
    "5.b": UserExpertise.UNIVERSAL,
    "5.c": UserExpertise.INSUFFICIENT_DATA,
    "5.d": UserExpertise.NOT_APPLICABLE,
}