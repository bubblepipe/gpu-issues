"""Bug categorization enums and lookup tables"""

from enum import Enum


class IsReallyBug(Enum):
    NO = "1.a no"
    SORT_OF_NO = "1.b sort of no, working as designed, but design is problematic"
    USER_MISINTERPRET = "1.c sort of yes, but user misinterpret the api"
    DEFINITELY_YES = "1.d really yes, definitely a bug"


class UserPerspective(Enum):
    COMPILATION_BUILD = "2.a Compilation and build system"
    TYPE_SHAPE_INFERENCE = "2.b type system and shape inference bugs"
    ACCELERATOR_BACKEND = "2.c accelerator and backend fragmentation"
    DISTRIBUTION_SYNC = "2.d distribution and synchronization bugs"
    NUMERICAL_PRECISION = "2.e numerical precision issues"
    PERFORMANCE_REGRESSION = "2.f performance regression, unreasonable duration"
    RESOURCE_EXHAUSTION = "2.g resource exhaustion (memory, file handles, threads)"
    API_INTEGRATION = "2.h api / integration issues"
    OTHER = "2.i other"
    NOT_A_BUG = "2.j not a bug, not applicable"


class DeveloperPerspective(Enum):
    ARCHITECTURAL_REFACTORING = "3.a architectural refactoring"
    ADDING_CHECKS = "3.b adding checks, bounds validation, error handling"
    ALGORITHM_OPTIMIZATION = "3.c algorithm optimization"
    RACE_CONDITIONS = "3.d addressing race conditions and distributed execution"
    RECONFIGURE_ENVIRONMENT = "3.e reconfigure environment (cuda library, gpu driver)"
    WONT_FIX_CONSTRAINT = "3.h won't fix, due to man hour constraint or priority"
    WONT_FIX_FUNDAMENTAL = "3.j won't fix, fundamentally not fixable (closed source)"
    NOT_A_BUG = "3.k not a bug, not applicable"


class AcceleratorSpecific(Enum):
    YES = "4.a yes"
    NO = "4.b no"
    DONT_KNOW = "4.c don't know"
    NOT_A_BUG = "4.d not a bug, not applicable"


IS_REALLY_BUG_LOOKUP = {
    "1.a": IsReallyBug.NO,
    "1.b": IsReallyBug.SORT_OF_NO,
    "1.c": IsReallyBug.USER_MISINTERPRET,
    "1.d": IsReallyBug.DEFINITELY_YES,
}


USER_PERSPECTIVE_LOOKUP = {
    "2.a": UserPerspective.COMPILATION_BUILD,
    "2.b": UserPerspective.TYPE_SHAPE_INFERENCE,
    "2.c": UserPerspective.ACCELERATOR_BACKEND,
    "2.d": UserPerspective.DISTRIBUTION_SYNC,
    "2.e": UserPerspective.NUMERICAL_PRECISION,
    "2.f": UserPerspective.PERFORMANCE_REGRESSION,
    "2.g": UserPerspective.RESOURCE_EXHAUSTION,
    "2.h": UserPerspective.API_INTEGRATION,
    "2.i": UserPerspective.OTHER,
    "2.j": UserPerspective.NOT_A_BUG,
}


DEVELOPER_PERSPECTIVE_LOOKUP = {
    "3.a": DeveloperPerspective.ARCHITECTURAL_REFACTORING,
    "3.b": DeveloperPerspective.ADDING_CHECKS,
    "3.c": DeveloperPerspective.ALGORITHM_OPTIMIZATION,
    "3.d": DeveloperPerspective.RACE_CONDITIONS,
    "3.e": DeveloperPerspective.RECONFIGURE_ENVIRONMENT,
    "3.h": DeveloperPerspective.WONT_FIX_CONSTRAINT,
    "3.j": DeveloperPerspective.WONT_FIX_FUNDAMENTAL,
    "3.k": DeveloperPerspective.NOT_A_BUG,
}

ACCELERATOR_SPECIFIC_LOOKUP = {
    "4.a": AcceleratorSpecific.YES,
    "4.b": AcceleratorSpecific.NO,
    "4.c": AcceleratorSpecific.DONT_KNOW,
    "4.d": AcceleratorSpecific.NOT_A_BUG,
}