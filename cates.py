"""Bug categorization enums and lookup tables"""

from enum import Enum


class IsReallyBug(Enum):
    NO = "1.a no"
    SORT_OF_NO = "1.b sort of no, working as designed, but design is problematic"
    USER_MISINTERPRET = "1.c sort of yes, but user misinterpret the api"
    DEFINITELY_YES = "1.d really yes, definitely a bug"
    DOCUMENTATION_BUG = "1.e documentation bug"


class UserPerspective(Enum):
    COMPILATION_BUILD = "2.a Compilation and build system"
    TYPE_SHAPE_INFERENCE = "2.b type system and shape inference bugs"
    ACCELERATOR_BACKEND = "2.c accelerator and backend fragmentation"
    DISTRIBUTION_SYNC = "2.d distribution and synchronization bugs"
    NUMERICAL_PRECISION = "2.e numerical precision issues"
    PERFORMANCE_REGRESSION = "2.f performance regression, unreasonable duration"
    RESOURCE_EXHAUSTION = "2.g resource exhaustion (memory, file handles, threads)"
    API_INTEGRATION = "2.h api / integration issues"
    DETERMINISM_REPRODUCIBILITY = "2.i determinism / reproducibility issues"
    OTHER = "2.j other"
    NOT_A_BUG = "2.k not a bug, not applicable"


class DeveloperPerspective(Enum):
    ARCHITECTURAL_REFACTORING = "3.a architectural refactoring"
    ADDING_CHECKS = "3.b adding checks, bounds validation, error handling"
    MATHEMATICAL_CORRECTNESS = "3.c Mathematical correctness fixes"
    SPECIALIZED_ALGORITHM = "3.d specialized algorithm selection"
    OTHER_ALGORITHM_OPTIMIZATION = "3.e other algorithm optimization"
    RACE_CONDITIONS = "3.f addressing race conditions and distributed execution"
    RECONFIGURE_ENVIRONMENT = "3.g reconfigure environment"
    WONT_FIX_CONSTRAINT = "3.h won't fix, due to man hour constraint or priority"
    WONT_FIX_FUNDAMENTAL = "3.i won't fix, fundamentally not fixable"
    WONT_FIX_OTHER = "3.j won't fix, due to other reasons"
    NOT_A_BUG = "3.k not a bug, not applicable"


class AcceleratorSpecific(Enum):
    VENDOR_SPECIFIC = "4.a yes, accelerator vendor-specific"
    ARCHITECTURE_SPECIFIC = "4.b yes, accelerator architecture-specific"
    OS_SPECIFIC = "4.c yes, OS-specific"
    DRIVER_VERSION_SPECIFIC = "4.d driver version-specific"
    ENVIRONMENT_SPECIFIC = "4.e other environmental specific configurations"
    NO = "4.f no"
    DONT_KNOW = "4.g don't know"
    NOT_A_BUG = "4.h not a bug, not applicable"


class UserExpertise(Enum):
    BEGINNER = "5.a Beginner"
    INTERMEDIATE = "5.b Intermediate"
    ADVANCED = "5.c Advanced"
    NOT_APPLICABLE = "5.d Not applicable"


class Confidence(Enum):
    HIGH = "6.a High - Clear evidence"
    MEDIUM = "6.b Medium - Some uncertainty"
    LOW = "6.c Low - Significant ambiguity"


IS_REALLY_BUG_LOOKUP = {
    "1.a": IsReallyBug.NO,
    "1.b": IsReallyBug.SORT_OF_NO,
    "1.c": IsReallyBug.USER_MISINTERPRET,
    "1.d": IsReallyBug.DEFINITELY_YES,
    "1.e": IsReallyBug.DOCUMENTATION_BUG,
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
    "2.i": UserPerspective.DETERMINISM_REPRODUCIBILITY,
    "2.j": UserPerspective.OTHER,
    "2.k": UserPerspective.NOT_A_BUG,
}


DEVELOPER_PERSPECTIVE_LOOKUP = {
    "3.a": DeveloperPerspective.ARCHITECTURAL_REFACTORING,
    "3.b": DeveloperPerspective.ADDING_CHECKS,
    "3.c": DeveloperPerspective.MATHEMATICAL_CORRECTNESS,
    "3.d": DeveloperPerspective.SPECIALIZED_ALGORITHM,
    "3.e": DeveloperPerspective.OTHER_ALGORITHM_OPTIMIZATION,
    "3.f": DeveloperPerspective.RACE_CONDITIONS,
    "3.g": DeveloperPerspective.RECONFIGURE_ENVIRONMENT,
    "3.h": DeveloperPerspective.WONT_FIX_CONSTRAINT,
    "3.i": DeveloperPerspective.WONT_FIX_FUNDAMENTAL,
    "3.j": DeveloperPerspective.WONT_FIX_OTHER,
    "3.k": DeveloperPerspective.NOT_A_BUG,
}

ACCELERATOR_SPECIFIC_LOOKUP = {
    "4.a": AcceleratorSpecific.VENDOR_SPECIFIC,
    "4.b": AcceleratorSpecific.ARCHITECTURE_SPECIFIC,
    "4.c": AcceleratorSpecific.OS_SPECIFIC,
    "4.d": AcceleratorSpecific.DRIVER_VERSION_SPECIFIC,
    "4.e": AcceleratorSpecific.ENVIRONMENT_SPECIFIC,
    "4.f": AcceleratorSpecific.NO,
    "4.g": AcceleratorSpecific.DONT_KNOW,
    "4.h": AcceleratorSpecific.NOT_A_BUG,
}

USER_EXPERTISE_LOOKUP = {
    "5.a": UserExpertise.BEGINNER,
    "5.b": UserExpertise.INTERMEDIATE,
    "5.c": UserExpertise.ADVANCED,
    "5.d": UserExpertise.NOT_APPLICABLE,
}

CONFIDENCE_LOOKUP = {
    "6.a": Confidence.HIGH,
    "6.b": Confidence.MEDIUM,
    "6.c": Confidence.LOW,
}