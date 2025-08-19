"""Prompts for GPU bug categorization"""

BUG_CATEGORIZATION_PROMPT = """
You are tasked with analyzing a GitHub issue and categorizing it across 
multiple dimensions. Please fetch and carefully analyze the issue at the 
provided URL, then categorize it according to the following framework.

## Analysis Framework

### 1. Bug Classification
Determine if this is truly a bug:
- **1.a**: Not a bug - Working as intended with correct design
- **1.b**: Design issue - Working as designed, but the design is problematic, unexpected, or counterintuitive
- **1.c**: User error - Functionality works correctly but user misunderstood the API/documentation
- **1.d**: Confirmed bug - Clear defect in implementation
- **1.e**: Documentation bug - Code works correctly but documentation is incorrect or misleading

### 2. User-Visible Symptoms
Identify the primary symptom experienced by framework users:
- **2.a**: Compilation failure - Code fails to compile
- **2.b**: Runtime crash - Program crashes during execution
- **2.c**: Incorrect results - Output differs from expected (compared to CPU/other implementations)
- **2.d**: Performance degradation - Slower than expected or performance regression
- **2.e**: Memory issues - Unexpected memory consumption or leaks
- **2.f**: API/Interface confusion - Confusing, inconsistent, or unexpected API behavior
- **2.g**: Not applicable - Not a bug

### 3. Root Cause Analysis
From the framework developer's perspective, identify the underlying cause:
- **3.a**: Design flaw - Fundamental issue with system architecture or design
- **3.b**: Missing safeguards - Lacking input validation, bounds checks, error handling, or edge case handling
- **3.c**: Concurrency issue - Race conditions, synchronization problems, or distributed execution issues
- **3.d**: Configuration error - Wrong algorithm selection, precision settings, library versions, or environment setup
- **3.e**: Unknown - Root cause cannot be determined from available information
- **3.f**: Not applicable - Not a bug

### 4. Resolution Status
How was the issue addressed:
- **4.a**: Fixed with code changes - Direct fix implemented
- **4.b**: Workaround provided - Alternative solution suggested without fixing root cause
- **4.c**: Not fixed - Issue remains open or was closed without resolution
- **4.d**: Not applicable - Not a bug

### 5. Platform Specificity
Determine if the bug is platform-dependent:
- **5.a**: Environment-specific - Bug occurs only on certain hardware, vendors, architectures, or OS
- **5.b**: Universal - Bug affects all environments
- **5.c**: Insufficient data - Cannot determine platform specificity from available information
- **5.d**: Not applicable - Not a bug

## Analysis Instructions

1. **For duplicate issues**: If the issue is marked as duplicate, locate and analyze the original issue instead.

2. **Consider all available information**:
   - Issue title and description
   - Error messages and stack traces
   - Hardware/software environment details
   - Reproduction steps
   - Comments and discussion threads
   - Linked pull requests or commits
   - Any test results or benchmarks

3. **Prioritization rule**: When multiple categories could apply, select the most specific and fundamental one. For example, if an issue is both a "performance problem" and "GPU-specific", prioritize based on the root cause.

4. **Output format**:
   - First, provide a detailed analysis explaining your reasoning for each category
   - Include relevant quotes or evidence from the issue
   - End with a summary line containing only the letter codes

**Expected final line format**: `1.d, 2.c, 3.b, 4.a, 5.a`
(Use only letter codes, comma-separated, no additional text)

Please fetch and analyze the issue at the following URL:
"""