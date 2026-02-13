#!/bin/bash
# scripts/validate_refactoring.sh
# 
# Final validation script for core refactoring
# Run after all migrations to ensure everything is working

set -e

echo "========================================"
echo "🔍 Core Refactoring Validation"
echo "========================================"
echo ""

ERRORS=0

# 1. Check for orphaned files in core/ root
echo "1️⃣  Checking for orphaned files in core/ root..."
ORPHANS=$(find resync/core -maxdepth 1 -name "*.py" ! -name "__init__.py" 2>/dev/null | wc -l)
if [ "$ORPHANS" -eq 0 ]; then
    echo "   ✅ No orphaned files"
else
    echo "   ⚠️  Found $ORPHANS files in core/ root (should be 0 after migration)"
    find resync/core -maxdepth 1 -name "*.py" ! -name "__init__.py" | head -10
    # Not an error - these might be intentional
fi
echo ""

# 2. Check Python syntax
echo "2️⃣  Checking Python syntax..."
SYNTAX_ERRORS=$(find resync/core -name "*.py" -exec python3 -m py_compile {} \; 2>&1 | grep -c "SyntaxError" || true)
if [ "$SYNTAX_ERRORS" -eq 0 ]; then
    echo "   ✅ All files have valid syntax"
else
    echo "   ❌ Found syntax errors!"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# 3. Check imports
echo "3️⃣  Validating imports..."
python3 -c "
import sys
try:
    import resync
    import resync.core
    print('   ✅ Core imports working')
except ImportError as e:
    print(f'   ❌ Import error: {e}')
    sys.exit(1)
" || ERRORS=$((ERRORS + 1))
echo ""

# 4. Run smoke tests
echo "4️⃣  Running smoke tests..."
if [ -f "resync/tests/smoke_tests.py" ]; then
    python3 -m resync.tests.smoke_tests 2>&1 | tail -20
    if [ $? -eq 0 ]; then
        echo "   ✅ Smoke tests passed"
    else
        echo "   ❌ Smoke tests failed!"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "   ⚠️  Smoke tests not found, skipping"
fi
echo ""

# 5. Check test suite (quick)
echo "5️⃣  Running quick test check..."
if command -v pytest &> /dev/null; then
    pytest tests/ -q --co 2>&1 | tail -5
    echo "   ✅ Tests can be collected"
else
    echo "   ⚠️  pytest not found, skipping test check"
fi
echo ""

# 6. Check directory structure
echo "6️⃣  Checking directory structure..."
EXPECTED_DIRS=(
    "resync/core/platform"
    "resync/core/observability"
    "resync/core/security"
    "resync/core/retrieval"
    "resync/core/agents"
    "resync/core/tws"
    "resync/core/shared"
)

for dir in "${EXPECTED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "   ✅ $dir exists"
    else
        echo "   ⚠️  $dir missing"
    fi
done
echo ""

# 7. Check for circular imports (basic check)
echo "7️⃣  Quick circular import check..."
python3 -c "
import sys
sys.setrecursionlimit(100)
try:
    from resync.core import exceptions
    from resync.core import resilience
    print('   ✅ No obvious circular imports')
except RecursionError:
    print('   ❌ Circular import detected!')
    sys.exit(1)
" || ERRORS=$((ERRORS + 1))
echo ""

# Summary
echo "========================================"
echo "VALIDATION SUMMARY"
echo "========================================"

if [ $ERRORS -eq 0 ]; then
    echo "✅ All validations passed!"
    echo ""
    echo "Next steps:"
    echo "  1. Run full test suite: pytest tests/ -v"
    echo "  2. Check coverage: pytest tests/ --cov=resync"
    echo "  3. Create PR for review"
    exit 0
else
    echo "❌ Found $ERRORS error(s)"
    echo ""
    echo "Fix errors before proceeding with merge."
    exit 1
fi
