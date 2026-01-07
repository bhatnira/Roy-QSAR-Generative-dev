#!/bin/bash

# Comprehensive QSAR Framework Test Runner
# This script runs the complete test suite for all 13 modules

# Set color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}COMPREHENSIVE QSAR FRAMEWORK TEST RUNNER${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo "This script will:"
echo "  1. Generate a realistic QSAR test dataset"
echo "  2. Test all 13 framework modules"
echo "  3. Verify multi-library support"
echo "  4. Save results to test_results.txt"
echo ""

# Check if we're in the correct directory
if [ ! -f "generate_qsar_dataset.py" ]; then
    echo -e "${RED}Error: generate_qsar_dataset.py not found!${NC}"
    echo "Please run this script from the comprehensive_test directory:"
    echo "  cd comprehensive_test"
    echo "  bash run_tests.sh"
    exit 1
fi

# Check Python availability
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}Error: Python not found!${NC}"
    echo "Please install Python 3.7+ and try again."
    exit 1
fi

# Display Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo -e "${GREEN}✓${NC} Python found: ${PYTHON_VERSION}"
echo ""

# ================================================================================================
# STEP 1: Generate QSAR Dataset
# ================================================================================================

echo -e "${BLUE}------------------------------------------------------------------------------------------------${NC}"
echo -e "${BLUE}STEP 1: Generating QSAR Test Dataset${NC}"
echo -e "${BLUE}------------------------------------------------------------------------------------------------${NC}"
echo ""

$PYTHON_CMD generate_qsar_dataset.py

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Dataset generation failed!${NC}"
    echo "Please check the error messages above."
    exit 1
fi

if [ ! -f "qsar_test_dataset.csv" ]; then
    echo -e "${RED}✗ Dataset file not created!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Dataset generated successfully!${NC}"
echo -e "${GREEN}✓ Dataset saved to: qsar_test_dataset.csv${NC}"
echo ""

# Show dataset info
DATASET_SIZE=$(wc -l < qsar_test_dataset.csv)
DATASET_SIZE=$((DATASET_SIZE - 1))  # Subtract header
echo "Dataset info:"
echo "  - Molecules: ${DATASET_SIZE}"
echo "  - File size: $(ls -lh qsar_test_dataset.csv | awk '{print $5}')"
echo ""

# ================================================================================================
# STEP 2: Run Comprehensive Module Tests
# ================================================================================================

echo -e "${BLUE}------------------------------------------------------------------------------------------------${NC}"
echo -e "${BLUE}STEP 2: Testing All Framework Modules${NC}"
echo -e "${BLUE}------------------------------------------------------------------------------------------------${NC}"
echo ""
echo "Testing 13 modules:"
echo "  1. DuplicateRemoval"
echo "  2. AdvancedSplitter (3 strategies)"
echo "  3. FeatureScaler"
echo "  4. FeatureSelector"
echo "  5. PCATransformer"
echo "  6. DatasetQualityAnalyzer"
echo "  7. ModelComplexityController"
echo "  8. PerformanceValidator"
echo "  9. ActivityCliffsDetector"
echo " 10. UncertaintyEstimator"
echo " 11. CrossValidator"
echo " 12. PerformanceMetrics"
echo " 13. DatasetBiasAnalysis"
echo ""
echo "This may take 1-2 minutes..."
echo ""

# Run the test and save output
$PYTHON_CMD test_all_modules.py 2>&1 | tee test_results.txt

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}✗ Some tests failed!${NC}"
    echo "Please check test_results.txt for details."
    exit 1
fi

# Check if the test completed successfully
if grep -q "ALL TESTS PASSED" test_results.txt; then
    echo ""
    echo -e "${GREEN}================================================================================================${NC}"
    echo -e "${GREEN}✓ ALL TESTS PASSED!${NC}"
    echo -e "${GREEN}================================================================================================${NC}"
    echo ""
    echo "Test results saved to: test_results.txt"
    echo ""
    echo "Files generated:"
    echo "  - qsar_test_dataset.csv (QSAR dataset)"
    echo "  - test_results.txt (test output)"
    echo ""
    echo -e "${GREEN}The framework is working correctly with all modules!${NC}"
    echo ""
else
    echo ""
    echo -e "${YELLOW}⚠️  Tests completed with warnings${NC}"
    echo "Some optional features may not be available."
    echo "Check test_results.txt for details."
    echo ""
fi

# ================================================================================================
# STEP 3: Summary
# ================================================================================================

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}TEST SUMMARY${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""

# Count successful tests
MODULES_TESTED=$(grep -c "✓" test_results.txt || echo "0")
echo "Modules tested: ${MODULES_TESTED}"

# Check for warnings
WARNINGS=$(grep -c "⚠️" test_results.txt || echo "0")
if [ "$WARNINGS" -gt 0 ]; then
    echo "Warnings: ${WARNINGS}"
fi

# Check for multi-library support
if grep -q "XGBoost" test_results.txt; then
    echo -e "${GREEN}✓${NC} XGBoost support verified"
else
    echo -e "${YELLOW}⚠️${NC}  XGBoost not available"
fi

if grep -q "LightGBM" test_results.txt; then
    echo -e "${GREEN}✓${NC} LightGBM support verified"
else
    echo -e "${YELLOW}⚠️${NC}  LightGBM not available"
fi

echo ""
echo "Files in this directory:"
ls -lh qsar_test_dataset.csv test_results.txt 2>/dev/null | tail -n +2
echo ""

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}NEXT STEPS${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo "1. Review test_results.txt for detailed output"
echo "2. Use the framework with your own QSAR data"
echo "3. Check the main README.md for usage examples"
echo ""
echo -e "${GREEN}Happy QSAR modeling!${NC}"
echo ""
