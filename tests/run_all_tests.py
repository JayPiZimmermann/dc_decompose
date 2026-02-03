"""
Run all DC decomposition tests and generate comprehensive report.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import time
from pathlib import Path


def run_test_script(script_path: str) -> tuple[bool, str]:
    """Run a test script and return success status and output."""
    try:
        print(f"🚀 Running {script_path}...")
        start_time = time.time()
        
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, timeout=300)
        
        elapsed = time.time() - start_time
        
        success = result.returncode == 0
        output = result.stdout + result.stderr
        
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {status} in {elapsed:.1f}s")
        
        return success, output
        
    except subprocess.TimeoutExpired:
        return False, "Test timed out after 300 seconds"
    except Exception as e:
        return False, f"Test execution error: {e}"


def main():
    """Run all tests and generate report."""
    print("🧪 DC Decomposition - Comprehensive Test Suite")
    print("=" * 60)
    
    test_dir = Path(__file__).parent
    
    # Define test scripts in order
    test_scripts = [
        "test_basic_layers.py",
        "test_residual_connections.py", 
        "test_complex_models.py",
    ]
    
    results = {}
    total_time = 0
    
    print(f"Found {len(test_scripts)} test scripts to run...")
    print()
    
    # Run each test script
    for script_name in test_scripts:
        script_path = test_dir / script_name
        if not script_path.exists():
            print(f"⚠️  Skipping {script_name} - file not found")
            results[script_name] = (False, "File not found")
            continue
        
        start_time = time.time()
        success, output = run_test_script(str(script_path))
        elapsed = time.time() - start_time
        total_time += elapsed
        
        results[script_name] = (success, output)
        
        if success:
            print(f"✅ {script_name} completed successfully")
        else:
            print(f"❌ {script_name} failed")
            # Print first few lines of error for debugging
            lines = output.split('\\n')
            error_lines = [line for line in lines if 'Error' in line or 'Failed' in line or '❌' in line]
            if error_lines:
                print(f"   Error preview: {error_lines[0][:100]}...")
        print()
    
    # Generate summary report
    print("=" * 60)
    print("📊 TEST SUMMARY REPORT")
    print("=" * 60)
    
    passed = sum(1 for success, _ in results.values() if success)
    total = len(results)
    
    print(f"Overall: {passed}/{total} test scripts passed ({100*passed/total:.1f}%)")
    print(f"Total execution time: {total_time:.1f}s")
    print()
    
    # Detailed results
    print("Detailed Results:")
    print("-" * 40)
    for script_name, (success, output) in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{script_name:<25} {status}")
        
        if success:
            # Extract summary from output
            lines = output.split('\\n')
            summary_lines = [line for line in lines if 'SUMMARY:' in line or 'tests passed' in line]
            if summary_lines:
                print(f"   {summary_lines[0].strip()}")
        else:
            # Show error details
            lines = output.split('\\n')
            error_lines = [line for line in lines if '❌' in line and 'Failed models' in line]
            if error_lines:
                print(f"   {error_lines[0].strip()}")
        print()
    
    # Overall assessment
    print("=" * 60)
    print("🔍 ANALYSIS & RECOMMENDATIONS")
    print("=" * 60)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Stacked tensor DC decomposition is working correctly")
        print("✅ Hook bypass functionality is working")
        print("✅ Basic layer support is complete")
        print("✅ Complex architectures are supported")
    else:
        print("📝 TEST RESULTS ANALYSIS:")
        print()
        
        # Analyze failures
        basic_failed = not results.get("test_basic_layers.py", (True, ""))[0]
        residual_failed = not results.get("test_residual_connections.py", (True, ""))[0] 
        complex_failed = not results.get("test_complex_models.py", (True, ""))[0]
        
        if basic_failed:
            print("❌ Basic layer tests failed - core functionality issues")
            print("   Priority: HIGH - Fix basic layer support first")
        else:
            print("✅ Basic layer tests passed - core functionality working")
            
        if residual_failed:
            print("❌ Residual connection tests failed")
            print("   Issue: Tensor addition operations not intercepted")
            print("   Solution: Implement tensor operation hooks or use DCAdd modules")
        else:
            print("✅ Residual connection tests passed (check reconstruction errors)")
            
        if complex_failed:
            print("❌ Complex model tests failed")
            print("   Issue: Advanced architectures have unsupported operations")
            print("   Solution: Add missing layer types and operation support")
        else:
            print("✅ Complex model tests passed")
    
    print()
    print("🔧 NEXT STEPS:")
    if basic_failed:
        print("1. Fix basic layer issues first")
    if residual_failed or complex_failed:
        print("1. Implement tensor addition operation hooks") 
        print("2. Add support for more complex operations")
        print("3. Handle multi-branch architectures properly")
    if passed == total:
        print("1. Consider testing with real-world models")
        print("2. Add performance benchmarks")
        print("3. Test on larger models and datasets")
    
    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()