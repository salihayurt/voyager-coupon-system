#!/usr/bin/env python3
"""
Voyager Coupon System - Setup Verification Script
Checks all components, imports, and configurations
"""

import sys
import os
import traceback
from pathlib import Path
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(title: str):
    """Print a colorful header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}🚀 {title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")

def print_success(message: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_error(message: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_info(message: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.END}")

def check_file_exists(file_path: str) -> bool:
    """Check if a file exists"""
    path = project_root / file_path
    if path.exists():
        print_success(f"{file_path}")
        return True
    else:
        print_error(f"{file_path} - NOT FOUND")
        return False

def test_import(module_path: str, component_name: str = None) -> bool:
    """Test importing a module or component"""
    try:
        if component_name:
            module = __import__(module_path, fromlist=[component_name])
            getattr(module, component_name)
            print_success(f"{module_path}.{component_name}")
        else:
            __import__(module_path)
            print_success(f"{module_path}")
        return True
    except Exception as e:
        error_msg = f"{module_path}"
        if component_name:
            error_msg += f".{component_name}"
        error_msg += f" - {str(e)}"
        print_error(error_msg)
        return False

def verify_file_structure():
    """Verify all required files exist"""
    print_header("📁 File Structure Verification")
    
    files_to_check = [
        # Core domain
        "core/domain/enums.py",
        "core/domain/user.py", 
        "core/domain/coupon.py",
        
        # Business rules
        "core/rules/business_rules.py",
        
        # Learning components
        "learning/state_encoder.py",
        "learning/action_space.py",
        "learning/reward_function.py",
        "learning/q_table.py",
        "learning/trainer.py",
        
        # Agents
        "agents/shared/message.py",
        "agents/shared/context.py",
        "agents/shared/base_agent.py",
        "agents/profitability_agent/agent.py",
        "agents/conversion_agent/agent.py",
        "agents/coordinator_agent/agent.py",
        
        # Orchestration
        "orchestration/workflow_engine.py",
        
        # Configuration
        "config/agents.yaml",
        "config/learning.yaml",
        
        # Requirements
        "requirements.txt"
    ]
    
    success_count = 0
    for file_path in files_to_check:
        if check_file_exists(file_path):
            success_count += 1
    
    print_info(f"Files found: {success_count}/{len(files_to_check)}")
    return success_count == len(files_to_check)

def verify_core_imports():
    """Verify core domain imports"""
    print_header("🏗️  Core Domain Imports")
    
    imports_to_test = [
        ("core.domain.enums", "DomainType"),
        ("core.domain.enums", "SegmentType"), 
        ("core.domain.enums", "ActionType"),
        ("core.domain.user", "User"),
        ("core.domain.user", "UserScores"),
        ("core.domain.coupon", "CouponDecision"),
        ("core.rules.business_rules", "RulesEngine"),
        ("core.rules.business_rules", "BusinessRule"),
        ("core.rules.business_rules", "RuleResult"),
    ]
    
    success_count = 0
    for module_path, component in imports_to_test:
        if test_import(module_path, component):
            success_count += 1
    
    print_info(f"Core imports successful: {success_count}/{len(imports_to_test)}")
    return success_count == len(imports_to_test)

def verify_learning_imports():
    """Verify learning component imports"""
    print_header("🧠 Learning Components Imports")
    
    imports_to_test = [
        ("learning.state_encoder", "StateEncoder"),
        ("learning.action_space", "ActionSpace"),
        ("learning.action_space", "DiscountAction"),
        ("learning.reward_function", "RewardCalculator"),
        ("learning.q_table", "QTable"),
        ("learning.trainer", "QLearningTrainer"),
    ]
    
    success_count = 0
    for module_path, component in imports_to_test:
        if test_import(module_path, component):
            success_count += 1
    
    print_info(f"Learning imports successful: {success_count}/{len(imports_to_test)}")
    return success_count == len(imports_to_test)

def verify_agent_imports():
    """Verify agent imports"""
    print_header("🤖 Agent Imports")
    
    imports_to_test = [
        ("agents.shared.message", "AgentMessage"),
        ("agents.shared.context", "SharedContext"),
        ("agents.shared.base_agent", "BaseVoyagerAgent"),
        ("agents.profitability_agent.agent", "ProfitabilityAgent"),
        ("agents.conversion_agent.agent", "ConversionAgent"),
        ("agents.coordinator_agent.agent", "CoordinatorAgent"),
    ]
    
    success_count = 0
    for module_path, component in imports_to_test:
        if test_import(module_path, component):
            success_count += 1
    
    print_info(f"Agent imports successful: {success_count}/{len(imports_to_test)}")
    return success_count == len(imports_to_test)

def verify_orchestration_imports():
    """Verify orchestration imports"""
    print_header("🎼 Orchestration Imports")
    
    imports_to_test = [
        ("orchestration.workflow_engine", "WorkflowEngine"),
    ]
    
    success_count = 0
    for module_path, component in imports_to_test:
        if test_import(module_path, component):
            success_count += 1
    
    print_info(f"Orchestration imports successful: {success_count}/{len(imports_to_test)}")
    return success_count == len(imports_to_test)

def verify_configurations():
    """Verify configuration files are valid"""
    print_header("⚙️  Configuration Verification")
    
    config_files = [
        "config/agents.yaml",
        "config/learning.yaml"
    ]
    
    success_count = 0
    for config_file in config_files:
        try:
            config_path = project_root / config_file
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if config:
                print_success(f"{config_file} - Valid YAML")
                success_count += 1
            else:
                print_error(f"{config_file} - Empty configuration")
        except Exception as e:
            print_error(f"{config_file} - {str(e)}")
    
    print_info(f"Valid configurations: {success_count}/{len(config_files)}")
    return success_count == len(config_files)

def test_basic_functionality():
    """Test basic system functionality"""
    print_header("🔧 Basic Functionality Tests")
    
    tests_passed = 0
    total_tests = 6
    
    # Test 1: Create basic enums
    try:
        from core.domain.enums import DomainType, SegmentType, ActionType
        domain = DomainType.ENUYGUN_HOTEL
        segment = SegmentType.HIGH_VALUE
        action = ActionType.OFFER_DISCOUNT
        print_success("Enums creation and access")
        tests_passed += 1
    except Exception as e:
        print_error(f"Enums test failed: {e}")
    
    # Test 2: Create UserScores and User
    try:
        from core.domain.user import User, UserScores
        from core.domain.enums import DomainType, SegmentType
        
        scores = UserScores(
            churn_score=0.5,
            activity_score=0.7,
            cart_abandon_score=0.3,
            price_sensitivity=0.6,
            family_score=0.4
        )
        
        user = User(
            user_id="TEST_12345",
            domain=DomainType.ENUYGUN_FLIGHT,
            is_oneway=True,
            user_basket=True,
            segment=SegmentType.PRICE_SENSITIVE_CUSTOMERS,
            scores=scores
        )
        print_success("User and UserScores creation")
        tests_passed += 1
    except Exception as e:
        print_error(f"User creation test failed: {e}")
    
    # Test 3: Business Rules Engine
    try:
        from core.rules.business_rules import RulesEngine
        rules_engine = RulesEngine()
        print_success("RulesEngine initialization")
        tests_passed += 1
    except Exception as e:
        print_error(f"RulesEngine test failed: {e}")
    
    # Test 4: State Encoder
    try:
        from learning.state_encoder import StateEncoder
        encoder = StateEncoder()
        print_success(f"StateEncoder - {encoder.state_space_size:,} total states (now includes family_score)")
        tests_passed += 1
    except Exception as e:
        print_error(f"StateEncoder test failed: {e}")
    
    # Test 5: Action Space
    try:
        from learning.action_space import ActionSpace, DiscountAction
        discount_5 = ActionSpace.get_discount_percentage(DiscountAction.DISCOUNT_5)
        action_10 = ActionSpace.get_action_from_discount(10)
        print_success(f"ActionSpace - {ActionSpace.total_actions()} actions available")
        tests_passed += 1
    except Exception as e:
        print_error(f"ActionSpace test failed: {e}")
    
    # Test 6: Q-Table
    try:
        from learning.q_table import QTable
        q_table = QTable(state_space_size=1000, action_space_size=6)
        print_success("QTable initialization")
        tests_passed += 1
    except Exception as e:
        print_error(f"QTable test failed: {e}")
    
    print_info(f"Functionality tests passed: {tests_passed}/{total_tests}")
    return tests_passed == total_tests

def check_dependencies():
    """Check if required dependencies are installed"""
    print_header("📦 Dependencies Check")
    
    required_packages = [
        "pydantic",
        "numpy", 
        "tqdm",
        "pyyaml"
    ]
    
    # Note: agno would be checked here but it's not in requirements.txt
    # We'll add it as a warning
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print_success(f"{package}")
        except ImportError:
            print_error(f"{package} - NOT INSTALLED")
            missing_packages.append(package)
    
    # Check for agno (special case)
    try:
        __import__("agno")
        print_success("agno")
    except ImportError:
        print_warning("agno - NOT INSTALLED (required for agents)")
        missing_packages.append("agno")
    
    if missing_packages:
        print_info(f"Missing packages: {', '.join(missing_packages)}")
        print_info("Install with: pip install " + " ".join(missing_packages))
    
    return len(missing_packages) == 0

def main():
    """Main verification function"""
    print(f"{Colors.BOLD}{Colors.MAGENTA}")
    print("🎯 VOYAGER COUPON SYSTEM - SETUP VERIFICATION")
    print("=" * 60)
    print(f"{Colors.END}")
    
    # Run all verification checks
    checks = [
        ("File Structure", verify_file_structure),
        ("Dependencies", check_dependencies),
        ("Core Imports", verify_core_imports),
        ("Learning Imports", verify_learning_imports),  
        ("Agent Imports", verify_agent_imports),
        ("Orchestration Imports", verify_orchestration_imports),
        ("Configurations", verify_configurations),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print_error(f"{check_name} verification failed: {e}")
            results.append((check_name, False))
    
    # Summary
    print_header("📊 Verification Summary")
    
    passed_checks = sum(1 for _, result in results if result)
    total_checks = len(results)
    
    for check_name, result in results:
        if result:
            print_success(f"{check_name}")
        else:
            print_error(f"{check_name}")
    
    print(f"\n{Colors.BOLD}", end="")
    if passed_checks == total_checks:
        print(f"{Colors.GREEN}🎉 ALL CHECKS PASSED! ({passed_checks}/{total_checks})")
        print(f"✨ Voyager Coupon System is ready to go! ✨{Colors.END}")
        return 0
    else:
        print(f"{Colors.YELLOW}⚠️  SOME CHECKS FAILED ({passed_checks}/{total_checks})")
        print(f"🔧 Please fix the issues above before proceeding.{Colors.END}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
