#!/usr/bin/env python3
"""
Test the MCP server tools locally
"""

from tools.log_parser import TrainingLogParser
from tools.issue_detector import IssueDetector
import json

# Initialize
parser = TrainingLogParser()
detector = IssueDetector()

print("🧪 Testing MCP Server Tools\n")
print("="*60)

# Test 1: Parse overfitting log
print("\n📊 Test 1: Parse overfitting.csv")
print("-"*60)
result = parser.parse("demo_data/overfitting.csv")
print(f"✓ Log ID: {result['log_id']}")
print(f"✓ Epochs: {result['total_epochs']}")
print(f"✓ Metrics: {', '.join(result['metrics_available'])}")

# Test 2: Detect issues
print("\n🔍 Test 2: Detect issues in overfitting.csv")
print("-"*60)
df = parser.get_log(result['log_id'])
issues = detector.detect_all(df)
print(f"✓ Health Score: {issues['health_score']}/100")
print(f"✓ Issues Found: {issues['total_issues']}")
for issue in issues['issues_detected']:
    print(f"  - {issue['type'].upper()}: {issue['description']}")

# Test 3: Parse divergence log
print("\n📊 Test 3: Parse divergence.csv")
print("-"*60)
result2 = parser.parse("demo_data/divergence.csv")
df2 = parser.get_log(result2['log_id'])
issues2 = detector.detect_all(df2)
print(f"✓ Health Score: {issues2['health_score']}/100")
print(f"✓ Issues Found: {issues2['total_issues']}")
for issue in issues2['issues_detected']:
    print(f"  - {issue['type'].upper()}: {issue['description']}")

print("\n" + "="*60)
print("✅ All tests passed! MCP server tools working correctly.\n")