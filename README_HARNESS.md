# Claude Test Harness for DEX Valuation Study

## 🚀 Overview

The Claude Test Harness is an AI-powered testing framework that demonstrates automated test generation, execution, failure diagnosis, and code repair using Claude AI. This harness showcases how AI can accelerate development workflows by automatically generating comprehensive test suites and fixing code issues.

## ✨ Key Features

- **🧪 Intelligent Test Generation**: Claude analyzes source code to generate comprehensive test suites
- **🔍 Automatic Failure Detection**: Executes tests and captures detailed failure information
- **🔧 Self-Healing Code**: Claude generates patches for failing tests
- **✅ Validation Loop**: Re-runs tests to confirm fixes work
- **📊 Performance Benchmarking**: Includes performance and memory profiling
- **🐳 Containerized Environment**: Fully Dockerized for consistency
- **📓 Interactive Jupyter Demo**: Step-by-step notebook demonstration

## 📋 Prerequisites

- Docker installed and running
- Anthropic API key (for Claude integration)
- 4GB+ available RAM
- Python 3.11+ (for local development)

## 🎯 Quick Start

### 1. Set up your environment

```bash
# Clone the repository if you haven't already
git clone <repository-url>
cd dex-valuation-study

# Set your Anthropic API key
export ANTHROPIC_API_KEY="your-api-key-here"
```

### 2. Run the automated demo

```bash
# Build and run the complete demo
make demo
```

This single command will:
1. Build the Docker container
2. Execute the Claude test harness notebook
3. Generate comprehensive tests
4. Run tests and identify failures
5. Generate and apply patches
6. Re-validate the fixes
7. Save results to `notebooks/claude_test_harness_executed.ipynb`

### 3. Explore the results

After the demo completes, you'll find:
- **Executed notebook**: `notebooks/claude_test_harness_executed.ipynb`
- **Generated tests**: `tests/generated/`
- **Applied patches**: `patches/`
- **Coverage report**: `htmlcov/index.html`

## 🛠️ Usage Options

### Interactive Jupyter Mode

Run the notebook interactively to step through the process:

```bash
make run
# Open browser to http://localhost:8888
# Navigate to notebooks/claude_test_harness.ipynb
```

### Run Tests Only

Execute the generated test suite:

```bash
make test
```

### Docker Compose (Advanced)

For multi-service deployment:

```bash
# Start all services
docker-compose up -d

# Run with specific profiles
docker-compose --profile test up     # Include test runner
docker-compose --profile docs up     # Include documentation server
docker-compose --profile monitoring up  # Include performance monitoring
```

### Shell Access

Debug or explore the container:

```bash
make shell
```

## 📚 How It Works

### 1. Test Generation Phase

The harness analyzes your Python modules and generates comprehensive test suites:

```python
harness = ClaudeTestHarness('../src/validation.py')
tests = harness.generate_tests()
```

Claude generates tests covering:
- Happy path scenarios
- Edge cases and boundaries
- Error conditions
- Type validation
- Performance constraints

### 2. Test Execution Phase

Tests are executed with detailed failure capture:

```python
results = harness.run_tests(tests)
# Returns: {passed: 8, failed: 2, errors: 0, failures: [...]}
```

### 3. Auto-Patching Phase

For each failure, Claude analyzes the error and generates a fix:

```python
for failure in results['failures']:
    patch = harness.generate_patch(failure)
    harness.apply_patch(patch)
```

### 4. Validation Phase

Re-run tests to confirm all fixes work:

```python
final_results = harness.validate_fix()
# All tests should now pass
```

## 🏗️ Architecture

```
dex-valuation-study/
├── notebooks/
│   └── claude_test_harness.ipynb    # Main demo notebook
├── src/                              # Source code to test
│   ├── validation.py
│   ├── preprocessing.py
│   ├── evaluation.py
│   └── data_collection.py
├── tests/
│   └── generated/                    # Claude-generated tests
├── patches/                          # Applied fixes
├── Dockerfile                        # Container definition
├── docker-compose.yml                # Multi-service setup
├── Makefile                          # Automation commands
└── requirements-dev.txt              # Testing dependencies
```

## 🔧 Configuration

### Environment Variables

- `ANTHROPIC_API_KEY`: Your Claude API key (required for AI features)
- `PYTHONPATH`: Set to `/app` in container
- `JUPYTER_ENABLE_LAB`: Enable JupyterLab interface

### Docker Health Checks

The container includes health checks that verify:
- Python packages are installed correctly
- Critical dependencies are available
- Services are responsive

### Performance Targets

Default performance benchmarks:
- Test generation: <5 seconds per module
- Test execution: <30 seconds total
- Patch generation: <10 seconds per fix
- Coverage target: >90%

## 📊 Example Output

### Test Generation Results
```
Generated Tests Summary:
- Module: validation.py
- Test functions: 12
- Edge cases covered: 15
- Performance tests: 3
```

### Failure Analysis
```
Test Results:
✅ Passed: 8
❌ Failed: 2
⚠️ Errors: 0

Failures:
1. test_zero_values - AssertionError
2. test_negative_handling - ValueError
```

### Auto-Patch Success
```
Patches Applied: 2
Re-validation: ✅ All tests pass
Coverage: 94.5%
```

## 🧪 Test Categories

The harness generates multiple test categories:

### Unit Tests
- Individual function testing
- Input/output validation
- Type checking

### Integration Tests
- Module interaction
- Data flow validation
- Pipeline testing

### Performance Tests
- Execution time benchmarks
- Memory usage profiling
- Scalability testing

### Regression Tests
- Previous bug prevention
- Backward compatibility

## 🐛 Troubleshooting

### Common Issues

#### 1. API Key Not Set
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

#### 2. Port Already in Use
```bash
# Change port in docker-compose.yml or use:
docker stop $(docker ps -q)
```

#### 3. Build Failures
```bash
# Clean and rebuild
make clean
make build
```

#### 4. Test Timeouts
Increase timeout in notebook cells or Makefile:
```python
results = harness.run_tests(tests, timeout=60)
```

### Debug Mode

Enable verbose output:
```python
harness = ClaudeTestHarness('../src/module.py', debug=True)
```

## 📈 Performance Optimization

### Docker Build Cache
The Dockerfile uses multi-stage builds to optimize caching:
- Base dependencies cached separately
- Application code added last
- Virtual environment reused

### Test Parallelization
Run tests in parallel using pytest-xdist:
```bash
pytest -n auto tests/
```

### Memory Management
- Temporary files cleaned automatically
- Backup files removed after validation
- Docker volumes for persistent data

## 🤝 Contributing

To extend the test harness:

1. **Add New Test Types**: Modify `generate_tests()` in the harness class
2. **Custom Patch Logic**: Extend `generate_patch()` method
3. **New Modules**: Add to `modules_to_test` list
4. **Performance Metrics**: Update `perf_targets` dictionary

## 📝 License

This test harness is part of the DEX Valuation Study project.

## 🔗 Related Documentation

- [Main Project README](README.md)
- [API Documentation](docs/api.md)
- [Claude API Reference](https://docs.anthropic.com/)

## 💡 Tips & Best Practices

1. **Start Small**: Test simple modules first (like validation.py)
2. **Review Patches**: Always review AI-generated patches before production use
3. **Version Control**: Commit before running auto-patch in production
4. **Monitor Resources**: Use `docker stats` to monitor container resources
5. **Cache Results**: Use docker-compose cache service for faster re-runs

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review generated test output in `tests/generated/`
3. Examine patches in `patches/` directory
4. Open an issue with error logs

---

**Made with ❤️ using Claude AI and Docker**