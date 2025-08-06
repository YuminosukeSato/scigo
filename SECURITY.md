# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Which versions are eligible for receiving such patches depends on the CVSS v3.0 Rating:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take the security of SciGo seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Please do NOT:
- Open a public GitHub issue for security vulnerabilities
- Post about the vulnerability on social media

### Please DO:
- Email us at: security@scigo.dev (or create a private security advisory on GitHub)
- Include the following information:
  - Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
  - Full paths of source file(s) related to the issue
  - Location of affected source code (tag/branch/commit or direct URL)
  - Step-by-step instructions to reproduce the issue
  - Proof-of-concept or exploit code (if possible)
  - Impact of the issue, including how an attacker might exploit it

### What to Expect
- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Resolution Timeline**: 
  - Critical: Within 7 days
  - High: Within 14 days
  - Medium: Within 30 days
  - Low: Within 90 days

### After Resolution
- We will notify you when the vulnerability has been fixed
- We will publicly disclose the vulnerability after the fix has been released
- We will credit you for the discovery (unless you prefer to remain anonymous)

## Security Best Practices for Users

### Input Validation
Always validate and sanitize input data before processing:
```go
// Good
if err := validateInput(data); err != nil {
    return fmt.Errorf("invalid input: %w", err)
}

// Bad
model.Fit(untrustedData, labels) // Direct use of untrusted data
```

### Resource Limits
Set appropriate resource limits to prevent DoS attacks:
```go
// Set maximum iterations
model.SetMaxIterations(1000)

// Set timeout for long-running operations
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()
```

### Error Handling
Never expose sensitive information in error messages:
```go
// Good
return fmt.Errorf("authentication failed")

// Bad
return fmt.Errorf("authentication failed for user %s with password %s", user, pass)
```

### Dependencies
Keep dependencies up to date:
```bash
# Check for vulnerabilities
go list -json -m all | nancy sleuth

# Update dependencies
go get -u ./...
go mod tidy
```

## Security Features in SciGo

### Built-in Protections
- **Panic Recovery**: All public APIs have panic recovery to prevent crashes
- **Input Validation**: Dimension checks and type validation on all inputs
- **Resource Limits**: Configurable limits on iterations and memory usage
- **Safe Defaults**: Conservative default parameters to prevent resource exhaustion

### Secure Coding Guidelines
We follow these guidelines in our codebase:
1. No use of `unsafe` package without thorough review
2. All array accesses are bounds-checked
3. No dynamic code execution (`eval`-like patterns)
4. Careful handling of user-supplied file paths
5. Proper cleanup of resources (defer close/cleanup)

## Vulnerability Disclosure Policy

We follow a coordinated disclosure process:

1. **Private Disclosure**: Security issues are first disclosed privately to give users time to upgrade
2. **Public Disclosure**: After patches are available, we publicly disclose the vulnerability
3. **CVE Assignment**: For significant vulnerabilities, we request CVE assignments
4. **Advisory Publication**: We publish security advisories on GitHub

## Automated Security Measures

### Continuous Security Scanning

We employ comprehensive automated security scanning:

1. **Vulnerability Detection**
   - `govulncheck`: Detects known vulnerabilities in Go code and dependencies
   - `Trivy`: Comprehensive vulnerability scanner for dependencies and containers
   - `Nancy`: Scans Go dependencies for known vulnerabilities

2. **Static Security Analysis**
   - `gosec`: Inspects source code for security problems
   - `staticcheck`: Advanced static analysis including security checks
   - `CodeQL`: GitHub's semantic code analysis for security vulnerabilities

3. **Supply Chain Security**
   - **SBOM Generation**: Automated Software Bill of Materials in CycloneDX format
   - **Dependency Updates**: Dependabot configured for automatic updates
   - **License Compliance**: Automated license checking for all dependencies

### Security Scanning Schedule

- **On Every Commit**: Basic security checks
- **On Pull Requests**: Full security suite
- **Daily**: Vulnerability database updates and scanning
- **Weekly**: Comprehensive dependency audit

### Running Security Scans Locally

```bash
# Install all security tools
make install-security-tools

# Run complete security scan
make security-scan

# Individual security checks
make vuln-check        # Check for known vulnerabilities
make gosec-scan        # Security-focused static analysis
make static-analysis   # General static analysis including security
make dependency-check  # Check dependencies for vulnerabilities
make generate-sbom     # Generate Software Bill of Materials
make security-report   # Generate comprehensive security report
```

## Security Audit History

| Date | Auditor | Scope | Results |
|------|---------|-------|---------|
| 2025-08-06 | Automated | govulncheck, gosec, SBOM | Initial security framework established |
| TBD | Third-party | Full audit | Planned |

## Contact

For security concerns, please contact:
- Primary: security@scigo.dev
- GitHub Security Advisories: [Create private advisory](https://github.com/YuminosukeSato/scigo/security/advisories/new)

## Acknowledgments

We would like to thank the following individuals for responsibly disclosing security issues:
- (Your name could be here!)

---

*This security policy is adapted from best practices recommended by the [OpenSSF](https://openssf.org/) and [GitHub Security Lab](https://securitylab.github.com/).*