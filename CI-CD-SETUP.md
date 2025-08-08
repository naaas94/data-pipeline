# CI/CD Setup

This document explains the streamlined CI/CD pipeline for the Enterprise Data Pipeline.

## Overview

The original CI/CD was overly complex with multiple jobs, security scanning, performance testing, and AWS ECS deployment. We've streamlined it to focus on the essentials:

1. **Testing** - Run unit tests and code quality checks
2. **Building** - Create Docker image and push to registry
3. **Deployment** - Simple deployment script for manual or automated deployment

## Files

### `.github/workflows/ci.yml`
- **Purpose**: Main CI/CD workflow
- **Triggers**: Push to main/develop branches, pull requests
- **Jobs**:
  - `test`: Runs tests, linting, and code formatting
  - `build`: Builds and pushes Docker image (only on main branch)

### `Dockerfile`
- **Purpose**: Streamlined Docker image for the pipeline
- **Features**: Single-stage build, security best practices, non-root user

### `deploy.sh`
- **Purpose**: Manual deployment script
- **Usage**: `./deploy.sh` or `REGISTRY=your-registry ./deploy.sh`

## How to Use

### 1. Automatic CI/CD (Recommended)
- Push to `main` or `develop` branches
- GitHub Actions will automatically:
  - Run tests and quality checks
  - Build Docker image (on main branch)
  - Push to GitHub Container Registry

### 2. Manual Deployment
```bash
# Make deploy script executable
chmod +x deploy.sh

# Deploy locally
./deploy.sh

# Deploy to specific registry
REGISTRY=ghcr.io/your-org ./deploy.sh
```

### 3. Local Development
```bash
# Build image locally
docker build -f Dockerfile.simple -t enterprise-data-pipeline:latest .

# Run tests
docker run --rm enterprise-data-pipeline:latest python -m pytest tests/ -v

# Run pipeline
docker run --rm enterprise-data-pipeline:latest
```

## Benefits of Streamlined Approach

1. **Clear Workflow**: Linear, easy-to-follow process
2. **Faster Execution**: Fewer jobs, quicker feedback
3. **Maintainable**: Reduced complexity, easier debugging
4. **Flexible**: Can be extended as needed
5. **Reliable**: Fewer potential failure points

## Migration from Complex CI/CD

### What We Removed:
- ❌ Multi-Python version matrix testing
- ❌ Security vulnerability scanning (Trivy)
- ❌ Performance testing
- ❌ Integration tests with PostgreSQL
- ❌ AWS ECS deployment
- ❌ Complex multi-stage Dockerfile
- ❌ Release automation

### What We Kept:
- ✅ Unit testing with pytest
- ✅ Code formatting with black
- ✅ Linting with flake8
- ✅ Coverage reporting
- ✅ Docker image building
- ✅ Container registry pushing
- ✅ Security best practices (non-root user)

## Future Enhancements

When you're ready to add more features:

1. **Security Scanning**: Add Trivy or Snyk
2. **Integration Tests**: Add database testing
3. **Performance Testing**: Add load testing
4. **Deployment**: Add Kubernetes or cloud deployment
5. **Monitoring**: Add health checks and metrics

## Troubleshooting

### Common Issues:

1. **Tests failing**: Check `tests/` directory and requirements
2. **Docker build failing**: Ensure all files are present
3. **Registry push failing**: Check GitHub secrets and permissions
4. **Deployment script failing**: Ensure Docker is running and you have permissions

### Debug Commands:
```bash
# Check workflow status
gh run list

# View workflow logs
gh run view <run-id>

# Test locally
docker build -f Dockerfile.simple -t test .
docker run --rm test python -m pytest tests/ -v
```

## Next Steps

1. **Review the streamlined workflow** in `.github/workflows/ci.yml`
2. **Test the deployment script** with `./deploy.sh`
3. **Update your README** to reflect the new CI/CD approach
4. **Consider removing** the old complex CI/CD files when you're confident

This streamlined approach provides a solid foundation that can be extended as your needs grow!
noteId: "5e2cf1c0744311f0ab4ead757d2417ec"
tags: []

---

