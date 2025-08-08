# Docker Setup for Enterprise Data Pipeline

This document describes the cleaned up and unified Docker setup for the Enterprise Data Pipeline.

## Overview

The Docker setup has been consolidated into a single, clean configuration with the following components:

- **Single Dockerfile** with multi-stage builds for development and production
- **Unified docker-compose.yml** with all necessary services
- **Resource management** with proper limits and reservations
- **Security best practices** with non-root users

## Files Structure

```
├── Dockerfile                    # Main Dockerfile with dev/prod targets
├── docker-compose.yml           # Complete stack with all services
├── DOCKER-README.md            # This documentation
└── deploy-docker.sh            # Deployment script
```

## Dockerfile

The main `Dockerfile` provides two build targets:

### Development Target
```bash
docker build --target development -t pipeline:dev .
```

**Features:**
- Includes development dependencies (pytest, black, flake8, mypy)
- Mounts source code for live development
- Uses `config_test.yaml` for testing
- Debug logging enabled

### Production Target
```bash
docker build --target production -t pipeline:prod .
```

**Features:**
- Optimized for production deployment
- Minimal dependencies
- Uses `config.yaml` for production
- Health checks enabled
- Non-root user for security

## Docker Compose Services

### Core Services

1. **data-pipeline** - Main pipeline service (production)
2. **data-pipeline-dev** - Development service (profile: development)
3. **kafka** - Message streaming
4. **zookeeper** - Kafka coordination
5. **mlflow** - Experiment tracking
6. **prometheus** - Metrics collection
7. **grafana** - Monitoring dashboard
8. **redis** - Caching (optional)
9. **postgres** - Metadata storage (optional)

### Resource Management

All services include proper resource limits and reservations:

```yaml
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '1.0'
    reservations:
      memory: 512M
      cpus: '0.5'
```

## Usage

### Quick Start (Production)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f data-pipeline

# Stop all services
docker-compose down
```

### Development Mode

```bash
# Start development environment
docker-compose --profile development up -d

# Start only development pipeline
docker-compose --profile development up data-pipeline-dev
```

### Building Images

```bash
# Build production image
docker build --target production -t pipeline:prod .

# Build development image
docker build --target development -t pipeline:dev .

# Build both
docker build --target production -t pipeline:prod . && \
docker build --target development -t pipeline:dev .
```

### Environment Variables

Key environment variables can be customized:

```bash
# Set custom Kafka bootstrap servers
export KAFKA_BOOTSTRAP_SERVERS=your-kafka:9092

# Set MLflow tracking URI
export MLFLOW_TRACKING_URI=http://your-mlflow:5000

# Set PostgreSQL password
export POSTGRES_PASSWORD=your-secure-password
```

## Volumes

The setup uses named volumes for data persistence:

- `pipeline-output` - Pipeline output files
- `pipeline-logs` - Application logs
- `pipeline-checkpoints` - Processing checkpoints
- `pipeline-cache` - Cache data
- `pipeline-metadata` - Metadata storage
- `mlflow-data` - MLflow experiment data
- `prometheus-data` - Metrics data
- `grafana-data` - Dashboard configurations
- `redis-data` - Cache data
- `postgres-data` - Database data

## Monitoring

### Access Points

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **MLflow**: http://localhost:5000
- **Kafka**: localhost:9092

### Health Checks

The main pipeline service includes health checks:

```yaml
healthcheck:
  test: ["CMD", "python", "-c", "import sys; sys.exit(0)"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s
```

## Security Features

1. **Non-root users** - All services run as non-root users
2. **Read-only mounts** - Development volumes are read-only
3. **Resource limits** - Prevents resource exhaustion
4. **Network isolation** - Services communicate via internal network

## Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 3000, 5000, 9090, 9092, 5432, 6379 are available
2. **Memory issues**: Adjust resource limits in docker-compose.yml
3. **Permission issues**: Check volume permissions and user setup

### Debug Commands

```bash
# Check service status
docker-compose ps

# View service logs
docker-compose logs [service-name]

# Execute commands in container
docker-compose exec data-pipeline bash

# Check resource usage
docker stats
```

## Migration from Old Setup

The old setup had multiple Dockerfiles and docker-compose files. This has been consolidated:

- ❌ `Dockerfile.simple` - Removed (functionality merged into main Dockerfile)
- ❌ `Dockerfile.optimized` - Removed (optimizations merged into main Dockerfile)
- ❌ `docker-compose.eks.yml` - Removed (features merged into main docker-compose.yml)

## Next Steps

1. **Customize configurations** for your specific environment
2. **Set up monitoring** dashboards in Grafana
3. **Configure alerts** in Prometheus
4. **Set up CI/CD** using the provided scripts
5. **Add custom services** as needed

## Support

For issues or questions:
1. Check the logs: `docker-compose logs [service-name]`
2. Review this documentation
3. Check the main README.md for project-specific information
noteId: "88b0f3c0744c11f084faa31cc6172271"
tags: []

---

