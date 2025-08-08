# Enterprise Data Pipeline Docker Deployment Script (PowerShell)
# Clean and simple deployment for the unified Docker setup

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Function to check if Docker is running
function Test-Docker {
    try {
        docker info | Out-Null
        Write-Success "Docker is running"
        return $true
    }
    catch {
        Write-Error "Docker is not running. Please start Docker and try again."
        return $false
    }
}

# Function to build images
function Build-Images {
    Write-Status "Building Docker images..."
    
    # Build production image
    Write-Status "Building production image..."
    docker build --target production -t pipeline:prod .
    
    # Build development image
    Write-Status "Building development image..."
    docker build --target development -t pipeline:dev .
    
    Write-Success "All images built successfully"
}

# Function to start services
function Start-Services {
    param([string]$Profile = "")
    
    if ($Profile) {
        Write-Status "Starting services with profile: $Profile"
        docker-compose --profile $Profile up -d
    }
    else {
        Write-Status "Starting all services..."
        docker-compose up -d
    }
    
    Write-Success "Services started successfully"
}

# Function to stop services
function Stop-Services {
    Write-Status "Stopping all services..."
    docker-compose down
    Write-Success "Services stopped successfully"
}

# Function to show logs
function Show-Logs {
    param([string]$Service = "data-pipeline")
    Write-Status "Showing logs for service: $Service"
    docker-compose logs -f $Service
}

# Function to show status
function Show-Status {
    Write-Status "Service status:"
    docker-compose ps
}

# Function to clean up
function Clear-DockerResources {
    Write-Warning "This will remove all containers, networks, and volumes. Are you sure? (y/N)"
    $response = Read-Host
    
    if ($response -match "^[yY][eE][sS]|[yY]$") {
        Write-Status "Cleaning up Docker resources..."
        docker-compose down -v --remove-orphans
        docker system prune -f
        Write-Success "Cleanup completed"
    }
    else {
        Write-Status "Cleanup cancelled"
    }
}

# Function to show help
function Show-Help {
    Write-Host "Enterprise Data Pipeline Docker Deployment Script" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\deploy-docker.ps1 [COMMAND]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  build              Build all Docker images"
    Write-Host "  start              Start all services (production)"
    Write-Host "  start-dev          Start development services"
    Write-Host "  stop               Stop all services"
    Write-Host "  restart            Restart all services"
    Write-Host "  logs [SERVICE]     Show logs for a service (default: data-pipeline)"
    Write-Host "  status             Show service status"
    Write-Host "  cleanup            Remove all containers, networks, and volumes"
    Write-Host "  help               Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\deploy-docker.ps1 build           # Build all images"
    Write-Host "  .\deploy-docker.ps1 start           # Start production services"
    Write-Host "  .\deploy-docker.ps1 start-dev       # Start development services"
    Write-Host "  .\deploy-docker.ps1 logs kafka      # Show Kafka logs"
    Write-Host "  .\deploy-docker.ps1 status          # Show all service status"
}

# Main script logic
switch ($Command) {
    "build" {
        if (Test-Docker) {
            Build-Images
        }
    }
    "start" {
        if (Test-Docker) {
            Start-Services
        }
    }
    "start-dev" {
        if (Test-Docker) {
            Start-Services -Profile "development"
        }
    }
    "stop" {
        Stop-Services
    }
    "restart" {
        Stop-Services
        Start-Sleep -Seconds 2
        Start-Services
    }
    "logs" {
        Show-Logs $args[0]
    }
    "status" {
        Show-Status
    }
    "cleanup" {
        Clear-DockerResources
    }
    default {
        Show-Help
    }
}
