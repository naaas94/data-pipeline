#!/bin/bash

# Enterprise Data Pipeline Docker Deployment Script
# Clean and simple deployment for the unified Docker setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to build images
build_images() {
    print_status "Building Docker images..."
    
    # Build production image
    print_status "Building production image..."
    docker build --target production -t pipeline:prod .
    
    # Build development image
    print_status "Building development image..."
    docker build --target development -t pipeline:dev .
    
    print_success "All images built successfully"
}

# Function to start services
start_services() {
    local profile=${1:-""}
    local command="docker-compose up -d"
    
    if [ -n "$profile" ]; then
        command="docker-compose --profile $profile up -d"
        print_status "Starting services with profile: $profile"
    else
        print_status "Starting all services..."
    fi
    
    eval $command
    print_success "Services started successfully"
}

# Function to stop services
stop_services() {
    print_status "Stopping all services..."
    docker-compose down
    print_success "Services stopped successfully"
}

# Function to show logs
show_logs() {
    local service=${1:-"data-pipeline"}
    print_status "Showing logs for service: $service"
    docker-compose logs -f "$service"
}

# Function to show status
show_status() {
    print_status "Service status:"
    docker-compose ps
}

# Function to clean up
cleanup() {
    print_warning "This will remove all containers, networks, and volumes. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_status "Cleaning up Docker resources..."
        docker-compose down -v --remove-orphans
        docker system prune -f
        print_success "Cleanup completed"
    else
        print_status "Cleanup cancelled"
    fi
}

# Function to show help
show_help() {
    echo "Enterprise Data Pipeline Docker Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build              Build all Docker images"
    echo "  start              Start all services (production)"
    echo "  start-dev          Start development services"
    echo "  stop               Stop all services"
    echo "  restart            Restart all services"
    echo "  logs [SERVICE]     Show logs for a service (default: data-pipeline)"
    echo "  status             Show service status"
    echo "  cleanup            Remove all containers, networks, and volumes"
    echo "  help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build           # Build all images"
    echo "  $0 start           # Start production services"
    echo "  $0 start-dev       # Start development services"
    echo "  $0 logs kafka      # Show Kafka logs"
    echo "  $0 status          # Show all service status"
}

# Main script logic
case "${1:-help}" in
    "build")
        check_docker
        build_images
        ;;
    "start")
        check_docker
        start_services
        ;;
    "start-dev")
        check_docker
        start_services "development"
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        stop_services
        sleep 2
        start_services
        ;;
    "logs")
        show_logs "$2"
        ;;
    "status")
        show_status
        ;;
    "cleanup")
        cleanup
        ;;
    "help"|*)
        show_help
        ;;
esac
