# replit.nix - Nix packages for Replit environment
{ pkgs, ... }:

{
  # System dependencies
  deps = [
    # Docker and Docker Compose
    pkgs.docker
    pkgs.docker-compose

    # Python and tools
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.python311Packages.virtualenv

    # Database and cache
    pkgs.postgresql_15
    pkgs.redis

    # Monitoring tools
    pkgs.curl
    pkgs.jq

    # Build tools
    pkgs.gcc
    pkgs.gnumake

    # Git
    pkgs.git
  ];

  # Environment setup
  env = {
    DOCKER_HOST = "unix:///var/run/docker.sock";
    COMPOSE_HTTP_TIMEOUT = "120";
  };

  # Scripts to run after environment setup
  scripts = {
    setup = ''
      echo "Setting up VIT Network environment..."

      # Create necessary directories
      mkdir -p secrets logs models data

      # Create secret files if not exist
      if [ ! -f secrets/db_password.txt ]; then
        echo "vit_password_$(openssl rand -hex 8)" > secrets/db_password.txt
      fi

      if [ ! -f secrets/api_key.txt ]; then
        echo "vit_api_key_$(openssl rand -hex 16)" > secrets/api_key.txt
      fi

      # Copy .env if not exist
      if [ ! -f .env ]; then
        cp .env.example .env
      fi

      echo "Environment ready!"
    '';

    start = ''
      echo "Starting VIT Network..."
      docker-compose up -d
      echo "Services started! Access API at port 8000"
    '';

    stop = ''
      echo "Stopping VIT Network..."
      docker-compose down
    '';

    logs = ''
      docker-compose logs -f --tail=100
    '';

    status = ''
      docker-compose ps
    '';

    clean = ''
      docker-compose down -v
      docker system prune -af
    '';
  };
}