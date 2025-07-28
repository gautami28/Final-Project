#!/bin/bash

# EC2 Deployment Script for Stock Prediction AI Application
# Run this script on your EC2 instance

echo "🚀 Starting EC2 deployment for Stock Prediction AI..."

# Update system
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and pip
echo "🐍 Installing Python and pip..."
sudo apt-get install -y python3 python3-pip python3-venv

# Install nginx
echo "🌐 Installing nginx..."
sudo apt-get install -y nginx

# Install gunicorn
echo "🔧 Installing gunicorn..."
pip3 install gunicorn

# Create application directory
echo "📁 Setting up application directory..."
sudo mkdir -p /var/www/stock-prediction-ai
sudo chown $USER:$USER /var/www/stock-prediction-ai

# Copy application files (assuming you're in the project directory)
echo "📋 Copying application files..."
cp -r * /var/www/stock-prediction-ai/

# Create virtual environment
echo "🔧 Creating virtual environment..."
cd /var/www/stock-prediction-ai
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Create gunicorn service file
echo "⚙️ Creating systemd service..."
sudo tee /etc/systemd/system/stock-prediction-ai.service > /dev/null <<EOF
[Unit]
Description=Stock Prediction AI Gunicorn daemon
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/var/www/stock-prediction-ai
Environment="PATH=/var/www/stock-prediction-ai/venv/bin"
ExecStart=/var/www/stock-prediction-ai/venv/bin/gunicorn --workers 3 --bind unix:/var/www/stock-prediction-ai/stock-prediction-ai.sock app:app

[Install]
WantedBy=multi-user.target
EOF

# Configure nginx
echo "🌐 Configuring nginx..."
sudo tee /etc/nginx/sites-available/stock-prediction-ai > /dev/null <<EOF
server {
    listen 80;
    server_name _;

    location / {
        include proxy_params;
        proxy_pass http://unix:/var/www/stock-prediction-ai/stock-prediction-ai.sock;
    }

    location /static {
        alias /var/www/stock-prediction-ai/new_UI/static;
    }
}
EOF

# Enable nginx site
echo "🔗 Enabling nginx site..."
sudo ln -s /etc/nginx/sites-available/stock-prediction-ai /etc/nginx/sites-enabled
sudo rm -f /etc/nginx/sites-enabled/default

# Test nginx configuration
echo "✅ Testing nginx configuration..."
sudo nginx -t

# Start services
echo "🚀 Starting services..."
sudo systemctl start stock-prediction-ai
sudo systemctl enable stock-prediction-ai
sudo systemctl restart nginx

# Check status
echo "📊 Checking service status..."
sudo systemctl status stock-prediction-ai
sudo systemctl status nginx

echo "🎉 Deployment complete!"
echo "🌐 Your application should be available at: http://your-ec2-public-ip"
echo "📝 To check logs: sudo journalctl -u stock-prediction-ai" 